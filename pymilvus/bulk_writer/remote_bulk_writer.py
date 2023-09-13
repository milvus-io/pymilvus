# Copyright (C) 2019-2023 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

import logging
import sys
from pathlib import Path
from typing import Any, Optional

from minio import Minio
from minio.error import S3Error

from pymilvus.orm.schema import CollectionSchema

from .constants import (
    DEFAULT_BUCKET_NAME,
    MB,
    BulkFileType,
)
from .local_bulk_writer import LocalBulkWriter

logger = logging.getLogger("remote_bulk_writer")
logger.setLevel(logging.DEBUG)


class RemoteBulkWriter(LocalBulkWriter):
    class ConnectParam:
        def __init__(
            self,
            bucket_name: str = DEFAULT_BUCKET_NAME,
            endpoint: Optional[str] = None,
            access_key: Optional[str] = None,
            secret_key: Optional[str] = None,
            secure: bool = False,
            session_token: Optional[str] = None,
            region: Optional[str] = None,
            http_client: Any = None,
            credentials: Any = None,
        ):
            self._bucket_name = bucket_name
            self._endpoint = endpoint
            self._access_key = access_key
            self._secret_key = secret_key
            self._secure = (secure,)
            self._session_token = (session_token,)
            self._region = (region,)
            self._http_client = (http_client,)  # urllib3.poolmanager.PoolManager
            self._credentials = (credentials,)  # minio.credentials.Provider

    def __init__(
        self,
        schema: CollectionSchema,
        remote_path: str,
        connect_param: ConnectParam,
        segment_size: int = 512 * MB,
        file_type: BulkFileType = BulkFileType.NPY,
    ):
        local_path = Path(sys.argv[0]).resolve().parent.joinpath("bulk_writer")
        super().__init__(schema, str(local_path), segment_size, file_type)
        self._remote_path = Path("/").joinpath(remote_path).joinpath(super().uuid)
        self._connect_param = connect_param
        self._client = None
        self._get_client()
        self._remote_files = []
        logger.info(f"Remote buffer writer initialized, target path: {self._remote_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object):
        super().__exit__(exc_type, exc_val, exc_tb)
        # remove the temp folder "bulk_writer"
        if Path(self._local_path).parent.exists() and not any(
            Path(self._local_path).parent.iterdir()
        ):
            Path(self._local_path).parent.rmdir()
            logger.info(f"Delete empty directory '{Path(self._local_path).parent}'")

    def _get_client(self):
        try:
            if self._client is None:

                def arg_parse(arg: Any):
                    return arg[0] if isinstance(arg, tuple) else arg

                self._client = Minio(
                    endpoint=arg_parse(self._connect_param._endpoint),
                    access_key=arg_parse(self._connect_param._access_key),
                    secret_key=arg_parse(self._connect_param._secret_key),
                    secure=arg_parse(self._connect_param._secure),
                    session_token=arg_parse(self._connect_param._session_token),
                    region=arg_parse(self._connect_param._region),
                    http_client=arg_parse(self._connect_param._http_client),
                    credentials=arg_parse(self._connect_param._credentials),
                )
            else:
                return self._client
        except Exception as err:
            logger.error(f"Failed to connect MinIO/S3, error: {err}")
            raise

    def append_row(self, row: dict, **kwargs):
        super().append_row(row, **kwargs)

    def commit(self, **kwargs):
        super().commit(call_back=self._upload)

    def _remote_exists(self, file: str) -> bool:
        try:
            minio_client = self._get_client()
            minio_client.stat_object(bucket_name=self._connect_param._bucket_name, object_name=file)
        except S3Error as err:
            if err.code == "NoSuchKey":
                return False
            self._throw(f"Failed to stat MinIO/S3 object, error: {err}")
        return True

    def _local_rm(self, file: str):
        try:
            Path(file).unlink()
            parent_dir = Path(file).parent
            if not any(Path(parent_dir).iterdir()):
                Path(parent_dir).rmdir()
                logger.info(f"Delete empty directory '{parent_dir!s}'")
        except Exception:
            logger.warning(f"Failed to delete local file: {file}")

    def _upload(self, file_list: list):
        remote_files = []
        try:
            logger.info("Prepare to upload files")
            minio_client = self._get_client()
            found = minio_client.bucket_exists(self._connect_param._bucket_name)
            if not found:
                self._throw(f"MinIO bucket '{self._connect_param._bucket_name}' doesn't exist")

            for file_path in file_list:
                ext = Path(file_path).suffix
                if ext not in {".json", ".npy"}:
                    continue

                relative_file_path = str(file_path).replace(str(super().data_path), "")
                minio_file_path = str(
                    Path.joinpath(self._remote_path, relative_file_path.lstrip("/"))
                ).lstrip("/")

                if self._remote_exists(minio_file_path):
                    logger.info(
                        f"Remote file '{minio_file_path}' already exists, will overwrite it"
                    )

                minio_client.fput_object(
                    bucket_name=self._connect_param._bucket_name,
                    object_name=minio_file_path,
                    file_path=file_path,
                )
                logger.info(f"Upload file '{file_path}' to '{minio_file_path}'")

                remote_files.append(str(minio_file_path))
                self._local_rm(file_path)
        except Exception as e:
            self._throw(f"Failed to call MinIO/S3 api, error: {e}")

        logger.info(f"Successfully upload files: {file_list}")
        self._remote_files.append(remote_files)
        return remote_files

    @property
    def data_path(self):
        return self._remote_path

    @property
    def batch_files(self):
        return self._remote_files

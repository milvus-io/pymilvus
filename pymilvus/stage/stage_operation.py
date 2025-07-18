import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import urllib3
from minio import Minio
from minio.error import S3Error

from pymilvus.stage.file_utils import FileUtils
from pymilvus.stage.stage_restful import apply_stage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StageOperation:
    def __init__(
        self, cloud_endpoint: str, api_key: str, stage_name: str, path: Optional[str] = None
    ):
        self.cloud_endpoint = cloud_endpoint
        self.api_key = api_key
        self.stage_name = stage_name
        self.local_file_paths = []
        self.total_bytes = 0
        self.stage_info = {}
        self.path = path
        self._convert_dir_path()
        self._client = None
        self._refresh_stage_and_client()

    def _convert_dir_path(self):
        input_path = self.path
        if not input_path or input_path.endswith("/"):
            self.path = input_path
            return
        self.path = input_path + "/"

    def _refresh_stage_and_client(self):
        logger.info("refreshing stage info...")
        response = apply_stage(self.cloud_endpoint, self.stage_name, self.path, self.api_key)
        self.stage_info = response.json()["data"]
        logger.info("stage info refreshed.")

        creds = self.stage_info["credentials"]
        http_client = urllib3.PoolManager(maxsize=100)
        self._client = Minio(
            endpoint=self.stage_info["endpoint"],
            access_key=creds["tmpAK"],
            secret_key=creds["tmpSK"],
            session_token=creds["sessionToken"],
            region=self.stage_info["region"],
            secure=True,
            http_client=http_client,
        )
        logger.info("storage client refreshed")

    def _validate_size(self):
        file_size_total = self.total_bytes
        file_size_limit = self.stage_info["condition"]["maxContentLength"]
        if file_size_total > file_size_limit:
            error_message = (
                f"localFileTotalSize {file_size_total} exceeds "
                f"the maximum contentLength limit {file_size_limit} defined in the condition."
                f"If you want to upload larger files, please contact us to lift the restriction."
            )
            raise ValueError(error_message)

    def upload_file_to_stage(self, local_path: str, concurrency: int = 20):
        self.local_file_paths, self.total_bytes = FileUtils.process_local_path(local_path)
        self._validate_size()

        file_count = len(self.local_file_paths)
        logger.info(
            f"begin to upload file to stage, localDirOrFilePath:{local_path}, fileCount:{file_count} to stageName:{self.stage_name}, stagePath:{self.path}"
        )
        start_time = time.time()

        uploaded_bytes = 0
        root_path = Path(local_path).resolve()
        uploaded_bytes_lock = threading.Lock()

        def _upload_task(file_path: str, idx: int, root_path: Path):
            nonlocal uploaded_bytes
            path_obj = Path(file_path).resolve()
            if root_path.is_file():
                relative_path = path_obj.name
            else:
                relative_path = path_obj.relative_to(root_path).as_posix()

            object_name = f"{self.stage_info['uploadPath']}{relative_path}"
            file_start_time = time.time()
            try:
                size = Path(file_path).stat().st_size
                logger.info(f"uploading file, fileName:{file_path}, size:{size} bytes")
                self._put_object(file_path, object_name)
                with uploaded_bytes_lock:
                    uploaded_bytes += size
                percent = uploaded_bytes / self.total_bytes * 100
                elapsed = time.time() - file_start_time
                logger.info(
                    f"Uploaded file, {idx + 1}/{file_count}: {file_path}, elapsed:{elapsed} s, {uploaded_bytes}/{self.total_bytes} bytes, progress: {percent:.2f}%"
                )
            except S3Error as e:
                logger.error(f"Failed to upload {file_path}: {e!s}")
                raise

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for i, file_path in enumerate(self.local_file_paths):
                futures.append(executor.submit(_upload_task, file_path, i, root_path))
            for f in futures:
                f.result()  # wait for all

        total_elapsed = time.time() - start_time
        logger.info(
            f"All files in {local_path} uploaded to stage, "
            f"stageName:{self.stage_info['stageName']}, stagePath: {self.path}, "
            f"totalFileCount:{file_count}, totalFileSize:{self.total_bytes}, cost time:{total_elapsed}s"
        )
        return {"stageName": self.stage_info["stageName"], "path": self.path}

    def _put_object(self, file_path: str, remote_file_path: str):
        expire_time_str = self.stage_info["credentials"]["expireTime"]
        expire_time = datetime.fromisoformat(expire_time_str.replace("Z", "+00:00"))

        now = datetime.now(timezone.utc)
        if now > expire_time:
            self._refresh_stage_and_client()

        self._upload_with_retry(file_path, remote_file_path)

    def _upload_with_retry(self, file_path: str, object_name: str, max_retries: int = 3):
        attempt = 0
        while attempt < max_retries:
            try:
                self._client.fput_object(
                    bucket_name=self.stage_info["bucketName"],
                    object_name=object_name,
                    file_path=file_path,
                )
                break
            except Exception as e:
                attempt += 1
                logger.warning(f"Attempt {attempt} failed to upload {file_path}: {e}")
                self._refresh_stage_and_client()

                if attempt == max_retries:
                    error_message = f"Upload failed after {max_retries} attempts"
                    raise RuntimeError(error_message) from e

                time.sleep(5)

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

import json
import logging
from typing import List, Optional, Union

import requests

from pymilvus.exceptions import MilvusException

logger = logging.getLogger(__name__)


def _http_headers(api_key: str):
    return {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) "
        "Chrome/17.0.963.56 Safari/535.11",
        "Accept": "application/json",
        "Accept-Encodin": "gzip,deflate,sdch",
        "Accept-Languag": "en-US,en;q=0.5",
        "Authorization": f"Bearer {api_key}",
    }


def _throw(msg: str):
    logger.error(msg)
    raise MilvusException(message=msg)


def _handle_response(url: str, res: json):
    inner_code = res["code"]
    if inner_code != 0:
        inner_message = res["message"]
        _throw(f"Failed to request url: {url}, code: {inner_code}, message: {inner_message}")


def _post_request(
    url: str,
    api_key: str,
    params: {},
    timeout: int = 20,
    verify: Optional[Union[bool, str]] = True,
    cert: Optional[Union[str, tuple]] = None,
    **kwargs,
) -> requests.Response:
    """Send a POST request with 1-way / 2-way optional certificate validation

    Args:
        url (str): The endpoint URL
        api_key (str): API key for authentication
        params (dict): JSON parameters for the request
        timeout (int): Timeout for the request
        verify (bool, str, optional): Either a boolean, to verify the server's TLS certificate
             or a string, which must be server's certificate path. Defaults to `True`.
        cert (str, tuple, optional): if String, path to ssl client cert file.
                                     if Tuple, ('cert', 'key') pair.

    Returns:
        requests.Response: Response object.
    """
    try:
        resp = requests.post(
            url=url,
            headers=_http_headers(api_key),
            json=params,
            timeout=timeout,
            verify=verify,
            cert=cert,
            **kwargs,
        )
        if resp.status_code != 200:
            _throw(f"Failed to post url: {url}, status code: {resp.status_code}")
        else:
            return resp
    except Exception as err:
        _throw(f"Failed to post url: {url}, error: {err}")


def _get_request(
    url: str, api_key: str, params: {}, timeout: int = 20, **kwargs
) -> requests.Response:
    try:
        resp = requests.get(
            url=url, headers=_http_headers(api_key), params=params, timeout=timeout, **kwargs
        )
        if resp.status_code != 200:
            _throw(f"Failed to get url: {url}, status code: {resp.status_code}")
        else:
            return resp
    except Exception as err:
        _throw(f"Failed to get url: {url}, error: {err}")


## bulkinsert RESTful api wrapper
def bulk_import(
    url: str,
    collection_name: str,
    db_name: str = "",
    files: Optional[List[List[str]]] = None,
    object_url: str = "",
    cluster_id: str = "",
    api_key: str = "",
    access_key: str = "",
    secret_key: str = "",
    stage_name: str = "",
    data_paths: [List[List[str]]] = None,
    verify: Optional[Union[bool, str]] = True,
    cert: Optional[Union[str, tuple]] = None,
    **kwargs,
) -> requests.Response:
    """call bulkinsert restful interface to import files

    Args:
        url (str): url of the server
        collection_name (str): name of the target collection
        db_name (str): name of target database
        partition_name (str): name of the target partition
        files (list of list of str): The files that contain the data to import.
             A sub-list contains a single JSON or Parquet file, or a set of Numpy files.
        object_url (str): The URL of the object to import.
             This URL should be accessible to the S3-compatible
             object storage service, such as AWS S3, GCS, Azure blob storage.
        cluster_id (str): id of a milvus instance(for cloud)
        api_key (str): API key to authenticate your requests.
        access_key (str): access key to access the object storage
        secret_key (str): secret key to access the object storage
        stage_name (str): name of the stage to import
        data_paths (list of list of str): The paths of files that contain the data to import.
        verify (bool, str, optional): Either a boolean, to verify the server's TLS certificate
             or a string, which must be server's certificate path. Defaults to `True`.
        cert (str, tuple, optional): if String, path to ssl client cert file.
                                     if Tuple, ('cert', 'key') pair.

    Returns:
        response of the restful interface
    """
    request_url = url + "/v2/vectordb/jobs/import/create"

    partition_name = kwargs.pop("partition_name", "")
    params = {
        "collectionName": collection_name,
        "dbName": db_name,
        "partitionName": partition_name,
        "files": files,
        "objectUrl": object_url,
        "clusterId": cluster_id,
        "accessKey": access_key,
        "secretKey": secret_key,
        "stageName": stage_name,
        "dataPaths": data_paths,
    }

    options = kwargs.pop("options", {})
    if isinstance(options, dict):
        params["options"] = options

    resp = _post_request(
        url=request_url, api_key=api_key, params=params, verify=verify, cert=cert, **kwargs
    )
    _handle_response(request_url, resp.json())
    return resp


def get_import_progress(
    url: str,
    job_id: str,
    cluster_id: str = "",
    api_key: str = "",
    verify: Optional[Union[bool, str]] = True,
    cert: Optional[Union[str, tuple]] = None,
    **kwargs,
) -> requests.Response:
    """get job progress

    Args:
        url (str): url of the server
        job_id (str): a job id
        cluster_id (str): id of a milvus instance(for cloud)
        api_key (str): API key to authenticate your requests.
        verify (bool, str, optional): Either a boolean, to verify the server's TLS certificate
             or a string, which must be server's certificate path. Defaults to `True`.
        cert (str, tuple, optional): if String, path to ssl client cert file.
                                     if Tuple, ('cert', 'key') pair.

    Returns:
        response of the restful interface
    """
    request_url = url + "/v2/vectordb/jobs/import/describe"

    params = {
        "jobId": job_id,
        "clusterId": cluster_id,
    }

    resp = _post_request(
        url=request_url, api_key=api_key, params=params, verify=verify, cert=cert, **kwargs
    )
    _handle_response(request_url, resp.json())
    return resp


def list_import_jobs(
    url: str,
    collection_name: str = "",
    cluster_id: str = "",
    api_key: str = "",
    page_size: int = 10,
    current_page: int = 1,
    verify: Optional[Union[bool, str]] = True,
    cert: Optional[Union[str, tuple]] = None,
    **kwargs,
) -> requests.Response:
    """list jobs in a cluster

    Args:
        url (str): url of the server
        collection_name (str): name of the target collection
        cluster_id (str): id of a milvus instance(for cloud)
        api_key (str): API key to authenticate your requests.
        page_size (int): pagination size
        current_page (int): pagination
        verify (bool, str, optional): Either a boolean, to verify the server's TLS certificate
             or a string, which must be server's certificate path. Defaults to `True`.
        cert (str, tuple, optional): if String, path to ssl client cert file.
                                     if Tuple, ('cert', 'key') pair.

    Returns:
        response of the restful interface
    """
    request_url = url + "/v2/vectordb/jobs/import/list"

    params = {
        "collectionName": collection_name,
        "clusterId": cluster_id,
        "pageSize": page_size,
        "currentPage": current_page,
    }

    resp = _post_request(
        url=request_url, api_key=api_key, params=params, verify=verify, cert=cert, **kwargs
    )
    _handle_response(request_url, resp.json())
    return resp

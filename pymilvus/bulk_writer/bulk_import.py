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
from urllib.parse import urlparse

import requests

from pymilvus.exceptions import MilvusException

logger = logging.getLogger("bulk_import")
logger.setLevel(logging.DEBUG)


def _http_headers(api_key: str):
    return {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) "
        "Chrome/17.0.963.56 Safari/535.11",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encodin": "gzip,deflate,sdch",
        "Accept-Languag": "en-US,en;q=0.5",
        "Authorization": f"Bearer {api_key}",
    }


def _throw(msg: str):
    logger.error(msg)
    raise MilvusException(message=msg)


def _handle_response(url: str, res: json):
    inner_code = res["code"]
    if inner_code != 200:
        inner_message = res["message"]
        _throw(f"Failed to request url: {url}, code: {inner_code}, message: {inner_message}")


def _post_request(
    url: str, api_key: str, params: {}, timeout: int = 20, **kwargs
) -> requests.Response:
    try:
        resp = requests.post(
            url=url, headers=_http_headers(api_key), json=params, timeout=timeout, **kwargs
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
    api_key: str,
    object_url: str,
    access_key: str,
    secret_key: str,
    cluster_id: str,
    collection_name: str,
    **kwargs,
) -> requests.Response:
    """call bulkinsert restful interface to import files

    Args:
        url (str): url of the server
        object_url (str): data files url
        access_key (str): access key to access the object storage
        secret_key (str): secret key to access the object storage
        cluster_id (str): id of a milvus instance(for cloud)
        collection_name (str): name of the target collection

    Returns:
        json: response of the restful interface
    """
    up = urlparse(url)
    if up.scheme.startswith("http"):
        request_url = f"{url}/v1/vector/collections/import"
    else:
        request_url = f"https://{url}/v1/vector/collections/import"

    params = {
        "objectUrl": object_url,
        "accessKey": access_key,
        "secretKey": secret_key,
        "clusterId": cluster_id,
        "collectionName": collection_name,
    }

    resp = _post_request(url=request_url, api_key=api_key, params=params, **kwargs)
    _handle_response(url, resp.json())
    return resp


def get_import_progress(
    url: str, api_key: str, job_id: str, cluster_id: str, **kwargs
) -> requests.Response:
    """get job progress

    Args:
        url (str): url of the server
        job_id (str): a job id
        cluster_id (str): id of a milvus instance(for cloud)

    Returns:
        json: response of the restful interface
    """
    up = urlparse(url)
    if up.scheme.startswith("http"):
        request_url = f"{url}/v1/vector/collections/import/get"
    else:
        request_url = f"https://{url}/v1/vector/collections/import/get"

    params = {
        "jobId": job_id,
        "clusterId": cluster_id,
    }

    resp = _get_request(url=request_url, api_key=api_key, params=params, **kwargs)
    _handle_response(url, resp.json())
    return resp


def list_import_jobs(
    url: str, api_key: str, cluster_id: str, page_size: int, current_page: int, **kwargs
) -> requests.Response:
    """list jobs in a cluster

    Args:
        url (str): url of the server
        cluster_id (str): id of a milvus instance(for cloud)
        page_size (int): pagination size
        current_page (int): pagination

    Returns:
        json: response of the restful interface
    """
    up = urlparse(url)
    if up.scheme.startswith("http"):
        request_url = f"{url}/v1/vector/collections/import/list"
    else:
        request_url = f"https://{url}/v1/vector/collections/import/list"

    params = {
        "clusterId": cluster_id,
        "pageSize": page_size,
        "currentPage": current_page,
    }

    resp = _get_request(url=request_url, api_key=api_key, params=params, **kwargs)
    _handle_response(url, resp.json())
    return resp

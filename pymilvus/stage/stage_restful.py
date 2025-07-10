import json
import logging
from typing import Optional, Union

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


## bulkinsert RESTful api wrapper
def apply_stage(
    url: str,
    stage_name: str,
    path: str,
    api_key: str = "",
    verify: Optional[Union[bool, str]] = True,
    cert: Optional[Union[str, tuple]] = None,
    **kwargs,
) -> requests.Response:
    """call bulkinsert restful interface to import files

    Args:
        url (str): url of the server
        stage_name (str): name of the stage
        path (str): the path of the stage
        api_key (str): API key to authenticate your requests.
        verify (bool, str, optional): Either a boolean, to verify the server's TLS certificate
             or a string, which must be server's certificate path. Defaults to `True`.
        cert (str, tuple, optional): if String, path to ssl client cert file.
                                     if Tuple, ('cert', 'key') pair.

    Returns:
        response of the restful interface
    """
    request_url = url + "/v2/stages/apply"

    params = {
        "stageName": stage_name,
        "path": path,
    }

    options = kwargs.pop("options", {})
    if isinstance(options, dict):
        params["options"] = options

    resp = _post_request(
        url=request_url, api_key=api_key, params=params, verify=verify, cert=cert, **kwargs
    )
    _handle_response(request_url, resp.json())
    return resp

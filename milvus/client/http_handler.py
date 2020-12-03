import functools
import json
import logging
import struct

import requests as rq
import ujson

from .exceptions import NotConnectError
from .types import Status
from .utils import set_uri

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


LOGGER = logging.getLogger(__name__)


def handle_error(returns=tuple()):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            nonlocal returns
            try:
                return func(self, *args, **kwargs)
            except rq.exceptions.Timeout:
                status = Status(Status.UNEXPECTED_ERROR, message='Request timeout')
                return returns if not returns else tuple([status]) + returns
            except json.decoder.JSONDecodeError as e:
                status = Status(Status.UNEXPECTED_ERROR, message=str(e))
                return returns if not returns else tuple([status]) + returns

        return wrapper

    return decorator


class HttpHandler:

    def __init__(self, host=None, port=None, **kwargs):
        self._status = None

        _uri = kwargs.get("uri", None)

        self._uri = (host or port or _uri) and set_uri(uri=_uri)
        self._max_retry = kwargs.get("max_retry", 3)

    def __enter__(self):
        self.ping()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def status(self):
        return self._status

    def ping(self, timeout=10):
        if self._uri is None:
            self._uri = set_uri(None)
        logging.info("Connecting server {}", self._uri)
        retry = self._max_retry
        try:
            while retry > 0:
                try:
                    rq.get(self._uri + "/state", timeout=timeout)
                    return True
                except (rq.exceptions.Timeout,
                        rq.exceptions.TooManyRedirects,
                        rq.exceptions.RequestException):
                    retry -= 1
                    if retry > 0:
                        continue

                    raise
        except:
            LOGGER.error("Cannot connect server {}", self._uri)
            raise NotConnectError("Cannot get server status")

        LOGGER.info("Connected server {}", self._uri)
        # try:
        #     js = response.json()
        #     return Status(js["code"], js["message"])
        # except ValueError:
        #     pass
        #
        # return Status(Status.UNEXPECTED_ERROR, "Error occurred when parse response.")

    def set_hook(self, **kwargs):
        pass

    def connect(self, host, port, uri, timeout):
        if self.connected():
            return Status(message=f"You have already connected {self._uri} !",
                          code=Status.CONNECT_FAILED)

        if (host or port or uri) or not self._uri:
            # if self._uri:
            #     return Status(message="The server address is set as {}, "
            #                           "you cannot connect other server".format(self._uri),
            #                   code=Status.CONNECT_FAILED)
            # else:
            self._uri = self._set_uri(host, port, uri=uri)

        self._status = self.ping()
        return self._status

    def connected(self):
        return False if not (self._status and self._status.OK()) else self.ping()

    def disconnect(self):
        self._status = None
        return Status()

    def _set_config(self, cmd, timeout):
        if cmd.startswith("set_config"):
            cmd_node = cmd.split(" ")
            config_node = cmd_node[1].split(".")
            request = {
                config_node[0]: {
                    config_node[1]: cmd_node[2]
                }
            }

            url = self._uri + "/system/config"
            payload = ujson.dumps(request)
            response = rq.put(url, data=payload, timeout=timeout)
            if response.status_code == 200:
                js = response.json()
                return Status(), js["message"]
            if response.status_code == 400:
                js = response.json()
                return Status(js["code"], js["message"]), None

            return Status(Status.UNEXPECTED_ERROR, response.reason)

    def _get_config(self, cmd, timeout):
        if cmd.startswith("get_config"):
            cmd_node = cmd.split(" ")
            config_node = cmd_node[1].split(".")

            url = self._uri + "/system/config"
            response = rq.get(url, timeout=timeout)
            if response.status_code == 200:
                js = response.json()
                rc_parent = js.get(config_node[0], None)
                if rc_parent is None:
                    return Status(Status.UNEXPECTED_ERROR,
                                  "Config {} not supported".format(cmd_node[1]))
                rc_child = rc_parent.get(config_node[1], None)
                if rc_child is None:
                    return Status(Status.UNEXPECTED_ERROR,
                                  "Config {} not supported".format(cmd_node[1]))

                return Status(), rc_child
                # return Status(), js["message"]

            if response.status_code == 400:
                js = response.json()
                return Status(js["code"], js["message"]), None

            return Status(Status.UNEXPECTED_ERROR, response.reason)

    @handle_error(returns=(None,))
    def cmd(self, cmd, timeout=10):
        if cmd.startswith("get_config"):
            return self._get_config(cmd, timeout)
        if cmd.startswith("set_config"):
            return self._set_config(cmd, timeout)

        url = self._uri + "/system/{}".format(cmd)

        response = rq.get(url, timeout=timeout)

        js = response.json()
        if response.status_code == 200:
            return Status(), js["reply"]

        return Status(code=js["code"], message=js["message"]), None

    def server_version(self, timeout):
        return self._cmd("version", timeout)

    def server_status(self, timeout):
        return self._cmd("status", timeout)

    @handle_error()
    def create_collection(self, collection_name, dimension, index_file_size,
                          metric_type, params=None, timeout=10):
        table_param = {
            "collection_name": collection_name,
            "dimension": dimension,
            "index_file_size": index_file_size,
            "metric_type": metric_type
        }

        data = ujson.dumps(table_param)
        url = self._uri + "/collections"

        try:
            response = rq.post(url, data=data)
            if response.status_code == 201:
                return Status(message='Create table successfully!')

            js = response.json()
            return Status(js["code"], js["message"])
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e))

    @handle_error(returns=(False,))
    def has_collection(self, table_name, timeout):
        url = self._uri + "/collections/" + table_name
        try:
            response = rq.get(url=url, timeout=timeout)
            if response.status_code == 200:
                return Status(), True

            if response.status_code == 404:
                return Status(), False

            js = response.json()
            return Status(js["code"], js["message"]), False
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e))

    @handle_error(returns=(None,))
    def count_collection(self, table_name, timeout):
        url = self._uri + "/collections/{}".format(table_name)

        try:
            response = rq.get(url, timeout=timeout)
            js = response.json()

            if response.status_code == 200:
                return Status(), js["count"]

            return Status(js["code"], js["message"]), None
        except Exception as e:
            return Status(1, message=str(e)), None

    @handle_error(returns=(None,))
    def describe_collection(self, table_name, timeout):
        url = self._uri + "/collections/{}".format(table_name)

        response = rq.get(url, timeout=timeout)

        if response.status_code >= 500:
            return Status(1, response.reason), None

        js = response.json()
        if response.status_code == 200:
            return Status(message='Describe table successfully!'), js

        return Status(js["code"], js["message"]), None

    @handle_error(returns=([],))
    def show_collections(self, timeout):
        url = self._uri + "/collections"

        response = rq.get(url, params={"offset": 0, "page_size": 0}, timeout=timeout)
        if response.status_code != 200:
            return Status(1, response.reason), []

        js = response.json()
        count = js["count"]

        response = rq.get(url, params={"offset": 0, "page_size": count}, timeout=timeout)
        if response.status_code != 200:
            return Status(1, response.reason), []

        tables = []
        js = response.json()

        for table in js["collections"]:
            tables.append(table["collection_name"])

        return Status(), tables

    @handle_error(returns=(None,))
    def show_collection_info(self, table_name, timeout=10):
        url = self._uri + "/collections/{}?info=stat".format(table_name)

        response = rq.get(url, timeout=timeout)
        if response.status_code == 200:
            return Status(), response.json()

        if response.status_code == 404:
            return Status(1, "Collection not found"), None

        if response.text:
            result = response.json()
            return Status(result["code"], result["message"]), None

        return Status(1, "Response is empty"), None

    @handle_error()
    def preload_collection(self, table_name, timeout):
        url = self._uri + "/system/task"
        params = {"load": {"collection_name": table_name}}
        data = ujson.dumps(params)


        response = rq.put(url, data=data, timeout=timeout)

        if response.status_code == 200:
            return Status(message="Load successfuly")

        js = response.json()
        return Status(code=js["code"], message=js["message"])

    @handle_error()
    def reload_segments(self, collection_name, segment_ids, timeout=10):
        raise NotImplementedError("Not implemented in http server")

    @handle_error()
    def drop_collection(self, table_name, timeout):
        url = self._uri + "/collections/" + table_name
        response = rq.delete(url, timeout=timeout)
        if response.status_code == 204:
            return Status(message="Delete successfully!")

        js = response.json()
        return Status(js["code"], js["message"])

    @handle_error(returns=([],))
    def bulk_insert(self, table_name, records, ids, partition_tag, params, timeout, **kwargs):
        url = self._uri + "/collections/{}/vectors".format(table_name)

        data_dict = dict()
        if ids:
            data_dict["ids"] = list(map(str, ids))
        if partition_tag:
            data_dict["partition_tag"] = partition_tag

        if isinstance(records[0], bytes):
            vectors = [struct.unpack(str(len(r)) + 'B', r) for r in records]
            data_dict["vectors"] = vectors
        else:
            data_dict["vectors"] = records

        data = ujson.dumps(data_dict)

        headers = {"Content-Type": "application/json"}

        response = rq.post(url, data=data, headers=headers)
        js = response.json()

        if response.status_code == 201:
            ids = [int(item) for item in list(js["ids"])]
            return Status(message='Add vectors successfully!'), ids

        return Status(js["code"], js["message"]), []

    @handle_error(returns=(None,))
    def get_vectors_by_ids(self, collection_name, ids, timeout):
        status, table_schema = self.describe_collection(collection_name, timeout)
        if not status.OK():
            return status, None
        metric = table_schema.metric_type

        # bin_vector = metric in list(MetricType.__members__.values())[3:]

        url = self._uri + "/collections/{}/vectors".format(collection_name)
        ids_list = list(map(str, ids))
        query_ids = ",".join(ids_list)
        url = url + "?ids=" + query_ids
        response = rq.get(url, timeout=timeout)
        result = response.json()

        if response.status_code == 200:
            vectors = result["vectors"]
            if not list(vectors):
                return Status(), []

            vector_results = []
            for vector_res in vectors:
                vector = list(vector_res["vector"])
                if metric.startswith("BIN"):
                    vector_results.append(bytes(vector))
                else:
                    vector_results.append(vector)
                    # return Status(),
            return Status(), vector_results

        return Status(result["code"], result["message"]), None

    @handle_error(returns=(None,))
    def get_vector_ids(self, table_name, segment_name, timeout):
        url = self._uri + \
              "/collections/{}/segments/{}/ids?page_size=1000000".format(table_name, segment_name)
        response = rq.get(url, timeout=timeout)
        result = response.json()

        if response.status_code == 200:
            return Status(), list(map(int, result["ids"]))

        return Status(result["code"], result["message"]), None

    @handle_error()
    def create_index(self, table_name, index_type, index_params, timeout):
        url = self._uri + "/collections/{}/indexes".format(table_name)

        request = dict()
        request["index_type"] = index_type
        request["params"] = index_params
        data = ujson.dumps(request)
        headers = {"Content-Type": "application/json"}

        response = rq.post(url, data=data, headers=headers, timeout=timeout)
        js = response.json()

        return Status(js["code"], js["message"])

    @handle_error(returns=(None,))
    def describe_index(self, table_name, timeout):
        url = self._uri + "/collections/{}/indexes".format(table_name)

        response = rq.get(url, timeout=timeout)

        if response.status_code >= 500:
            return Status(1,
                          "Unexpected error.\n\tStatus code : {}, reason : {}"
                          .format(response.status_code, response.reason))

        js = response.json()

        if response.status_code == 200:
            return Status(), js

        return Status(js["code"], js["message"]), None

    @handle_error()
    def drop_index(self, table_name, timeout):
        url = self._uri + "/collections/{}/indexes".format(table_name)

        response = rq.delete(url)

        if response.status_code == 204:
            return Status()

        js = response.json()
        return Status(js["code"], js["message"])

    @handle_error()
    def create_partition(self, table_name, partition_tag, timeout=10):
        url = self._uri + "/collections/{}/partitions".format(table_name)

        data = ujson.dumps({"partition_tag": partition_tag})
        headers = {"Content-Type": "application/json"}

        response = rq.post(url, data=data, headers=headers, timeout=timeout)
        if response.status_code == 201:
            return Status()

        js = response.json()
        return Status(js["code"], js["message"])

    @handle_error(returns=([],))
    def show_partitions(self, table_name, timeout):
        url = self._uri + "/collections/{}/partitions".format(table_name)
        query_data = {"offset": 0, "page_size": 100}

        response = rq.get(url, params=query_data, timeout=timeout)
        if response.status_code >= 500:
            return Status(1,
                          "Unexpected error. Status code : 500, reason: {}"
                          .format(response.reason)), None

        js = response.json()
        if response.status_code == 200:
            partition_list = [item["partition_tag"] for item in js["partitions"]]

            return Status(), partition_list

        return Status(js["code"], js["message"]), []

    @handle_error(returns=(False,))
    def has_partition(self, collection_name, tag, timeout=30):
        url = self._uri + "/collections/{}/partitions".format(collection_name)

        response = rq.get(url, timeout=timeout)
        if response.status_code == 200:
            result = response.json()
            if result["count"] > 0:
                partitions = [p["partition_tag"] for p in list(result["partitions"])]
                return Status(), tag in partitions
            return Status(), False

        js = response.json()
        return Status(js["code"], js["message"]), False

    @handle_error()
    def drop_partition(self, collection_name, partition_tag, timeout=10):
        url = self._uri + "/collections/{}/partitions".format(collection_name)
        request = {
            "partition_tag": partition_tag
        }
        payload = ujson.dumps(request)

        response = rq.delete(url, data=payload, timeout=timeout)
        if response.status_code == 204:
            return Status()

        js = response.json()
        return Status(js["code"], js["message"])

    @handle_error(returns=(None,))
    def search(self, collection_name, top_k, query_records, partition_tags=None,
               search_params=None, timeout=None, **kwargs):
        url = self._uri + "/collections/{}/vectors".format(collection_name)

        search_body = dict()
        if partition_tags:
            search_body["partition_tags"] = partition_tags
        search_body["topk"] = top_k
        search_body["params"] = search_params

        if isinstance(query_records[0], bytes):
            vectors = [struct.unpack(str(len(r)) + 'B', r) for r in query_records]
            search_body["vectors"] = vectors
        else:
            vectors = query_records
            search_body["vectors"] = vectors

        data = ujson.dumps({"search": search_body})
        headers = {"Content-Type": "application/json"}

        response = rq.put(url, data, headers=headers)

        if response.status_code == 200:
            pass
            # return Status(), TopKQueryResult2(response)

        js = response.json()
        return Status(js["code"], js["message"]), None

    @handle_error(returns=(None,))
    def search_by_ids(self, collection_name, ids, top_k, partition_tags=None,
                      search_params=None, timeout=None, **kwargs):
        url = self._uri + "/collections/{}/vectors".format(collection_name)
        body_dict = dict()
        body_dict["topk"] = top_k
        body_dict["ids"] = list(map(str, ids))
        if partition_tags:
            body_dict["partition_tags"] = partition_tags
        if search_params:
            body_dict["params"] = search_params

        data = ujson.dumps({"search": body_dict})
        headers = {"Content-Type": "application/json"}

        response = rq.put(url, data, headers=headers, timeout=timeout)

        if response.status_code == 200:
            pass
            # return Status(), TopKQueryResult2(response)

        js = response.json()
        return Status(js["code"], js["message"]), None

    @handle_error(returns=(None,))
    def search_in_files(self, collection_name, file_ids, query_records,
                        top_k, search_params, timeout, **kwargs):
        url = self._uri + "/collections/{}/vectors".format(collection_name)

        body_dict = dict()
        body_dict["topk"] = top_k
        body_dict["file_ids"] = list(map(str, file_ids))
        body_dict["params"] = search_params

        if isinstance(query_records[0], bytes):
            vectors = [struct.unpack(str(len(r)) + 'B', r) for r in query_records]
            body_dict["vectors"] = vectors
        else:
            vectors = query_records
            body_dict["vectors"] = vectors

        data = ujson.dumps({"search": body_dict})
        headers = {"Content-Type": "application/json"}

        response = rq.put(url, data, headers=headers, timeout=timeout)

        if response.status_code == 200:
            pass
            # return Status(), TopKQueryResult2(response)

        js = response.json()
        return Status(js["code"], js["message"]), None

    @handle_error()
    def delete_by_id(self, table_name, id_array, timeout=None):
        url = self._uri + "/collections/{}/vectors".format(table_name)
        headers = {"Content-Type": "application/json"}
        ids = list(map(str, id_array))
        request = {"delete": {"ids": ids}}

        response = rq.put(url, data=ujson.dumps(request), headers=headers, timeout=timeout)
        result = response.json()
        return Status(result["code"], result["message"])

    @handle_error()
    def flush(self, table_name_array, *kwargs):
        url = self._uri + "/system/task"
        headers = {"Content-Type": "application/json"}
        request = {"flush": {"collection_names": table_name_array}}

        response = rq.put(url, ujson.dumps(request), headers=headers)
        result = response.json()
        return Status(result["code"], result["message"])

    @handle_error()
    def compact(self, table_name, timeout):
        url = self._uri + "/system/task"
        headers = {"Content-Type": "application/json"}
        request = {"compact": {"collection_name": table_name}}

        response = rq.put(url, ujson.dumps(request), headers=headers)
        result = response.json()
        return Status(result["code"], result["message"])

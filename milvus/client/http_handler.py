import copy
import json
import logging
import ujson
from urllib.parse import urlparse
import struct

import requests as rq

from .abstract import ConnectIntf, IndexParam, TableSchema, TopKQueryResult2, PartitionParam
from .check import is_legal_host, is_legal_port
from .exceptions import NotConnectError, ParamError
from .types import Status, IndexType, MetricType

from ..settings import DefaultConfig as config

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

IndexValue2NameMap = {
    IndexType.INVALID: "INVALID",
    IndexType.FLAT: "FLAT",
    IndexType.IVFLAT: "IVFFLAT",
    IndexType.IVF_SQ8: "IVFSQ8",
    IndexType.IVF_SQ8H: "IVFSQ8H",
    IndexType.IVF_PQ: "IVFPQ",
    IndexType.HNSW: "HNSW"
}

IndexName2ValueMap = {
    "INVALID": IndexType.INVALID,
    "FLAT": IndexType.FLAT,
    "IVFFLAT": IndexType.IVFLAT,
    "IVFSQ8": IndexType.IVF_SQ8,
    "IVFSQ8H": IndexType.IVF_SQ8H,
    "IVFPQ": IndexType.IVF_PQ,
    "HNSW": IndexType.HNSW
}

MetricValue2NameMap = {
    MetricType.L2: "L2",
    MetricType.IP: "IP",
    MetricType.HAMMING: "HAMMING",
    MetricType.JACCARD: "JACCARD",
    MetricType.TANIMOTO: "TANIMOTO",
}

MetricName2ValueMap = {
    "L2": MetricType.L2,
    "IP": MetricType.IP,
    "HAMMING": MetricType.HAMMING,
    "JACCARD": MetricType.JACCARD,
    "TANIMOTO": MetricType.TANIMOTO,
}


class HttpHandler(ConnectIntf):

    def __init__(self, host=None, port=None, **kwargs):
        self._status = None

        _uri = kwargs.get("uri", None)

        self._uri = (host or port or _uri) and self._set_uri(host, port, uri=_uri)

    def __enter__(self):
        self.ping()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _set_uri(self, host, port, **kwargs):
        """
        Set server network address

        :raises: ParamError

        """
        if host is not None:
            _port = port if port is not None else config.HTTP_PORT
            _host = host
        elif port is None:
            try:
                _uri = kwargs.get("uri", None)
                _uri = urlparse(_uri) if _uri else urlparse(config.HTTP_URI)
                _host = _uri.hostname
                _port = _uri.port
            except (AttributeError, ValueError, TypeError) as e:
                raise ParamError("uri is illegal: {}".format(e))
        else:
            raise ParamError("Param is not complete. Please invoke as follow:\n"
                             "\t(host = ${HOST}, port = ${PORT})\n"
                             "\t(uri = ${URI})\n")

        if not is_legal_host(_host) or not is_legal_port(_port):
            raise ParamError("host or port is illeagl")

        return "http://{}:{}".format(str(_host), str(_port))

    @property
    def status(self):
        return self._status

    def ping(self, timeout=1):
        try:
            if self._uri is None:
                self._uri = self._set_uri(None, None)
            response = rq.get(self._uri + "/state", timeout=timeout)
        except:
            raise NotConnectError("Cannot get server status")
        try:
            js = response.json()
            return Status(js["code"], js["message"])
        except ValueError:
            pass

        return Status(Status.UNEXPECTED_ERROR, "Error occurred when parse response.")

    def set_hook(self, **kwargs):
        pass

    def connect(self, host, port, uri, timeout):
        if self.connected():
            return Status(message="You have already connected {} !".format(self._uri),
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

    def create_table(self, param, timeout):

        table_param = copy.deepcopy(param)

        table_param['metric_type'] = MetricValue2NameMap.get(param['metric_type'], None)
        data = ujson.dumps(table_param)

        try:
            response = rq.post(self._uri + "/tables", data=data)
            if response.status_code == 201:
                return Status(message='Create table successfully!')

            js = response.json()
            return Status(Status(js["code"], js["message"]))
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout')
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e))

    def has_table(self, table_name, timeout):
        url = self._uri + "/tables/" + table_name
        try:
            response = rq.get(url=url, timeout=timeout)
            if response.status_code == 200:
                return Status(), True

            if response.status_code == 404:
                return Status(), False

            js = response.json()
            return Status(Status(js["code"], js["message"])), False
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout')
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e))

    def count_table(self, table_name, timeout):
        url = self._uri + "/tables/{}".format(table_name)

        try:
            response = rq.get(url, timeout=timeout)
            js = response.json()

            if response.status_code == 200:
                return Status(), js["count"]

            return Status(Status(js["code"], js["message"])), None
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout'), None
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e)), None

    def describe_table(self, table_name, timeout):
        url = self._uri + "/tables/{}".format(table_name)

        try:
            response = rq.get(url, timeout=timeout)

            if response.status_code >= 500:
                return Status(Status.UNEXPECTED_ERROR, response.reason), None

            js = response.json()
            if response.status_code == 200:
                metric_map = dict()
                _ = [metric_map.update({i.name: i.value}) for i in MetricType if i.value > 0]

                table = TableSchema(
                    table_name=js["table_name"],
                    dimension=js["dimension"],
                    index_file_size=js["index_file_size"],
                    metric_type=metric_map[js["metric_type"]])

                return Status(message='Describe table successfully!'), table

            return Status(Status(js["code"], js["message"])), None
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout')
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e))

    def show_tables(self, timeout):
        url = self._uri + "/tables"

        try:
            response = rq.get(url, params={"offset": 0, "page_size": 0}, timeout=timeout)
            if response.status_code != 200:
                return Status(Status.UNEXPECTED_ERROR, response.reason), []

            js = response.json()
            count = js["count"]

            response = rq.get(url, params={"offset": 0, "page_size": count}, timeout=timeout)
            if response.status_code != 200:
                return Status(Status.UNEXPECTED_ERROR, response.reason), []

            tables = []
            js = response.json()

            for table in js["tables"]:
                tables.append(table["table_name"])

            return Status(), tables
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout'), []
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e)), []

    def drop_table(self, table_name, timeout):
        url = self._uri + "/tables/" + table_name
        try:
            response = rq.delete(url, timeout=timeout)
            if response.status_code == 204:
                return Status(message="Delete successfully!")

            js = response.json()
            return Status(js["code"], js["message"])
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout')
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e))

    def insert(self, table_name, records, ids, partition_tag, timeout, **kwargs):
        url = self._uri + "/tables/{}/vectors".format(table_name)

        data_dict = dict()
        if ids:
            data_dict["ids"] = ids
        if partition_tag:
            data_dict["tag"] = partition_tag

        if isinstance(records[0], bytes):
            vectors = [struct.unpack(str(len(r)) + 'B', r) for r in records]
            data_dict["records_bin"] = vectors
        else:
            vectors = records
            data_dict["records"] = vectors

        data = ujson.dumps(data_dict)

        headers = {"Content-Type": "application/json"}

        try:
            response = rq.post(url, data=data, headers=headers)
            js = response.json()

            if response.status_code == 201:
                ids = [int(item) for item in list(js["ids"])]
                return Status(message='Add vectors successfully!'), ids

            return Status(Status(js["code"], js["message"])), []
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout'), []
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e)), []

    def search(self, table_name, top_k, nprobe, query_records, partition_tags=None, **kwargs):
        url = self._uri + "/tables/{}/vectors".format(table_name)

        body_dict = dict()
        if partition_tags:
            body_dict["tags"] = partition_tags
        body_dict["topk"] = top_k
        body_dict["nprobe"] = nprobe

        if isinstance(query_records[0], bytes):
            vectors = [struct.unpack(str(len(r)) + 'B', r) for r in query_records]
            body_dict["records_bin"] = vectors
        else:
            vectors = query_records
            body_dict["records"] = vectors

        data = ujson.dumps(body_dict)
        headers = {"Content-Type": "application/json"}

        try:
            response = rq.put(url, data, headers=headers)

            if response.status_code == 200:
                return Status(), TopKQueryResult2(response)

            js = response.json()
            return Status(Status(js["code"], js["message"])), None
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout'), None
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e)), None

    def search_in_files(self, table_name, file_ids, query_records, top_k, nprobe=16, **kwargs):
        url = self._uri + "/tables/{}/vectors".format(table_name)

        body_dict = dict()
        body_dict["topk"] = top_k
        body_dict["nprobe"] = nprobe
        body_dict["file_ids"] = file_ids

        if isinstance(query_records[0], bytes):
            vectors = [struct.unpack(str(len(r)) + 'B', r) for r in query_records]
            body_dict["records_bin"] = vectors
        else:
            vectors = query_records
            body_dict["records"] = vectors

        data = ujson.dumps(body_dict)
        headers = {"Content-Type": "application/json"}

        try:
            response = rq.put(url, data, headers=headers)

            if response.status_code == 200:
                return Status(), TopKQueryResult2(response)

            js = response.json()
            return Status(Status(js["code"], js["message"])), None
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout'), None
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e)), None

    def search_by_id(self, table_name, top_k, nprobe, id_array, partition_tag_array):
        return Status(Status.UNEXPECTED_ERROR, "Not uncompleted"), None

    def create_index(self, table_name, index, timeout):
        url = self._uri + "/tables/{}/indexes".format(table_name)
        try:
            index["index_type"] = IndexValue2NameMap.get(index["index_type"])
            data = ujson.dumps(index)

            headers = {"Content-Type": "application/json"}
            response = rq.post(url, data=data, headers=headers)

            js = response.json()

            return Status(js["code"], js["message"])
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout')
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e))

    def server_version(self, timeout):
        return self._cmd("version", timeout)

    def server_status(self, timeout):
        return self._cmd("status", timeout)

    def preload_table(self, table_name, timeout):
        url = self._uri + "/system/task"
        params = {"action": "load", "target": table_name}

        try:
            response = rq.get(url, params=params, timeout=timeout)

            if response.status_code == 200:
                return Status(message="Load successfuly")

            js = response.json()
            return Status(code=js["code"], message=js["message"])
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout')
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e))

    def describe_index(self, table_name, timeout):
        url = self._uri + "/tables/{}/indexes".format(table_name)

        try:
            response = rq.get(url, timeout=timeout)

            if response.status_code >= 500:
                return Status(Status.UNEXPECTED_ERROR,
                              "Unexpected error.\n\tStatus code : {}, reason : {}"
                              .format(response.status_code, response.reason))

            js = response.json()

            if response.status_code == 200:
                index_type = IndexName2ValueMap.get(js["index_type"])
                return Status(), IndexParam(table_name, index_type, js["nlist"])

            return Status(js["code"], js["message"]), None
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout'), None
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e)), None

    def drop_index(self, table_name, timeout):
        url = self._uri + "/tables/{}/indexes".format(table_name)
        try:
            response = rq.delete(url)

            if response.status_code == 204:
                return Status()

            js = response.json()
            return Status(js["code"], js["message"])
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout')
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e))

    def _cmd(self, cmd, timeout=10):
        url = self._uri + "/system/{}".format(cmd)

        try:
            response = rq.get(url, timeout=timeout)

            js = response.json()
            if response.status_code == 200:
                return Status(), js["reply"]

            return Status(code=js["code"], message=js["message"]), None
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout'), None
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e)), None

    def create_partition(self, table_name, partition_name, partition_tag, timeout=10):
        url = self._uri + "/tables/{}/partitions".format(table_name)

        try:
            data = ujson.dumps({"partition_name": partition_name, "partition_tag": partition_tag})
            headers = {"Content-Type": "application/json"}

            response = rq.post(url, data=data, headers=headers)
            if response.status_code == 201:
                return Status()

            js = response.json()
            return Status(Status(js["code"], js["message"]))
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout')
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e))

    def show_partitions(self, table_name, timeout):
        url = self._uri + "/tables/{}/partitions".format(table_name)
        query_data = {"offset": 0, "page_size": 100}

        try:
            response = rq.get(url, params=query_data, timeout=timeout)
            if response.status_code >= 500:
                return Status(Status.UNEXPECTED_ERROR,
                              "Unexpected error. Status code : 500, reason: {}"
                              .format(response.reason)), None

            js = response.json()
            if response.status_code == 200:
                partition_list = \
                    [PartitionParam(table_name, item["partition_name"], item["partition_tag"])
                     for item in js["partitions"]]

                return Status(), partition_list

            return Status(Status(js["code"], js["message"])), []
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout'), []
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e)), []

    def drop_partition(self, table_name, partition_tag, timeout=10):
        url = self._uri + "/tables/{}/partitions/{}".format(table_name, partition_tag)

        try:
            response = rq.delete(url, timeout=timeout)
            if response.status_code == 204:
                return Status()

            js = response.json()
            return Status(Status(js["code"], js["message"]))
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout')
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e))

    def delete_by_id(self, table_name, id_array, timeout=None):
        return Status(Status.UNEXPECTED_ERROR, "Not uncompleted")

    def flush(self, table_name_array):
        return Status(Status.UNEXPECTED_ERROR, "Not uncompleted")

    def compact(self, table_name, timeout):
        return Status(Status.UNEXPECTED_ERROR, "Not uncompleted")

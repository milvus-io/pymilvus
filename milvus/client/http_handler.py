import requests as rq
import json
from urllib.parse import urlparse

from .abstract import ConnectIntf, IndexParam, TableSchema, TopKQueryResult2, PartitionParam
from .types import Status
from .exceptions import ParamError
from .check import is_legal_host, is_legal_port
from .exceptions import NotConnectError
from .prepare import *

from ..settings import DefaultConfig as config


class HttpHandler(ConnectIntf):

    def __init__(self, host=None, port=None, **kwargs):
        self._uri = kwargs.get("uri", "127.0.0.1:19121")
        self.status = Status()

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

        self._uri = "http://{}:{}".format(str(_host), str(_port))

    def ping(self, timeout=1):
        try:
            response = rq.get(self._uri + "/state", timeout=timeout)
        except:
            raise NotConnectError("Cannot get server status")
        try:
            js = response.json()
            return Status(js["code"], js["message"])
        except ValueError as e:
            pass

        return Status(Status.UNEXPECTED_ERROR, "Error occurred when parse response.")

    def set_hook(self, **kwargs):
        pass

    def connect(self, host, port, uri, timeout):
        self._set_uri(host, port, uri=uri)
        return self.ping()

    def connected(self):
        return self.ping()

    def disconnect(self):
        pass

    def create_table(self, param, timeout):
        _ = Prepare.table_schema(param)

        try:
            response = rq.post(self._uri + "/tables", data=param)
            if 200 == response.status_code:
                return Status(message='Create table successfully!')

            js = response.json()
            return Status(Status(js["code"], js["message"]))
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout')
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e))

    def has_table(self, table_name, timeout):
        _ = Prepare.table_name(table_name)

        url = self._uri + "/tables/" + table_name
        try:
            response = rq.get(url=url, timeout=timeout)
            if 200 == response.status_code:
                return Status(), True

            js = response.json()
            return Status(Status(js["code"], js["message"])), False
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout')
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e))

    def count_table(self, table_name, timeout):
        _ = Prepare.table_name(table_name)

        url = self._uri + "/tables/{}".format(table_name)

        try:
            response = rq.get(url, timeout=timeout)
            js = response.json()

            if 200 == response.status_code:
                return Status(), js["count"]

            return Status(Status(js["code"], js["message"])), None
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout'), None
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e)), None

    def describe_table(self, table_name, timeout):
        _ = Prepare.table_name(table_name)

        url = self._uri + "/tables/{}".format(table_name)

        try:
            response = rq.get(url, timeout=timeout)
            js = response.json()
            if 200 == response.status_code:
                table = TableSchema(
                    table_name=js["table_name"],
                    dimension=js["dimension"],
                    index_file_size=js["index_file_size"],
                    metric_type=js["metric_type"]
                )

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
            if 200 != response.status_code:
                return Status(Status.UNEXPECTED_ERROR, "Error"), []

            js = response.json()
            count = js["count"]

            response = rq.get(url, params={"offset": 0, "page_size": count}, timeout=timeout)
            if 200 != response.status_code:
                return Status(Status.UNEXPECTED_ERROR, "Error"), []

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
            js = response.json()
            if 204 == response.status_code:
                return Status(message="Delete successfully!")

            return Status(js["code"], js["message"])
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout')
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e))

    def insert(self, table_name, records, ids, partition_tag, timeout, **kwargs):
        _ = Prepare.insert_param(table_name, records, partition_tag, ids)

        url = self._uri + "/tables/{}/vectors".format(table_name)

        data_dict = dict()
        if ids:
            data_dict["ids"] = ids
        if partition_tag:
            data_dict["tag"] = partition_tag

        data_dict["records"] = records

        data = json.dumps(data_dict)
        headers = {"Content-Type": "application/json"}

        try:
            response = rq.post(url, data=data, headers=headers)
            js = response.json()

            if 200 == response.status_code:
                ids = [int(item) for item in list(js["ids"])]
                return Status(message='Add vectors successfully!'), ids

            return Status(Status(js["code"], js["message"])), []
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout'), []
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e)), []

    def search(self, table_name, top_k, nprobe,
               query_records, query_ranges=None, partition_tags=None, **kwargs):

        _ = Prepare.search_param(table_name, top_k, nprobe, query_records, query_ranges, partition_tags)

        url = self._uri + "/tables/{}/vectors".format(table_name)

        body_dict = dict()
        if partition_tags:
            body_dict["tags"] = partition_tags
        body_dict["topk"] = top_k
        body_dict["nprobe"] = nprobe
        body_dict["records"] = query_records
        data = json.dumps(body_dict)
        headers = {"Content-Type": "application/json"}

        try:
            response = rq.put(url, data, headers=headers)

            if 200 == response.status_code:
                return Status(), TopKQueryResult2(response)

            js = response.json()
            return Status(Status(js["code"], js["message"])), None
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout'), None
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e)), None

    def search_in_files(self, table_name, file_ids, query_records, top_k,
                        nprobe=16, query_ranges=None, **kwargs):
        _ = Prepare.search_vector_in_files_param(table_name, query_records, query_ranges, top_k, nprobe, file_ids)
        url = self._uri + "/tables/{}/vectors".format(table_name)

        body_dict = dict()
        body_dict["topk"] = top_k
        body_dict["nprobe"] = nprobe
        body_dict["records"] = query_records
        body_dict["file_ids"] = file_ids
        data = json.dumps(body_dict)
        headers = {"Content-Type": "application/json"}

        try:
            response = rq.put(url, data, headers=headers)

            if 200 == response.status_code:
                return Status(), TopKQueryResult2(response)

            js = response.json()
            return Status(Status(js["code"], js["message"])), None
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout'), None
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e)), None

    def create_index(self, table_name, index, timeout):
        _ = Prepare.index_param(table_name, index)

        try:
            url = self._uri + "/tables/{}/indexes".format(table_name)

            data = json.dumps(index)

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
        return Status()
        # super().preload_table(table_name, timeout)

    def describe_index(self, table_name, timeout):
        _ = Prepare.table_name(table_name)
        url = self._uri + "tables/{}/indexes".format(table_name)

        try:
            response = rq.get(url)
            js = response.json()

            if 200 == response.status_code:
                return Status(), IndexParam(table_name, js["index_type"], js["nlist"])
            else:
                return Status(js["code"], js["message"]), None

        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout')
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e))

    def drop_index(self, table_name, timeout):
        _ = Prepare.table_name(table_name)

        try:
            url = self._uri + "/tables/{}/indexes".format(table_name)

            response = rq.delete(url)

            if 204 == response.status_code:
                return Status()
            else:
                js = response.json()
                return Status(js["code"], js["message"])
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout')
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e))

    def _cmd(self, cmd, timeout=10):
        url = self._uri + "/cmd/{}".format(cmd)

        try:
            response = rq.get(url, timeout=timeout)

            js = response.json()
            if 200 == response.status_code:
                return Status(), js["reply"]

            Status(code=js["code"], message=js["message"]), None
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout'), None
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e)), None

    def create_partition(self, table_name, partition_name, partition_tag, timeout=10):
        _ = Prepare.partition_param(table_name, partition_name, partition_tag)

        url = self._uri + "/tables/{}/partitions".format(table_name)

        try:
            data = json.dumps({"parition_name": partition_name, "partition_tag": partition_tag})
            headers = {"Content-Type": "application/json"}

            response = rq.post(url, data=data, headers=headers)
            if 201 == response.status_code:
                return Status()

            js = response.json()
            return Status(Status(js["code"], js["message"]))
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout')
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e))

    def show_partitions(self, table_name, timeout):
        _ = Prepare.table_name(table_name)

        url = self._uri + "/tables/{}/partitions".format(table_name)
        
        try:
            response = rq.get(url, timeout)
            js = response.json()
            if 200 == response.status_code:
                partition_list = []
                for partition in js["partitions"]:
                    partition_param = PartitionParam(
                        partition["table_name"],
                        partition["partition_name"],
                        partition["tag"]
                    )
                    partition_list.append(partition_param)

                return Status(), partition_list

            return Status(Status(js["code"], js["message"]))
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout')
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e))

    def drop_partition(self, table_name, partition_tag, timeout=10):
        url = self._uri + "/tables/{}/partitions/{}".format(table_name, partition_tag)
        
        try:
            response = rq.delete(url, timeout=timeout)
            if 204 == response.status_code:
                return Status()

            js = response.json()
            return Status(Status(js["code"], js["message"]))
        except rq.exceptions.Timeout:
            return Status(Status.UNEXPECTED_ERROR, message='Request timeout')
        except Exception as e:
            return Status(Status.UNEXPECTED_ERROR, message=str(e))

# Copyright (C) 2019-2021 Zilliz. All rights reserved.
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

import copy
import threading
from urllib import parse
from typing import Tuple

from ..client.check import is_legal_host, is_legal_port, is_legal_address
from ..client.grpc_handler import GrpcHandler

from ..settings import Config
from ..exceptions import ExceptionsMessage, ConnectionConfigException, ConnectionNotExistException

VIRTUAL_PORT = 443

def synchronized(func):
    """
    Decorator in order to achieve thread-safe singleton class.
    """
    func.__lock__ = threading.Lock()

    def lock_func(*args, **kwargs):
        with func.__lock__:
            return func(*args, **kwargs)

    return lock_func


class SingleInstanceMetaClass(type):
    instance = None

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(cls, *args, **kwargs):
        if cls.instance:
            return cls.instance
        cls.instance = cls.__new__(cls)
        cls.instance.__init__(*args, **kwargs)
        return cls.instance

    @synchronized
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)


class Connections(metaclass=SingleInstanceMetaClass):
    """ Class for managing all connections of milvus.  Used as a singleton in this module.  """

    def __init__(self):
        """ Constructs a default milvus alias config

            default config will be read from env: MILVUS_URI and MILVUS_CONN_ALIAS
            with default value: default="localhost:19530"

         Read default connection config from environment variable: MILVUS_URI.
            Format is:
                [scheme://][<user>@<password>]host[:<port>]

                scheme is one of: http, https, or <empty>

        Examples:
            localhost
            localhost:19530
            test_user@localhost:19530
            http://test_userlocalhost:19530
            https://test_user:password@localhost:19530

        """
        self._alias = {}
        self._connected_alias = {}
        self._connection_references = {}
        self._con_lock = threading.RLock()
        # info = self.__parse_info(
        #   uri=Config.MILVUS_URI,
        #   host=Config.DEFAULT_HOST,
        #   port=Config.DEFAULT_PORT,
        #   user = Config.MILVUS_USER,
        #   password = Config.MILVUS_PASSWORD,
        #   token = Config.MILVUS_TOKEN,
        #   secure=Config.DEFAULT_SECURE,
        #   db_name=Config.MILVUS_DB_NAME
        # )

        # default_conn_config = {
        #     "user": info["user"],
        #     "address": info["address"],
        #     "db_name": info["db_name"],
        #     "secure": info["secure"],
        # }

        # self.add_connection(**{Config.MILVUS_CONN_ALIAS: default_conn_config})

    def add_connection(self, **kwargs):
        """ Configures a milvus connection.

        Addresses priority in kwargs: address, uri, host and port

        :param kwargs:
            * *address* (``str``) -- Optional. The actual address of Milvus instance.
                Example address: "localhost:19530"

            * *uri* (``str``) -- Optional. The uri of Milvus instance.
                Example uri: "http://localhost:19530", "tcp:localhost:19530", "https://ok.s3.south.com:19530".

            * *host* (``str``) -- Optional. The host of Milvus instance.
                Default at "localhost", PyMilvus will fill in the default host if only port is provided.

            * *port* (``str/int``) -- Optional. The port of Milvus instance.
                Default at 19530, PyMilvus will fill in the default port if only host is provided.

        Example::

            connections.add_connection(
                default={"host": "localhost", "port": "19530"},
                dev1={"host": "localhost", "port": "19531"},
                dev2={"uri": "http://random.com/random"},
                dev3={"uri": "http://localhost:19530"},
                dev4={"uri": "tcp://localhost:19530"},
                dev5={"address": "localhost:19530"},
                prod={"uri": "http://random.random.random.com:19530"},
            )
        """
        for alias, config in kwargs.items():
            parsed = self.__parse_info(**config)

            if alias in self._connected_alias:
                if (
                    self._alias[alias].get("address") != parsed["address"]
                    or self._alias[alias].get("user") != parsed["user"]
                    or self._alias[alias].get("db_name") != parsed["db_name"]
                    or self._alias[alias].get("secure") != parsed["secure"]
                ):
                    raise ConnectionConfigException(message=ExceptionsMessage.ConnDiffConf % alias)
            alias_config = {
                "address": parsed["address"],
                "user": parsed["user"],
                "db_name":  parsed["db_name"],
                "secure": parsed["secure"],
            }

            self._alias[alias] = alias_config

    def disconnect(self, alias: str):
        """ Disconnects connection from the registry.

        :param alias: The name of milvus connection
        :type alias: str
        """
        if not isinstance(alias, str):
            raise ConnectionConfigException(message=ExceptionsMessage.AliasType % type(alias))

        with self._con_lock:
            if alias in self._connected_alias:
                gh = self._connected_alias.pop(alias)
                self._connection_references[id(gh)] -= 1
                if self._connection_references[id(gh)] <= 0:
                    gh.close()
                    del self._connection_references[id(gh)]

    def remove_connection(self, alias: str):
        """ Removes connection from the registry.

        :param alias: The name of milvus connection
        :type alias: str
        """
        if not isinstance(alias, str):
            raise ConnectionConfigException(message=ExceptionsMessage.AliasType % type(alias))

        self.disconnect(alias)
        self._alias.pop(alias, None)

    # pylint: disable=too-many-statements
    def connect(self, alias=Config.MILVUS_CONN_ALIAS, user="", password="", token="", db_name="", **kwargs):
        """
        Constructs a milvus connection and register it under given alias.

        :param alias: The name of milvus connection
        :type alias: str

        param user: Optional. Use which user to connect to Milvus instance. If user and password
            are provided, we will add related header in every RPC call.
        :type user: str

        :param password: Optional and required when user is provided. The password corresponding to
            the user.
        :type password: str

        :param token: Optional. Serving as the key for identification and authentication purposes.
            Whenever a token is furnished, we shall supplement the corresponding header to each RPC call.
        :type token: str

        :param db_name: Optional. default database name of this connection
        :type db_name: str

        :param kwargs:
            * *address* (``str``) -- Optional. The actual address of Milvus instance.
                Example address: "localhost:19530"
            * *uri* (``str``) -- Optional. The uri of Milvus instance.
                Example uri: "http://localhost:19530", "tcp:localhost:19530", "https://ok.s3.south.com:19530".
            * *host* (``str``) -- Optional. The host of Milvus instance.
                Default at "localhost", PyMilvus will fill in the default host if only port is provided.
            * *port* (``str/int``) -- Optional. The port of Milvus instance.
                Default at 19530, PyMilvus will fill in the default port if only host is provided.
            * *secure* (``bool``) --
                Optional. Default is false. If set to true, tls will be enabled.
            * *client_key_path* (``str``) --
                Optional. If use tls two-way authentication, need to write the client.key path.
            * *client_pem_path* (``str``) --
                Optional. If use tls two-way authentication, need to write the client.pem path.
            * *ca_pem_path* (``str``) --
                Optional. If use tls two-way authentication, need to write the ca.pem path.
            * *server_pem_path* (``str``) --
                Optional. If use tls one-way authentication, need to write the server.pem path.
            * *server_name* (``str``) --
                Optional. If use tls, need to write the common name.

        :raises NotImplementedError: If handler in connection parameters is not GRPC.
        :raises ParamError: If pool in connection parameters is not supported.
        :raises Exception: If server specified in parameters is not ready, we cannot connect to
                           server.

        :example:
            >>> from pymilvus import connections
            >>> connections.connect("test", host="localhost", port="19530")
        """
        # pylint: disable=too-many-statements

        def connect_milvus(**kwargs):
            with self._con_lock:
                # Check if the alias is already connected
                if alias in self._connected_alias:
                    if (
                        self._alias[alias]["address"] != kwargs["address"]
                        or self._alias[alias]["db_name"] != kwargs["db_name"]
                        or self._alias[alias]["user"] != kwargs["user"]
                        or self._alias[alias]["secure"] != kwargs["secure"]
                    ):
                        raise ConnectionConfigException(message=ExceptionsMessage.AliasUsed % alias)
                    return

                # Check if alias already made but not connected yet.
                if (
                    alias in self._alias
                    and (
                        self._alias[alias]["address"] != kwargs["address"]
                        or self._alias[alias]["db_name"] != kwargs["db_name"]
                        # or self._alias[alias]["user"] != kwargs["user"] # Can use different user
                        # or self._alias[alias]["secure"] != kwargs["secure"] # Can use different secure
                    )
                ):
                    raise ConnectionConfigException(message=ExceptionsMessage.AliasUsed % alias)

                gh = None

                # Check if reusable connection already exists
                for key, connection_details in self._alias.items():

                    if (
                        key in self._connected_alias
                        and connection_details["address"] == kwargs["address"]
                        and connection_details["user"] == kwargs["user"]
                        and connection_details["db_name"] == kwargs["db_name"]
                        and connection_details["secure"] == kwargs["secure"]
                    ):
                        gh = self._connected_alias[key]
                        break
                if gh is None:
                    gh = GrpcHandler(**kwargs)
                    t = kwargs.get("timeout", None)
                    timeout = t if isinstance(t, (int, float)) else Config.MILVUS_CONN_TIMEOUT
                    gh._wait_for_channel_ready(timeout=timeout)

                kwargs.pop('password', None)
                kwargs.pop('token', None)

                self._connected_alias[alias] = gh

                self._alias[alias] = copy.deepcopy(kwargs)

                if id(gh) not in self._connection_references:
                    self._connection_references[id(gh)] = 1
                else:
                    self._connection_references[id(gh)] += 1

        if not isinstance(alias, str):
            raise ConnectionConfigException(message=ExceptionsMessage.AliasType % type(alias))

        # Grab the relevant info for connection
        address = kwargs.pop("address", "")
        uri = kwargs.pop("uri", "")
        host = kwargs.pop("host", "")
        port =  kwargs.pop("port", "")
        secure = kwargs.pop("secure", None)

        # Clean the connection info
        address = '' if address is None else str(address)
        uri = '' if uri is None else str(uri)
        host = '' if host is None else str(host)
        port = '' if port is None else str(port)
        user = '' if user is None else str(user)
        password = '' if password is None else str(password)
        token = '' if token is None else (str(token))
        db_name = '' if db_name is None else str(db_name)

        # Replace empties with defaults from enviroment
        uri = uri if uri != '' else Config.MILVUS_URI
        host = host if host != '' else Config.DEFAULT_HOST
        port = port if port != '' else Config.DEFAULT_PORT
        user = user if user != '' else Config.MILVUS_USER
        password = password if password != '' else Config.MILVUS_PASSWORD
        token = token if token != '' else Config.MILVUS_TOKEN
        db_name = db_name if db_name != '' else Config.MILVUS_DB_NAME

        # Check if alias exists first
        if alias in self._alias:
            kwargs = dict(self._alias[alias].items())
            # If user is passed in, use it, if not, use previous connections user.
            prev_user = kwargs.pop("user")
            kwargs["user"] = user if user != "" else prev_user

            # If new secure parameter passed in, use that
            prev_secure = kwargs.pop("secure")
            kwargs["secure"] = secure if secure is not None else prev_secure

             # If db_name is passed in, use it, if not, use previous db_name.
            prev_db_name = kwargs.pop("db_name")
            kwargs["db_name"] = db_name if db_name != "" else prev_db_name

        # If at least one address info is given, parse it
        elif set([address, uri, host, port]) != {''}:
            secure = secure if secure is not None else Config.DEFAULT_SECURE
            parsed = self.__parse_info(address, uri, host, port, db_name, user, password, token, secure)
            kwargs.update(parsed)

        # If no details are given and no alias exists
        else:
            raise ConnectionConfigException(message=ExceptionsMessage.ConnLackConf % alias)

        connect_milvus(**kwargs)


    def list_connections(self) -> list:
        """ List names of all connections.

        :return list:
            Names of all connections.

        :example:
            >>> from pymilvus import connections
            >>> connections.connect("test", host="localhost", port="19530")
            >>> connections.list_connections()
            // TODO [('default', None), ('test', <pymilvus.client.grpc_handler.GrpcHandler object at 0x7f05003f3e80>)]
        """
        with self._con_lock:
            return [(k, self._connected_alias.get(k, None)) for k in self._alias]

    def get_connection_addr(self, alias: str):
        """
        Retrieves connection configure by alias.

        :param alias: The name of milvus connection
        :type  alias: str

        :return dict:
            The connection configure which of the name is alias.
            If alias does not exist, return empty dict.

        :example:
            >>> from pymilvus import connections
            >>> connections.connect("test", host="localhost", port="19530")
            >>> connections.list_connections()
            [('default', None), ('test', <pymilvus.client.grpc_handler.GrpcHandler object at 0x7f4045335f10>)]
            >>> connections.get_connection_addr('test')
            {'host': 'localhost', 'port': '19530'}
        """
        if not isinstance(alias, str):
            raise ConnectionConfigException(message=ExceptionsMessage.AliasType % type(alias))

        return self._alias.get(alias, {})

    def has_connection(self, alias: str) -> bool:
        """ Check if connection named alias exists.

        :param alias: The name of milvus connection
        :type  alias: str

        :return bool:
            if the connection of name alias exists.

        :example:
            >>> from pymilvus import connections
            >>> connections.connect("test", host="localhost", port="19530")
            >>> connections.list_connections()
            [('default', None), ('test', <pymilvus.client.grpc_handler.GrpcHandler object at 0x7f4045335f10>)]
            >>> connections.get_connection_addr('test')
            {'host': 'localhost', 'port': '19530'}
        """
        if not isinstance(alias, str):
            raise ConnectionConfigException(message=ExceptionsMessage.AliasType % type(alias))
        with self._con_lock:
            return alias in self._connected_alias

    def __parse_info(
            self,
            address: str = "",
            uri: str = "",
            host: str = "",
            port: str = "",
            db_name: str = "",
            user: str = "",
            password: str = "",
            token: str = "",
            secure: bool = False,
            **kwargs) -> dict:

        extracted_address = ""
        extracted_user = ""
        extracted_password = ""
        extracted_db_name = ""
        extracted_token = ""
        extracted_secure = None
        # If URI
        if uri != "":
            extracted_address, extracted_user, extracted_password, extracted_db_name, extracted_secure = (
                self.__parse_address_from_uri(uri)
            )
        # If Address
        elif address != "":
            if not is_legal_address(address):
                raise ConnectionConfigException(
                    message=f"Illegal address: {address}, should be in form 'localhost:19530'")
            extracted_address = address
        # If Host port
        else:
            self.__verify_host_port(host, port)
            extracted_address = f"{host}:{port}"
        ret = {}
        ret["address"] = extracted_address
        ret["user"] = user if extracted_user == "" else str(extracted_user)
        ret["password"] = password if extracted_password == "" else str(extracted_password)
        ret["db_name"] = db_name if extracted_db_name == "" else str(extracted_db_name)
        ret["token"] = token if extracted_token == "" else str(extracted_token)
        ret["secure"] = secure if extracted_secure is None else extracted_secure

        return ret

    def __verify_host_port(self, host, port):
        if not is_legal_host(host):
            raise ConnectionConfigException(message=ExceptionsMessage.HostType)
        if not is_legal_port(port):
            raise ConnectionConfigException(message=ExceptionsMessage.PortType)
        if not 0 <= int(port) < 65535:
            raise ConnectionConfigException(message=f"port number {port} out of range, valid range [0, 65535)")

    def __parse_address_from_uri(self, uri: str) -> Tuple[str, str, str, str]:
        illegal_uri_msg = "Illegal uri: [{}], expected form 'https://user:pwd@example.com:12345'"
        try:
            parsed_uri = parse.urlparse(uri)
        except (Exception) as e:
            raise ConnectionConfigException(
                message=f"{illegal_uri_msg.format(uri)}: <{type(e).__name__}, {e}>") from None

        if len(parsed_uri.netloc) == 0:
            raise ConnectionConfigException(message=f"{illegal_uri_msg.format(uri)}") from None

        group = parsed_uri.path.split("/")
        if len(group) > 1:
            db_name = group[1]
        else:
            db_name = ""

        host = parsed_uri.hostname if parsed_uri.hostname is not None else ""
        port = parsed_uri.port if parsed_uri.port is not None else ""
        user = parsed_uri.username if parsed_uri.username is not None else ""
        password = parsed_uri.password if parsed_uri.password is not None else ""
        secure = parsed_uri.scheme.lower() == "https:"

        if host == "":
            raise ConnectionConfigException(message=f"Illegal uri: URI is missing host address: {uri}")
        if port == "":
            raise ConnectionConfigException(message=f"Illegal uri: URI is missing port: {uri}")

        self.__verify_host_port(host, port)
        addr = f"{host}:{port}"

        if not is_legal_address(addr):
            raise ConnectionConfigException(message=illegal_uri_msg.format(uri))

        return addr, user, password, db_name, secure


    def _fetch_handler(self, alias=Config.MILVUS_CONN_ALIAS) -> GrpcHandler:
        """ Retrieves a GrpcHandler by alias. """
        if not isinstance(alias, str):
            raise ConnectionConfigException(message=ExceptionsMessage.AliasType % type(alias))

        conn = self._connected_alias.get(alias, None)
        if conn is None:
            raise ConnectionNotExistException(message=ExceptionsMessage.ConnectFirst)

        return conn


# Singleton Mode in Python
connections = Connections()

"""Public ``DQL`` operation constructor namespace for ``execute_dql``."""

from pymilvus.client.dql import DQLRequest, DQLResult, _bind_dql_op

from .milvus_client import MilvusClient


class DQL:
    """Namespace of read-only DQL operation constructors for ``execute_dql``."""

    search = staticmethod(_bind_dql_op("search", MilvusClient.search))


__all__ = ["DQL", "DQLRequest", "DQLResult"]

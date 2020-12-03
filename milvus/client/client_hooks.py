from .hooks import BaseSearchHook
from .grpc_client.grpc_results import QueryResult


class SearchHook(BaseSearchHook):

    def handle_response(self, _response):
        """
        use class `TopKQueryResult` to deal with response from server
        """
        return QueryResult(_response)


class HybridSearchHook(BaseSearchHook):

    def handle_response(self, _response):
        return QueryResult(_response)

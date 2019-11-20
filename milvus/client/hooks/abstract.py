class AbstractHook:
    pass


class BaseaSearchHook(AbstractHook):

    def prepare_data(self, *args, **kwargs):
        """
        Prepare grpc data
        """
        pass

    def pre_search(self, *args, **kwargs):
        """
        This method allow user to prepare for starting searching vectors.
        Called before searching vectors on server.
        """
        pass

    def aft_search(self, *args, **kwargs):
        """
        Called after searching vectors on server.
        """
        pass

    def on_response(self, *args, **kwargs):
        """
        Indicate if deal with response from server.
        return bool variables. False default

        When return True, the method tell caller to return raw response received from
        server directly.
        """
        return False

    def handle_response(self, _response):
        """
        deal with response from server and return processing results
        """
        pass

class AbstractHook:
    pass


class BaseaSearchHook(AbstractHook):

    def pre_search(self, *args, **kwargs):
        pass

    def aft_search(self, *args, **kwargs):
        pass

    def on_response(self, *args, **kwargs):
        return False

    def handle_response(self, _response):
        pass

class Search(object):

    def __init__(self, collection, data, params, limit, expr=None, partition_names=None, fields=None, **kwargs):
        pass

    def execute(self, sync=True, **kwargs):
        pass

class SearchResult(object):

    def __iter__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def done(self):
        pass


class Hits(object):
    def __iter__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    @property
    def distances(self):
        pass

    @property
    def ids(self):
        pass


class Hit(object):

    @property
    def distance(self):
        pass

    @property
    def id(self):
        pass

    @property
    def score(self):
        pass
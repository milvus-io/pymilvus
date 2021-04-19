class Search(object):

    def __init__(self, collection, data, params, limit, expr=None, partition_names=None, fields=None, **kwargs):
        """
        Construct a search object.

        :param collection: The collection object used to search
        :type collection: class `collection.Collection`

        :param data: Data to search, the dimension of data needs to align with column number
        :type  data: list-like(list, tuple, numpy.ndarray) object or pandas.DataFrame

        :param params: Search parameters
        :type  params: dict

        :param limit:
        :type  limit: int

        :param expr: Search expression
        :type  expr: str

        :param fields: The fields to return in the search result
        :type  fields: list[str]
        """
        pass

    def execute(self, sync=True, **kwargs):
        """
        Return the search result.

        :param sync: Whether to wait the search task done. True to wait, false not to
        :type sync: bool

        :return: Search result.
        :rtype: class `SearchResult`
        """
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
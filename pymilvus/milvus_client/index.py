class IndexParam:
    def __init__(self, field_name: str, index_type: str, index_name: str, **kwargs):
        self._field_name = field_name
        self._index_type = index_type
        self._index_name = index_name
        self._kwargs = kwargs

    @property
    def field_name(self):
        return self._field_name

    @property
    def index_name(self):
        return self._index_name

    @property
    def index_type(self):
        return self._index_type

    def __iter__(self):
        yield "field_name", self.field_name
        if self.index_type:
            yield "index_type", self.index_type
        yield "index_name", self.index_name
        yield from self._kwargs.items()

    def __str__(self):
        return str(dict(self))

    def __eq__(self, other: None):
        if isinstance(other, self.__class__):
            return dict(self) == dict(other)

        if isinstance(other, dict):
            return dict(self) == other
        return False


class IndexParams:
    def __init__(self, field_name: str = "", **kwargs):
        self._indexes = {}
        if field_name:
            self.add_index(field_name, **kwargs)

    def add_index(self, field_name: str, index_type: str = "", index_name: str = "", **kwargs):
        index_param = IndexParam(field_name, index_type, index_name, **kwargs)
        pair_key = (field_name, index_name)
        self._indexes[pair_key] = index_param

    def __iter__(self):
        for v in self._indexes.values():
            yield dict(v)

    def __str__(self):
        return str(list(self))

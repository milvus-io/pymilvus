class CollectionSchema(object):

    def __init__(self, fields, auto_id=True, description="", **kwargs):
        self.fields = fields
        self.auto_id = auto_id
        self.description = description

    @property
    def primary_field(self):
        for field in self.fields:
            if field.is_primary:
                return field


class FieldSchema(object):
    def __init__(self, name, type, is_primary=False, description="", type_params=None, **kwargs):
        self.name = name
        self._type = type
        self.is_primary = is_primary
        self.description = description
        self._type_params = type_params
        self._kwargs = kwargs

    @property
    def params(self):
        return self._type_params

    @property
    def data_type(self):
        return self._type

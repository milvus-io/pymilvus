class Prepare(object):
    @classmethod
    def prepare_insert_data_for_list_or_tuple(cls, data, schema):
        if not isinstance(data, list) and not isinstance(data, tuple):
            raise Exception("data is not invalid")

        fields = schema.fields
        if len(data) != len(fields):
            raise Exception(f"collection has {len(fields)} fields, but go {len(data)} fields")

        entities = [{
            "name": field.name,
            "type": field.dtype,
            "values": data[i],
        } for i, field in enumerate(fields)]

        return entities


from pymilvus.client.prepare import Prepare

from pymilvus import CollectionSchema, FieldSchema, DataType


class TestPrepare:
    def test_search_requests_with_expr_offset(self):
        fields = [
            FieldSchema("pk", DataType.INT64, is_primary=True),
            FieldSchema("v", DataType.FLOAT_VECTOR, dim=2),
        ]

        schema = CollectionSchema(fields).to_dict()
        data = [
            [1., 2.],
            [1., 2.],
            [1., 2.],
            [1., 2.],
            [1., 2.],
        ]

        search_params = {
            "metric_type": "L2",
            "offset": 10,
        }

        ret = Prepare.search_requests_with_expr("name", data, "v", search_params, 100, schema=schema)

        offset_exists = False
        for p in ret[0].search_params:
            if p.key == "offset":
                offset_exists = True
                assert p.value == "10"

        assert offset_exists is True

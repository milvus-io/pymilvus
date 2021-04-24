import pytest
from tests.utils import *
from tests.constants import *

uid = "create_collection"

class TestCreateCollection:
    """
    ******************************************************************
      The following cases are used to test `create_collection` function
    ******************************************************************
    """
    @pytest.fixture(
        scope="function",
        params=gen_single_filter_fields()
    )
    def get_filter_field(self, request):
        yield request.param

    @pytest.fixture(
        scope="function",
        params=gen_single_vector_fields()
    )
    def get_vector_field(self, request):
        yield request.param

    @pytest.fixture(
        scope="function",
        params=gen_segment_row_limits()
    )
    def get_segment_row_limit(self, request):
        yield request.param

    def test_create_collection_fields(self, connect, get_filter_field, get_vector_field):
        '''
        target: test create normal collection with different fields
        method: create collection with diff fields: metric/field_type/...
        expected: no exception raised
        '''
        filter_field = get_filter_field
        # logging.getLogger().info(filter_field)
        vector_field = get_vector_field
        collection_name = gen_unique_str(uid)
        fields = {
                "fields": [filter_field, vector_field],
        }
        # logging.getLogger().info(fields)
        connect.create_collection(collection_name, fields)
        assert connect.has_collection(collection_name)

    def test_create_collection_fields_create_index(self, connect, get_filter_field, get_vector_field):
        '''
        target: test create normal collection with different fields
        method: create collection with diff fields: metric/field_type/...
        expected: no exception raised
        '''
        filter_field = get_filter_field
        vector_field = get_vector_field
        collection_name = gen_unique_str(uid)
        fields = {
                "fields": [filter_field, vector_field],
        }
        connect.create_collection(collection_name, fields)
        assert connect.has_collection(collection_name)
        
    @pytest.mark.skip("no segment_row_limit")
    def test_create_collection_segment_row_limit(self, connect):
        '''
        target: test create normal collection with different fields
        method: create collection with diff segment_row_limit
        expected: no exception raised
        '''
        collection_name = gen_unique_str(uid)
        fields = copy.deepcopy(default_fields)
        #  fields["segment_row_limit"] = get_segment_row_limit
        connect.create_collection(collection_name, fields)
        assert connect.has_collection(collection_name)

    @pytest.mark.skip("no flush")
    def _test_create_collection_auto_flush_disabled(self, connect):
        '''
        target: test create normal collection, with large auto_flush_interval
        method: create collection with corrent params
        expected: create status return ok
        '''
        disable_flush(connect)
        collection_name = gen_unique_str(uid)
        try:
            connect.create_collection(collection_name, default_fields)
        finally:
            enable_flush(connect)

    def test_create_collection_after_insert(self, connect, collection):
        '''
        target: test insert vector, then create collection again
        method: insert vector and create collection
        expected: error raised
        '''
        # pdb.set_trace()
        connect.bulk_insert(collection, default_entity)

        with pytest.raises(Exception) as e:
            connect.create_collection(collection, default_fields)

    def test_create_collection_after_insert_flush(self, connect, collection):
        '''
        target: test insert vector, then create collection again
        method: insert vector and create collection
        expected: error raised
        '''
        connect.bulk_insert(collection, default_entity)
        connect.flush([collection])
        with pytest.raises(Exception) as e:
            connect.create_collection(collection, default_fields)

    def test_create_collection_without_connection(self, dis_connect):
        '''
        target: test create collection, without connection
        method: create collection with correct params, with a disconnected instance
        expected: error raised
        '''
        collection_name = gen_unique_str(uid)
        with pytest.raises(Exception) as e:
            dis_connect.create_collection(collection_name, default_fields)

    def test_create_collection_existed(self, connect):
        '''
        target: test create collection but the collection name have already existed
        method: create collection with the same collection_name
        expected: error raised
        '''
        collection_name = gen_unique_str(uid)
        connect.create_collection(collection_name, default_fields)
        with pytest.raises(Exception) as e:
            connect.create_collection(collection_name, default_fields)

    def test_create_after_drop_collection(self, connect, collection):
        '''
        target: create with the same collection name after collection dropped 
        method: delete, then create
        expected: create success
        '''
        connect.drop_collection(collection)
        time.sleep(2)
        connect.create_collection(collection, default_fields)

    @pytest.mark.level(2)
    def test_create_collection_multithread(self, connect):
        '''
        target: test create collection with multithread
        method: create collection using multithread, 
        expected: collections are created
        '''
        threads_num = 8 
        threads = []
        collection_names = []

        def create():
            collection_name = gen_unique_str(uid)
            collection_names.append(collection_name)
            connect.create_collection(collection_name, default_fields)
        for i in range(threads_num):
            t = threading.Thread(target=create, args=())
            threads.append(t)
            t.start()
            time.sleep(0.2)
        for t in threads:
            t.join()
        
        for item in collection_names:
            assert item in connect.list_collections()
            connect.drop_collection(item)


class TestCreateCollectionInvalid(object):
    """
    Test creating collections with invalid params
    """
    @pytest.fixture(
        scope="function",
        params=gen_invalid_metric_types()
    )
    def get_metric_type(self, request):
        yield request.param

    @pytest.fixture(
        scope="function",
        params=gen_invalid_ints()
    )
    def get_segment_row_limit(self, request):
        yield request.param

    @pytest.fixture(
        scope="function",
        params=gen_invalid_ints()
    )
    def get_dim(self, request):
        yield request.param

    @pytest.fixture(
        scope="function",
        params=gen_invalid_strs()
    )
    def get_invalid_string(self, request):
        yield request.param

    @pytest.fixture(
        scope="function",
        params=gen_invalid_field_types()
    )
    def get_field_type(self, request):
        yield request.param

    @pytest.mark.level(2)
    @pytest.mark.skip("no segment row limit")
    def test_create_collection_with_invalid_segment_row_limit(self, connect, get_segment_row_limit):
       collection_name = gen_unique_str()
       fields = copy.deepcopy(default_fields)
       fields["segment_row_limit"] = get_segment_row_limit
       with pytest.raises(Exception) as e:
           connect.create_collection(collection_name, fields)

    @pytest.mark.level(2)
    def test_create_collection_with_invalid_dimension(self, connect, get_dim):
        dimension = get_dim
        collection_name = gen_unique_str()
        fields = copy.deepcopy(default_fields)
        fields["fields"][-1]["params"]["dim"] = dimension
        with pytest.raises(Exception) as e:
             connect.create_collection(collection_name, fields)

    @pytest.mark.level(2)
    def test_create_collection_with_invalid_collectionname(self, connect, get_invalid_string):
        collection_name = get_invalid_string
        with pytest.raises(Exception) as e:
            connect.create_collection(collection_name, default_fields)

    @pytest.mark.level(2)
    def test_create_collection_with_empty_collectionname(self, connect):
        collection_name = ''
        with pytest.raises(Exception) as e:
            connect.create_collection(collection_name, default_fields)

    @pytest.mark.level(2)
    def test_create_collection_with_none_collectionname(self, connect):
        collection_name = None
        with pytest.raises(Exception) as e:
            connect.create_collection(collection_name, default_fields)

    def test_create_collection_None(self, connect):
        '''
        target: test create collection but the collection name is None
        method: create collection, param collection_name is None
        expected: create raise error
        '''
        with pytest.raises(Exception) as e:
            connect.create_collection(None, default_fields)

    def test_create_collection_no_dimension(self, connect):
        '''
        target: test create collection with no dimension params
        method: create collection with corrent params
        expected: create status return ok
        '''
        collection_name = gen_unique_str(uid)
        fields = copy.deepcopy(default_fields)
        fields["fields"][-1]["params"].pop("dim")
        with pytest.raises(Exception) as e:
            connect.create_collection(collection_name, fields)

    @pytest.mark.skip("no segment row limit")
    def test_create_collection_no_segment_row_limit(self, connect):
       '''
       target: test create collection with no segment_row_limit params
       method: create collection with correct params
       expected: use default default_segment_row_limit
       '''
       collection_name = gen_unique_str(uid)
       fields = copy.deepcopy(default_fields)
       fields.pop("segment_row_limit")
       connect.create_collection(collection_name, fields)
       res = connect.describe_collection(collection_name)
       # logging.getLogger().info(res)
       assert res["segment_row_limit"] == default_server_segment_row_limit

    def test_create_collection_limit_fields(self, connect):
        collection_name = gen_unique_str(uid)
        limit_num = 64
        fields = copy.deepcopy(default_fields)
        for i in range(limit_num):
            field_name = gen_unique_str("field_name")
            field = {"name": field_name, "type": DataType.INT64}
            fields["fields"].append(field)
        with pytest.raises(Exception) as e:
            connect.create_collection(collection_name, fields)

    @pytest.mark.level(2)
    def test_create_collection_invalid_field_name(self, connect, get_invalid_string):
        collection_name = gen_unique_str(uid)
        fields = copy.deepcopy(default_fields)
        field_name = get_invalid_string
        field = {"name": field_name, "type": DataType.INT64}
        fields["fields"].append(field)
        with pytest.raises(Exception) as e:
            connect.create_collection(collection_name, fields)

    def test_create_collection_invalid_field_type(self, connect, get_field_type):
        collection_name = gen_unique_str(uid)
        fields = copy.deepcopy(default_fields)
        field_type = get_field_type
        field = {"name": "test_field", "type": field_type}
        fields["fields"].append(field)
        with pytest.raises(Exception) as e:
            connect.create_collection(collection_name, fields)

class AbsMilvus:

    def client_version(self):
        """
        Returns the version of the client.

        :return: Version of the client.

        :rtype: (str)
        """
        pass

    def server_status(self, timeout=30):
        """
        Returns the status of the Milvus server.

        :return:
            Status: Whether the operation is successful.

            str : Status of the Milvus server.

        :rtype: (Status, str)
        """
        pass

    def server_version(self, timeout=30):
        """
        Returns the version of the Milvus server.

        :return:
           Status: Whether the operation is successful.

           str : Version of the Milvus server.

        :rtype: (Status, str)
        """
        pass

    def _cmd(self, cmd, timeout=30):
        pass

    """ Collection
    """

    def create_collection(self, collection_name, fields, timeout=30):
        """
        Creates a collection.

        :type  collection_name: str
        :param collection_name: collection name.

        :param fields: field params.
        :type  fields: list, field num limitation : 32
            ` {"fields": [
                    {"field": "A", "type": DataType.INT64, "index": {"name":"", "type":"", "params": {..}}}
                    {"field": "B", "type": DataType.INT64},
                    {"field": "C", "type": DataType.INT64},
                    {"field": "Vec", "type": DataType.BINARY_VECTOR,
                     "params": {"metric_type": MetricType.L2, "dimension": 128}}
                ],
            "segment_size": 100}`

        :return:
            None

        :raises:
            CollectionExistException(BaseException)
            InvalidDimensionException(BaseException)
            InvalidMetricTypeException(BaseException)
            IllegalCollectionNameException(BaseException)
        """
        pass

    def has_collection(self, collection_name, timeout=30):
        """

        Checks whether a collection exists.

        :param collection_name: Name of the collection to check.
        :type  collection_name: str

        :return:
            bool

        :raises:
            IllegalCollectionNameException(BaseException)

        """
        pass

    def describe_collection(self, collection_name, timeout=30):
        """
        Returns information of a collection.
        Returned information should contain field info and index info of each field

        :type  collection_name: str
        :param collection_name: Name of the collection to describe.

        :returns:
            TableSchema

        :raises:
            CollectionNotExistException(BaseException)
            IllegalCollectionNameException(BaseException)
        """
        pass

    def list_collections(self, timeout=30):
        """
        Returns collection list.

        :return:
            collections: list of collection names, return when operation is successful

        :raises:

        """
        pass

    def get_collection_stats(self, collection_name, timeout=30):
        """
        Returns collection statistics information.

        :return:
            statistics: statistics information

        :raises:
            CollectionNotExistException(BaseException)
            IllegalCollectionNameException(BaseException)

        """
        pass

    def drop_collection(self, collection_name, timeout=30):
        """
        Deletes a collection by name.

        :type  collection_name: str
        :param collection_name: Name of the collection being deleted

        :returns:
            Status, indicate if operation is successful
                - SUCCESS_BUT_NOT_DROP_COLLECTION
                - DROPPED

        :raises:
            IllegalCollectionNameException(BaseException)
        """
        pass

    """ Index
    """

    def create_index(self, collection_name, field_name, index_name, params, timeout=None, **kwargs):
        """
        Creates index for a collection.

        :param collection_name: Collection used to create index.
        :type  collection_name: str

        :param field_name:
        :type  field_name: str

        :param index_name:
        :type  index_name: str

        :param params: index params
        :type params:
            `{
                "index_type": IndexType.IVF_FLAT,
                "params": {
                    "nlist": 128
                }
            }`

        :return:
            Status:
                - SUCCESS_BUT_NOT_CREATE_INDEX
                - CREATED

        :raises:
            CollectionNotExistException(BaseException)
            IllegalCollectionNameException(BaseException)
            IllegalFieldNameException(BaseException)
            IllegalIndexNameException(BaseException)
            InvalidIndexParamsException(BaseException)
            InvalidIndexTypeException(BaseException)
        """
        pass

    # @deprecated
    # def list_indexes(self, collection_name, field_name):
    #     """
    #     """
    #     pass
    #
    # @depracated
    def describe_index(self, collection_name, field_name, timeout=30):
        """
        Show index information of a collection.

        :type collection_name: str
        :param collection_name: table name been queried

        :returns:
            IndexSchema:

        :raises:
            CollectionNotExistException(BaseException)
            IllegalCollectionNameException(BaseException)
            IllegalFieldNameException(BaseException)
            IllegalIndexNameException(BaseException)

        """
        pass

    def get_index_progress(self, collection_name, field_name, timeout=30):
        """
        Show index information of a collection.

        :type collection_name: str
        :param collection_name: table name been queried

        :returns:
            IndexSchema:

        :raises:
            CollectionNotExistException(BaseException)
            IllegalCollectionNameException(BaseException)
            IllegalFieldNameException(BaseException)
            IllegalIndexNameException(BaseException)

        """
        pass

    def drop_index(self, collection_name, field_name, index_name, timeout=30):
        """
        Removes an index.

        :param collection_name: target collection name.
        :type collection_name: str

        :return:
            Status:
                - SUCCESS_BUT_NOT_DROP_INDEX
                - DROPPED

        :raises:
            CollectionNotExistException(BaseException)
            IllegalCollectionNameException(BaseException)
            IllegalFieldNameException(BaseException)
            IllegalIndexNameException(BaseException)

        """
        pass

    """ Partition
    """

    def create_partition(self, collection_name, partition_name, timeout=30):
        """
        create a partition for a collection.

        :param collection_name: Name of the collection.
        :type  collection_name: str

        :param partition_name: Name of the partition.
        :type  partition_name: str

        :param partition_name: Name of the partition tag.
        :type  partition_name: str

        :return:
            Status: Whether the operation is successful.
                - SUCCESS_BUT_NOT_CREATE_PARTITION
                - CREATED

        :raises:
            CollectionNotExistException(BaseException)
            IllegalCollectionNameException(BaseException)
            IllegalPartitionTagException(BaseException)
            ExceedPartitionMaxLimitException(BaseException)

        """
        pass

    def has_partition(self, collection_name, partition_name):
        """
        Check if specified partition exists.


        :param collection_name: target table name.
        :type  collection_name: str

        :param partition_name: partition tag.
        :type  partition_name: str

        :return:
            exists: bool, if specified partition exists

        :raises:
            CollectionNotExistException(BaseException)
            IllegalCollectionNameException(BaseException)
            IllegalPartitionTagEixception(BaseException)
            ExceedPartitionMaxLimitException(BaseException)

        """
        pass

    def list_partitions(self, collection_name, timeout=30):
        """
        Show all partitions in a collection.

        :param collection_name: target table name.
        :type  collection_name: str

        :param timeout: time waiting for response.
        :type  timeout: int

        :return:
            partition_list: list[str]

        :raises:
            CollectionNotExistException(BaseException)
            IllegalCollectionNameException(BaseException)

        """
        pass

    def drop_partition(self, collection_name, partition_name, timeout=30):
        """
        Deletes a partition in a collection.

        :param collection_name: Collection name.
        :type  collection_name: str

        :param partition_name: Partition name.
        :type  partition_name: str

        :return:
            Status: Whether the operation is successful.
                - SUCCESS_BUT_NOT_DROP_PARTITION
                - DROPPED

        :raises:
            IllegalCollectionNameException(BaseException)
            CollectionNotExistException(BaseException)
            PartitionTagNotExistException(BaseException)
            IllegalPartitionTagException(BaseException)

        """
        pass

    """ CRUD
    """

    def insert(self, collection_name, entities, ids=None, partition_name=None, params=None):
        """
        Insert vectors to a collection.

        :param collection_name:
        :type  collection_name: str

        :param entities:
        :type  entities: dict
        `[
                {"field": "A", "values": A_list, "type": DataType.Int64},
                {"field": "B", "values": A_list, "type": DataType.Int64},
                {"field": "C", "values": A_list, "type": DataType.Int64},
                {"field": "Vec", "values": vec, "type": DataType.VECTOR}
        ]`

        :type  collection_name: str
        :param collection_name: Name of the collection to insert entities to.

        :type partition_name: str or None.
            If partition_name is None, entities will be inserted to the collection rather than partitions.

        :param partition_name: Tag of a partition.

       :return:
            ids: list[int]

       :raises:
            CollectionNotExistException(BaseException)
            IllegalCollectionNameException(BaseException)
            InvalidRowRecordException(BaseException)
            InvalidVectorIdException(BaseException)
            PartitionTagNotExistException(BaseException)
            InvalidPartitionTagException(BaseException)
            FieldsNotMatchException(BaseException)
        """
        pass

    def delete_entity_by_id(self, collection_name, ids, timeout=None):
        """
        Deletes entitiess in a collection by entity ID.

        :param collection_name: Name of the collection.
        :type  collection_name: str

        :param ids: list of entity id
        :type  ids: list[int]

        :return:
            Status: Whether the operation is successful.
                - ID_NOT_EXIST
                - DELETED

        :raises:
            CollectionNotExistException(BaseException)
            IllegalCollectionNameException(BaseException)
            InvalidEntityIdException(BaseException)
            LimitMaxIdException(BaseException)

        """
        pass

    def get_entity_by_id(self, collection_name, ids, fields=None, timeout=None):
        """
        Returns raw vectors according to ids.

        :param collection_name: Name of the collection
        :type collection_name: str

        :param ids: list of vector id
        :type ids: list

        :return:
            entities:
                `collection mappings ["A", "B", "Vec"]
                 access value of field "A" in first result:
                    a = entities[0].A
                `

        :raises:
            CollectionNotExistException(BaseException)
            InvalidEntityIdException(BaseException)
            IllegalCollectionNameException(BaseException)
            TODO: exception for field not match
        """
        pass

    def list_id_in_segment(self, collection_name, segment_name, timeout=None):
        """
        :returns:
            ids: list[int]

        :raises:
            CollectionNotExistException(BaseException)
            IllegalCollectionNameException(BaseException)

        """
        pass

    def search(self, collection_name, query_entities, partition_names=None, fields=None, **kwargs):
        """
        :param collection_name:
        :type  collection_name: str

        :param query_entities:
        :type  query_entities: dict
        TODO: update example

             `{
                 "bool": {
                     "must": [
                         {"term": {"A": {"values": [1, 2, 5]}}},
                         {"range": {"B": {"ranges": {"GT": 1, "LT": 100}}}},
                         {"vector": {"Vec": {"topk": 10, "query": vec[: 1], "params": {"index_name": Indextype.IVF_FLAT, "nprobe": 10}}}}
                     ],
                 },
             }`
            OR
             `{
                 "bool": {
                     "must": [
                         {"vector": {"Vec": {"topk": 10, "query": vec[: 1], "params": {"index_name": Indextype.IVF_FLAT, "nprobe": 10}}}}
                     ],
                 },
             }`

        :param partition_names: partition tag list
        :type  partition_names: list[str]

        :param params: extra params.
        :type prams: dict

        :return
            result: query result

        :raises:
            CollectionNotExistException(BaseException)
            IllegalCollectionNameException(BaseException)
            InvalidTopkException(BaseException)
            InvalidSearchParamException(BaseException)
            PartitionTagNotExistException(BaseException)
            InvalidPartitionTagException(BaseException)

        """
        pass

    def search_in_segment(self, collection_name, segment_ids, query_entities, params=None, timeout=None):
        """
        :param collection_name:
        :type  collection_name: str

        :param segment_ids:
        :type  segment_ids: list[int]

        :param query_entities:
        :type  query_entities: dict

        :param params: extra params.
        :type prams: dict

        :return
            result: query result

        :raises:
            CollectionNotExistException(BaseException)
            IllegalCollectionNameException(BaseException)
            InvalidTopkException(BaseException)
            InvalidSearchParamException(BaseException)
            PartitionTagNotExistException(BaseException)
            InvalidPartitionTagException(BaseException)

        """
        pass

    def calc_distance(self, vectors_left, vectors_right, params=None, timeout=None, **kwargs):
        """
        Calculate distance between two vector arrays.

        :param vectors_left: The vectors on the left of operator.
        :type  vectors_left: dict
        `{"ids": [1, 2, 3, .... n], "collection": "c_1", "partitions": ["p_1", "p_2"], "field": "v_1"}`
        or
        `{"float_vectors": [[1.0, 2.0], [3.0, 4.0], ... [9.0, 10.0]]}`
        or
        `{"bin_vectors": [b'\x94', b'N', ... b'\xca']}`

        :param vectors_right: The vectors on the right of operator.
        :type  vectors_right: dict
        `{"ids": [1, 2, 3, .... n], "collection": "col_1", "partitions": ["p_1", "p_2"], "field": "v_1"}`
        or
        `{"float_vectors": [[1.0, 2.0], [3.0, 4.0], ... [9.0, 10.0]]}`
        or
        `{"bin_vectors": [b'\x94', b'N', ... b'\xca']}`

        :param params: parameters, currently only support "metric_type", default value is "L2"
                       extra parameter for "L2" distance: "sqrt", true or false, default is false
                       extra parameter for "HAMMING" and "TANIMOTO": "dim", set this value if dimension is not a multiple of 8, otherwise the dimension will be calculted by list length
        :type  params: dict
            There are examples of supported metric_type:
                `{"metric": "L2"}`
                `{"metric": "IP"}`
                `{"metric": "HAMMING"}`
                `{"metric": "TANIMOTO"}`
            Note: "L2", "IP", "HAMMING", "TANIMOTO" are case insensitive

        :return: 2-d array distances
        :rtype: list[list[int]] for "HAMMING" or list[list[float]] for others
            Assume the vectors_left: L_1, L_2, L_3
            Assume the vectors_right: R_a, R_b
            Distance between L_n and R_m we called "D_n_m"
            The returned distances are arranged like this:
              [D_1_a, D_1_b, D_2_a, D_2_b, D_3_a, D_3_b]
        """
        pass

    """ Memory
    """

    def load_collection(self, collection_name, timeout=None):
        """
        Loads a collection for cache.

        :type collection_name: str

        :param collection_name: collection to load

        :return:
            None

        :raises:
            CollectionNotExistException(BaseException)
            IllegalCollectionNameException(BaseException)

        """
        pass

    def reload_segments(self, collection_name, segment_ids):
        """
            Load segment delete docs to cache

        : return:
            None

        :raises:
            CollectionNotExistException(BaseException)
            IllegalCollectionNameException(BaseException)

        """
        pass

    def flush(self, collection_names=None, timeout=None, **kwargs):
        """
        Flushes vector data in one collection or multiple collections to disk.

        :type  collection_name_array: list
        :param collection_name: Name of one or multiple collections to flush.

        :return:
            None

        :raises:
            CollectionNotExistException(BaseException)
            IllegalCollectionNameException(BaseException)

        """
        pass

    """ Config
    """

    def get_config(self, key):
        """
        Gets Milvus configurations.

        """
        pass

    def set_config(self, key, value):
        """
        Sets Milvus configurations.

        """
        pass

    def release_collection(self, collection_name, timeout=30):

        pass

    def load_partitions(self, collection_name, partition_names):

        pass

    def release_partitions(self, collection_name, partition_names):

        pass

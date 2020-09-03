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

    def get_collection_info(self, collection_name, timeout=30):
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

    def count_entities(self, collection_name, timeout=30):
        """
        Returns the number of entities in target collection.

        :type  collection_name: str
        :param collection_name: target collection name.

        :returns:
            count: int, count of entities

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
    # def get_index_info(self, collection_name, field_name, index_name, timeout=30):
    #     """
    #     Show index information of a collection.
    #
    #     :type collection_name: str
    #     :param collection_name: table name been queried
    #
    #     :returns:
    #         IndexSchema:
    #
    #     :raises:
    #         CollectionNotExistException(BaseException)
    #         IllegalCollectionNameException(BaseException)
    #         IllegalFieldNameException(BaseException)
    #         IllegalIndexNameException(BaseException)
    #
    #     """
    #     pass

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

    def create_partition(self, collection_name, partition_tag, timeout=30):
        """
        create a partition for a collection.

        :param collection_name: Name of the collection.
        :type  collection_name: str

        :param partition_name: Name of the partition.
        :type  partition_name: str

        :param partition_tag: Name of the partition tag.
        :type  partition_tag: str

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

    def has_partition(self, collection_name, partition_tag):
        """
        Check if specified partition exists.


        :param collection_name: target table name.
        :type  collection_name: str

        :param partition_tag: partition tag.
        :type  partition_tag: str

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

    def drop_partition(self, collection_name, partition_tag, timeout=30):
        """
        Deletes a partition in a collection.

        :param collection_name: Collection name.
        :type  collection_name: str

        :param partition_tag: Partition name.
        :type  partition_tag: str

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

    def insert(self, collection_name, entities, ids=None, partition_tag=None, params=None):
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

        :type partition_tag: str or None.
            If partition_tag is None, entities will be inserted to the collection rather than partitions.

        :param partition_tag: Tag of a partition.

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

    def search(self, collection_name, query_entities, partition_tags=None, fields=None, **kwargs):
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

        :param partition_tags: partition tag list
        :type  partition_tags: list[str]

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

    def compact(self, collection_name, timeout=None, **kwargs):
        """
        Compacts segments in a collection. This function is recommended after deleting vectors.

        :type  collection_name: str
        :param collection_name: Name of the collections to compact.

        :return:
            Status:
                - SUCCESS_BUT_NOT_COMPACT
                - COMPACTED

        :raises:
            CollectionNotExistException(BaseException)
            IllegalCollectionNameException(BaseException)

        """
        pass

    """ Config
    """

    def get_config(self, parent_key, child_key):
        """
        Gets Milvus configurations.

        """
        pass

    def set_config(self, parent_key, child_key, value):
        """
        Sets Milvus configurations.

        """
        pass

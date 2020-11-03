**v0.4.0(Developing)**

**v0.3.0**
  * Incompatibly upgrade APIs for supporting hybrid functionality. The passing parameters of the following APIs has been changed:
    - [create_collection()](api.html#milvus.Milvus.create_collection)
    - [insert()](api.html#milvus.Milvus.insert)
    - [create_index()](api.html#milvus.Milvus.create_index)
    - [drop_index()](api.html#milvus.Milvus.drop_index)
    - [search()](api.html#milvus.Milvus.search)
    - [search_in_segment()](api.html#milvus.Milvus.search_in_segment)
    - [get_entity_by_id()](api.html#milvus.Milvus.get_entity_by_id)

  * Add passing parameter `threshold` in API [compact()](api.html#milvus.Milvus.compact)
  * Raise exception if the status of returned values from server is not OK for all the APIs.
  * Remove parameter `status` in return values of APIs except [compact()](api.html#milvus.Milvus.compact) and 
    [delete_entity_by_id()](api.html#milvus.Milvus.delete_entity_by_id) where status is reserved experimentally.

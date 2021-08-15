#### pymilvus.utility

---



##### Checking job states

| Methods                                                      | Description                                                  | 参数                                                         | 返回值                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| utility.loading_progress(collection_name, partition_names=[], using="default") | Show # loaded entities vs. # total entities                  | collection_name 类型是string<br />partition_names 类型是 list | dict{<br />num_loaded_entities: int,<br />num_total_entities:int} |
| utility.wait_for_loading_complete(collection_name, partition_names=[], timeout=None,using="default") | Block until loading is done or Raise Exception after timeout. | collection_name 类型是 string<br />partition_names 类型是 list | None或Raise Exception                                        |
| utility.index_building_progress(collection_name, index_name="",using="default") | Show # indexed entities vs. # total entities                 | collection_name 类型是 string<br />index_name 类型是 string  | dict{<br />num_indexed_entities: int,<br />num_total_entities:int} |
| utility.wait_for_index_building_complete(collection_name, index_name, timeout = None,using="default") | Block until building is done or Raise Exception after timeout. | collection_name 类型是string<br />partition_name 类型是 string<br />timeout 类型是 int (秒) | None或Raise Exception                                        |
| utility.has_collection(collection_name,using="default")      | Checks whether a specified collection exists.                | collection_name 类型是string                                 | boolean                                                      |
| utility.has_partition(collecton_name, partition_name,using="default") | Checks if a specified partition exists in a collection.      | collection_name 类型是string<br />partition_name 类型是 string | boolean                                                      |
| utility.list_collections(timeout=None, using="default")      | Returns a list of all collection names                       |                                                              | list(string)                                                 |


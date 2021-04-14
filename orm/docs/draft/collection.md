#### pymilvus.Collection

---



##### Accessing and constructing collection

| Methods                                            | Descriptions                                 | 参数                                                         | 返回值         |
| -------------------------------------------------- | :------------------------------------------- | ------------------------------------------------------------ | -------------- |
| Collection(name, data=None, schema=None, **kwargs) | 创建Collection，如果不存在同名的，则新建一个 | name 类型 string<br />data 类型 是pandas.DataFrame<br />schema 类型 是CollectionSchema<br />kwargs，暂时为空 | Collection对象 |
|                                                    |                                              |                                                              |                |



##### Manipulating and querying collection meta

| Properties              | Descriptions                  | 参数 | 返回值                  |
| ----------------------- | ----------------------------- | ---- | ----------------------- |
| Collection.schema       | Return the collection schema. | /    | CollectionSchema 对象   |
| Collection.description  | 返回自定义描述                | /    | 类型 string，自定义描述 |
| Collection.name         | 返回collection名字            | /    | 类型 string, 名字       |
| Collection.is_empty     | 是否为空                      | /    | 类型 boolean            |
| Collection.num_entities | 返回行数                      | /    | 类型int                 |



##### Manipulating, loading, and querying collection

| Methods                                                      | Descriptions                                                 | 参数                                                         | 返回值                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------- |
| Collection.drop(**kwargs)                                    | Drop the collection, as well as its indexes.                 | /                                                            | None 或 Raise Exception       |
| Collection.load(field_names=None, index_names=None, partition_names=None, **kwargs) | Load the collection from disk to memory.                     | field_names   类型是 list(string)<br />index_names 类型是 list(string)<br />partitions_names 类型是 list(string)<br />kwargs 暂时为空 | None或者Raise Exception       |
| Collection.release(**kwargs)                                 | Release the collection from memory.                          | /                                                            | None或者Raise Exception       |
| Collection.insert(data, partition_name ="", **kwargs)        | Insert data into the collection, or into one of its partitions. | data 类型是list-like(list, tuple, numpy.ndarray) 对象或者pandas.DataFrame，data的维度需要和列的数目对齐<br />partition_name 类型是 string | None或者Raise Exception       |
| Collection.search(data, params, limit, expr="", partition_names=None, fields=None, **kwargs) | Vector similarity search with an optional boolean expression as filters. | data是 list-like(list, tuple, numpy.ndarray)<br /><br />params 类型是 dict<br /><br />limit 类型是 int <br />expr 类型是string<br />partitions_names类型是 list(string)<br />fields类型是list(string)<br />kwargs 目前为空 | Search对象或者Raise Exception |



##### Accessing and constructing partition

| Methods                                             | Descriptions                                                 | 参数                       | 返回值                  |
| --------------------------------------------------- | ------------------------------------------------------------ | -------------------------- | ----------------------- |
| Collection.partitions                               | Return all partitions of the collection.                     | /                          | list(Partition对象)     |
| Collection.partition(partition_name)                | Return the partition corresponding to name. Create a new one if not existed. | partition_name类型是string | Partition对象           |
| Collection.has_partition(partition_name)            | Checks if a specified partition exists.                      | partition_name类型是string | boolean                 |
| Collection.drop_partition(partition_name, **kwargs) | Drop the partition and its corresponding index files.        | partition_name类型是string | None或者Raise Exception |



##### Accessing and constructing index

| Methods                                                      | Descriptions                                                 | 参数                                                         | 返回值                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------- |
| Collection.indexes                                           | Return all indexes of the collection.                        | /                                                            | list(Index对象)               |
| Collection.index(index_name)                                 | Return the index corresponding to name.                      | index_name类型是 string                                      | None或者Index对象             |
| Collection.create_index(field_name, index_name, index_params, **kwargs) | Create index on a specified column according to the index parameters. Return Index Object. | field_name类型是string<br />index_params类型是dict<br />index_name类型是 string | Index对象或者 Raise Exception |
| Collection.drop_index(index_name, **kwargs)                  | Drop index and its corresponding index files.                | index_name类型是string                                       | None或者Raise Exception       |


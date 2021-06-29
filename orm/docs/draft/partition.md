#### pymilvus.Partition

---

##### Manipulating and querying partition meta

| Methods                                     | Descriptions                           | 参数 | 返回值  |
| ------------------------------------------- | -------------------------------------- | ---- | ------- |
| Partition(collection, name, description="") | collection类型是 Collection<br />      |      |         |
| Partition.description                       | Return the description text.           | /    | string  |
| Partition.name                              | Return the partition name.             | /    | string  |
| Partition.is_empty                          | Return whether the partition is empty. | /    | boolean |
| Partition.num_entities                      | Return the number of entities.         | /    | int     |

##### Manipulating, loading, and querying partition

| Methods                                                      | Descriptions                                                 | 参数                                                         | 返回值                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------- |
| Partition.drop(**kwargs)                                     | Drop the partition, as well as its corresponding index files. | kwargs reversed.目前为空                                     | None或者Raise Exception                                 |
| Partition.load(field_names=None, index_names=None, **kwargs) | Load the partition from disk to memory.                      | field_names  类型是list(string)<br />index_names类型是list(string)<br />kwargs reversed.目前为空 | None或者Raise Exception                                 |
| Partition.release(**kwargs)                                  | Release the partition from memory.                           | kwargs reversed.目前为空                                     | None或者Raise Exception                                 |
| Partition.insert(data, **kwargs)                             | Insert data into partition.                                  | data 类型是list-like(list, tuple, numpy.ndarray) 对象或者pandas.DataFrame，data的维度需要和列的数目对齐<br />kwargs可以是 sync=False | None或者InsertFuture或者Raise Exception                 |
| Partition.search(data, anns_field, params, limit, expr=None, output_fields=None, **kwargs) | Vector similarity search with an optional boolean expression as filters. | data是 list-like(list, tuple),或者pd.Series <br />anns_field 类型是 string, 表示在哪个列上进行向量的近似查询<br />params 类型是 dict<br /><br />limit 类型是 int <br />expr 类型是string<br />output_fields类型是list(string)<br />kwargs 可以是 async=False | SearchResultFuture或者 SearchResult 或者Raise Exception |
| Partition.query(expr="", output_fields=None) | Query with a set of criteria, and results in a list of records that match the query exactly. | expr 类型是string<br />output_fields 类型是 list(string), 表示要返回哪些列的原始数据 | dict 或者Raise Exception      |



#### pymilvus.Partition

---

##### Manipulating and querying partition meta

| Methods                | Descriptions                           | 参数 | 返回值  |
| ---------------------- | -------------------------------------- | ---- | ------- |
| Partition.description  | Return the description text.           | /    | string  |
| Partition.name         | Return the partition name.             | /    | string  |
| Partition.is_empty     | Return whether the partition is empty. | /    | boolean |
| Partition.num_entities | Return the number of entities.         | /    | int     |

##### Manipulating, loading, and querying partition

| Methods                                                      | Descriptions                                                 | 参数                                                         | 返回值                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------- |
| Partition.drop(**kwargs)                                     | Drop the partition, as well as its corresponding index files. | /                                                            | None或者Rase Exception        |
| Partition.load(field_names=None, index_names=None, **kwargs) | Load the partition from disk to memory.                      | field_names  类型是list(string)<br />index_names类型是list(string)<br /> | None或者Rase Exception        |
| Partition.release(**kwargs)                                  | Release the partition from memory.                           | /                                                            | None或者Rase Exception        |
| Partition.insert(data, **kwargs)                             | Insert data into partition.                                  | data 类型是list-like(list, tuple, numpy.ndarray) 对象或者pandas.DataFrame，data的维度需要和列的数目对齐<br /> | None或者Rase Exception        |
| Partition.search(data, params, limit, expr=None, fields=None, **kwargs) | Vector similarity search with an optional boolean expression as filters. | data是 list-like(list, tuple, numpy.ndarray)<br /><br />params 类型是 dict<br /><br />limit类型是int<br />expr 类型是string<br />fields类型是list(string) | Search对象或者Raise Exception |



#### pymilvus.Index

---

##### Accessing and constructing Index

| Methods                                              | Descriptions                                                 | 参数                                                         | 返回值                       |
| ---------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------- |
| Index(collection, field_name, index_params, name="") | Create index on a specified column according to the index parameters. | collection类型是 Collection<br />name 类型是string<br />field_name 类型是string<br />index_params类型是dict | Index对象或者Raise Exception |
| Index.name                                           | Return the index name.                                       | /                                                            | string                       |
| Index.params                                         | Return the index params.                                     | /                                                            | dict (克隆)                  |
| Index.collection_name                                | Return corresponding collection name.                        | /                                                            | string                       |
| Index.field_name                                     | Return corresponding column name.                            | /                                                            | string                       |

##### Manipulating

| Methods      | Descriptions                                  | 参数 | 返回值                  |
| ------------ | --------------------------------------------- | ---- | ----------------------- |
| Index.drop() | Drop index and its corresponding index files. | /    | None或者Raise Exception |

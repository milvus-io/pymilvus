#### pymilvus.Index

---

##### Accessing and constructing Index

| Methods                                              | Descriptions                                                 | 参数                                                         | 返回值                       |
| ---------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------- |
| Index(collection, field_name, index_params, name="") | Create index on a specified column according to the index parameters. | collection类型是 Collection<br />name 类型是 string<br />field_name 类型是 string<br />index_params 类型是 dict | Index 对象或者 Raise Exception |
| Index.name                                           | Return the index name.                                       | /                                                            | string                       |
| Index.params                                         | Return the index params.                                     | /                                                            | dict (克隆)                  |
| Index.collection_name                                | Return corresponding collection name.                        | /                                                            | string                       |
| Index.field_name                                     | Return corresponding column name.                            | /                                                            | string                       |

##### Manipulating

| Methods      | Descriptions                                  | 参数 | 返回值                  |
| ------------ | --------------------------------------------- | ---- | ----------------------- |
| Index.drop() | Drop index and its corresponding index files. | /    | None 或者 Raise Exception |

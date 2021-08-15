### CollectionSchema

| Methods              | Descriptions                           | 参数描述 | 返回值 |
| -------------------- | -------------------------------------- | -------------------- | -------------------- |
| CollectionSchema(fields, description="", **kwargs) | 构造一个CollectionSchema对象 | 参数fields是一个 list-like的对象，每个元素是FieldSchema对象<br />description 类型 string 自定义描述 | CollectionSchema对象或者Raise Exception |
| CollectionSchema.fields | 返回所有的列 | /                                                            | list，每个元素是一个 FieldSchema 对象 |
| CollectionSchema.description | 返回自定义描述 | /                                                            | string 自定义描述                    |
| CollectionSchema.primary_field | 返回主键列的FieldSchema | /                                                            | None 或 FieldSchema 对象        |
| CollectionSchema.auto_id | 是否自动生成主键 | /                                                            | bool                                  |
|  |  |                                                              |                                       |



### FieldSchema



| Methods                                             | Descriptions            | 参数描述                                                     | 返回值                                     |
| --------------------------------------------------- | ----------------------- | ------------------------------------------------------------ | ------------------------------------------ |
| FieldSchema(name, dtype,  description="", **kwargs) | 构造一个FieldScheam对象 | name 参数类型是string<br />dtype参数类型是 名为 DataType 的 python enum<br />description 类型是 string，自定义描述 | FieldScheam对象或者Raise Exception         |
|                                                     |                         |                                                              |                                            |
| FieldSchema.name                                    | 列名                    | /                                                            | string                                     |
| FieldSchema.dtype                                   | 返回数据类型            | /                                                            | DataType                                   |
| FieldSchema.description                             | 返回自定义描述          | /                                                            | string, 自定义描述                         |
| FieldSchema.xxx                                     | 其他属性                | /                                                            | None 或者确定的值<br />比如ndim, str_len等 |



#### DataType


| DataType Enum  |
| ----------------------- |
| DataType.BOOL |
| DataType.INT8 |
| DataType.INT16 |
| DataType.INT32 |
| DataType.INT64 |
| DataType.FLOAT |
| DataType.DOUBLE |
| DataType.BINARY_VECTOR |
| DataType.FLOAT_VECTOR |



### 例子

```python
fields = [
    FieldSchema("A", DataType.INT32, True),
    FieldSchema("B", DataType.INT64),
    FieldSchema("C",  DataType.FLOAT),
   FieldSchema("Vec", DataType.FLOAT_VECTOR)]

schema = Schema(fields, description = "This is a test collection.")

assert len(schema.fields()) == len(fields)
```

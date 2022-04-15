# Collection()

This is the constructor method to create a collection with the specified schema or to get an existing collection with the name.

## Invocation

```python
Collection(name, schema=None, using='default', shards_num=2, **kwargs)
```

## Parameters

| Parameter    | Description                                                  | Type                            | Required |
| ------------ | ------------------------------------------------------------ | ------------------------------- | -------- |
| `name`       | Name of the collection                                       | String                          | True     |
| `schema`     | Schema of the collection to create                           | class *schema.CollectionSchema* | False    |
| `using`      | Milvus Connection used to create the collection              | String                          | False    |
| `shards_num` | Shard number of the collection to create. <br/>It corresponds to the number of data nodes used to insert data. | INT32   | False    |
| `kwargs` <ul><li>consistency_level</li></ul>     | <br/><ul><li>With which <a href="https://github.com/milvus-io/milvus/blob/master/docs/developer_guides/how-guarantee-ts-works.md">consistency level</a> to create the collection</li></ul> | String/Integer                  | False    |

A schema specifies the properties of a collection and the fields within. See [Schema](https://milvus.io/docs/v2.0.x/schema.md) for more information.

## Return

A new collection object created with the specified schema or an existing collection object by name.

### Properties

| Property        | Description                                                  | Type                            |
| --------------- | ------------------------------------------------------------ | ------------------------------- |
| `name`          | Name of the collection                                       | String                          |
| `schema`        | Schema of the collection                                     | class *schema.CollectionSchema* |
| `description`   | Description of the collection                                | String                          |
| `is_empty`      | Boolean value to indicate if the collection is empty         | Bool                            |
| `num_entities`  | Number of entities in the collection                         | Integer                         |
| `primary_field` | Schema of the primary field in the collection                | class *schema.FieldSchema*      |
| `partitions`    | List of all partitions in the collection                     | list[String]                    |
| `indexes`       | List of all indexes in the collection                        | list[String]                    |

## Raises

`CollectionNotExistException`: error if the collection does not exist.

## Example

```python
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection
book_id = FieldSchema(
  name="book_id", 
  dtype=DataType.INT64, 
  is_primary=True, 
)
word_count = FieldSchema(
  name="word_count", 
  dtype=DataType.INT64,  
)
book_intro = FieldSchema(
  name="book_intro", 
  dtype=DataType.FLOAT_VECTOR, 
  dim=2
)
schema = CollectionSchema(
  fields=[book_id, word_count, book_intro], 
  description="Test book search"
)
collection_name = "book"
collection = Collection(
    name=collection_name, 
    schema=schema, 
    using='default', 
    shards_num=2,
    consistency_level="Strong"
)
collection.schema
{
  auto_id: False
  description: Test book search
  fields: [{
    name: book_id
    description: 
    type: 5
    is_primary: True
    auto_id: False
  }, {
    name: word_count
    description: 
    type: 5
  }, {
    name: book_intro
    description: 
    type: 101
    params: {'dim': 2}
  }]
}
collection.description
'Test book search'
collection.name
'book'
collection.is_empty
True
collection.primary_field
{
    name: book_id
    description: 
    type: 5
    is_primary: True
    auto_id: False
  }
```
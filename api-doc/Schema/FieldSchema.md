# FieldSchema()

This is the constructor method to create a FieldSchema.

## Invocation

```python
FieldSchema(name, dtype, description='', **kwargs)
```

## Return

A FieldSchema object.

### Properties

| Property             | Description                                                                  |
| -------------------- | ---------------------------------------------------------------------------- |
| `name`               | Name of the field                                                            |
| `is_primary`         | Boolean value that indicates if the field is the primary key field           |

## Example

```python
from pymilvus import CollectionSchema, FieldSchema, DataType
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
```
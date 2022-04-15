# CollectionSchema()

This is the constructor method to create a CollectionSchema.

## Invocation

```python
CollectionSchema(fields, description='', **kwargs)
```

## Return

A CollectionSchema object.

### Properties

| Property             | Description                                                                  |
| -------------------- | ---------------------------------------------------------------------------- |
| `fields`             | List of FieldSchema                                                          |
| `description`        | Description of the CollectionSchema                                          |
| `auto_id`            | Boolean value that indicates if the primary keys are automatically generated |

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
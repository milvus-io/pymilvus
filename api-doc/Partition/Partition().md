# Partition()

This is the constructor method to create a partition in the specified collection.

## Invocation

```python
Partition(collection, name, description='', **kwargs)
```

## Parameters

| Parameter    | Description                                                  | Type                            | Required |
| ------------ | ------------------------------------------------------------ | ------------------------------- | -------- |
| `collection` | Name of the collection                                       | String                          | True     |
| `name`       | Name of the partition to create                              | String                          | True     |
| `description`   | Description of the collection                             | String                          | False    |

## Return

A new partition object created with the specified name.

### Properties

| Property        | Description                                                  | Type                            |
| --------------- | ------------------------------------------------------------ | ------------------------------- |
| `name`          | Name of the partition                                        | String                          |
| `description`   | Description of the collection                                | String                          |
| `is_empty`      | Boolean value to indicate if the partition is empty          | Bool                            |
| `num_entities`  | Number of entities in the partition                          | Integer                         |


## Example

```python
from pymilvus import Partition
partition = Partition("book", "novel", "")
```
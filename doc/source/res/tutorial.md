## Prerequisites
Before we start, there are some prerequisites. Make sure that:
- You have a running Milvus instance.
- PyMilvus is correctly installed.

In Python shell, your installation of PyMilvus is ok if the following command doesn't raise an exception:
```python
>>> from milvus import Milvus, DataType
```

## Connect to Milvus
First of all, we need to make connection with Milvus server after importing.
By default Milvus runs on localhost in port 19530, so you can use default value to connect to Milvus.

```python
>>> host = '127.0.0.1'
>>> port = '19530'
>>> client = Milvus(host, port)
```

## Collection
## Create Collection
## Create Partition
## Get Collection Stats and Info
## Entities
## Insert Entities
## Flush
## Count Entites
## Get Entities
  ### Get Entities by ID
  ### Search Entities by Vector Similiarity
  ### Get Entities filtered by fields.
## Deletion
  ### Delete Entities by ID
  ### Compact
  ### Drop a Partition
  ### Drop a Collection

========
Tutorial
========

This is a basic introduction to Milvus by PyMilvus.

For a runnable python script,
checkout `example.py <https://github.com/milvus-io/pymilvus/blob/master/examples/example.py>`_ on PyMilvus Github,
or `hello milvus <https://milvus.io/docs/v2.0.0/example_code.md>`_ on Milvus official website. It's a good recommended
start to get started with Milvus and PyMilvus as well.


.. note::
   This tutorial uses float vectors as example vector field data, if you want to learn example about binary vectors, see
   `binary vector example <https://github.com/milvus-io/pymilvus/blob/master/examples/collection.py>`_.


Prerequisites
=============

Before you start, there are some prerequisites.

Make sure that:

* You have a running Milvus instance.
* PyMilvus is correctly installed, see :doc:`install`.

Connect to Milvus
=================

First of all, you need to import `pymilvus`.

>>> from pymilvus import connections

Then, you can make connection with Milvus server.
By default Milvus runs on localhost in port 19530, so you can use default value to connect to Milvus.

>>> connections.connect() # connect by default value

Or you can add other Milvus server address by:

>>> host = '127.0.0.1'
>>> port = '19530'
>>> connections.add_connection(dev={"host": host, "port": port})

After connecting, you can communicate with Milvus in the following ways. If you are confused about the
terminology, see `Milvus Terminology <https://milvus.io/docs/v2.0.0/glossary.md>`_ for explanations.


Collection
==========

Now it's time to create a new collection. You can list all the collections already exist. For a brand
new Milvus running instance, the result should be empty.

>>> from pymilvus import utility
>>> utility.list_collections()
[]

Create Collection
=================

To create a collection, you need to provide schema for it.

In this tutorial, you will create a collection with three fields: `id`, `year` and `embedding`.

- The type of `id` field is `int64`, and it is set as primary field.
- The type of `year` field is `int64`, and the type of `embedding` is `FLOAT_VECTOR` whose `dim` is 128.

>>> from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
>>> dim = 128
>>> fields = [
...     FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
...     FieldSchema(name="year", dtype=DataType.INT64, description="year"),
...     FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
... ]
>>> schema = CollectionSchema(fields, description='desc of collection')
>>> collection_name = "tutorial"
>>> tutorial = Collection(collection_name, schema, consistency_level="Strong")

Then you can list collections and 'tutorial' will be in the result.

>>> utility.list_collections()
['tutorial']


This tutorial is a basic intro tutorial, building index won't be covered by this tutorial.
If you want to go further into Milvus with indexes, it's recommended to check our
`index examples <https://github.com/milvus-io/pymilvus/blob/master/examples/example_index.py>`_.

If you're already known about indexes from ``index examples``, and you want a full lists of params supported
by PyMilvus, you check out `Index <https://milvus.io/api-reference/pymilvus/v2.0/param.html>`_
chapter of the PyMilvus documentation.

Further more, if you want to get a thorough view of indexes, check our official website for
`Vector Index <https://milvus.io/docs/index.md>`_.

Create Partition
================

If you don't create a partition, there will be a default one called "``_default``", all the entities will be
inserted into the "``_default``" partition. You can check it by ``Collection.partitions()``

>>> tutorial.partitions
[{"name": "_default", "description": "", "num_entities": 0}]

You can provide a partition name to create a new partition.

>>> tutorial.create_partition("comedy")
>>> tutorial.partitions
[{"name": "_default", "description": "", "num_entities": 0}, {"name": "comedy", "description": "", "num_entities": 0}]

Insert Entities
===============

An entity is a group of fields that corresponds to real world objects. In this tutorial, collection has three fields.
Here is an example of 30 entities structured in list of list.

.. note:
   The field `id` was set as primary and not auto_id above, so we shall input the value for it when inserting.

>>> import random
>>> num_entities = 30
>>> entities = [
...     [i for i in range(num_entities)], # field id
...     [random.randrange(1949, 2021) for _ in range(num_entities)],  # field year
...     [[random.random() for _ in range(128)] for _ in range(num_entities)],  # field embedding
... ]
>>> insert_result = tutorial.insert(entities)
>>> insert_result
(insert count: 30, delete count: 0, upsert count: 0, timestamp: 430704946903515140)

.. note:
   If ``partition_name`` isn't provided, these entities will be inserted into the "``_default``" partition,
   otherwise, them will be inserted into specified partition.


Search
======

Search Entities by Vector Similarity
------------------------------------

You can get entities by vector similarity. Assuming there's a ``embedding_A``, and you want to get top 2 entities that are most similar with it.

In the example below, you can perform search based on vector similarity.

.. note:
    Before searching, you need to load this collection into memory.

>>> tutorial.load()

>>> embedding_A = [[random.random() for _ in range(128)] for _ in range(1)]
>>> search_params = {"metric_type": "L2"}
>>> results = tutorial.search(embedding_A, "embedding", search_params, limit=2, expr="year > 2001", output_fields=["year", "id"])

.. note::
    For more about the parameter expr, please refer to: https://github.com/milvus-io/milvus/blob/master/docs/design_docs/query_boolean_expr.md

.. note::
    If the collection is index-built, you need to specify search param, and pass parameter `search_params` like: `collection.search(..., search_params={...})`.
    You can refer to `Index params <https://milvus.io/cn/api-reference/pymilvus/v2.0/param.html>`_ for more details.

.. note::
    If parameter `partition_names` is specified, milvus executes search request on these partition instead of whole collection.

The returned ``results`` is a 2-D like structure, 1 for 1 entity querying, 2 for limit 2. For more clarity, we obtain
the film as below. If you want to know how to deal with search result in a better way, you can refer to
`search result <https://milvus.io/cn/api-reference/pymilvus/v2.0/results.html>`_ in PyMilvus doc.

>>> result = results[0]
>>> embedding_1 = result[0]
>>> embedding_2 = result[1]

Then how do we get ids, distances and fields? It's as below.

.. note::
   Because vectors are randomly generated, so the retrieved vector id and distance may differ.

>>> embedding_1
(distance: 18.32852554321289, id: 5)

>>> print(embedding_1.entity)
id: 5, distance: 18.32852554321289, entity: {'year': 2003, 'id': 5},

>>> embedding_1.entity.id
5

>>> embedding_1.entity.year
2003


Drop a Partition
----------------

You can also drop a partition.

.. Danger::
   Once you drop a partition, all the data in this partition will be deleted too.

>>> tutorial.drop_partition("comedy")


Drop a Collection
-----------------

Finally, you can drop an entire collection.

.. Danger::
   Once you drop a collection, all the data in this collection will be dropped too.

>>> utility.drop_collection(tutorial.name)

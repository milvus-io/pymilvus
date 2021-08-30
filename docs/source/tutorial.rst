========
Tutorial
========

This is a basic introduction to Milvus by PyMilvus.

For a runnable python script,
checkout `example.py <https://github.com/milvus-io/pymilvus/blob/main/examples/example.py>`_ on PyMilvus Github,
or `hello milvus <https://milvus.io/docs/v2.0.0/example_code.md>`_ on Milvus official website. It's a good recommended
start to get started with Milvus and PyMilvus as well.


.. note::
   Here we use float vectors as example vector field data, if you want to learn example about binary vectors, see
   `binary vector example <https://github.com/milvus-io/pymilvus/blob/main/examples/collection.py>`_.


Prerequisites
=============

Before we start, there are some prerequisites.

Make sure that:

- You have a running Milvus instance.
- PyMilvus is correctly installed, see :doc:`install`.

Connect to Milvus
=================

First of all, we need to import `pymilvus`.

>>> from pymilvus import connections

Then, we can make connection with Milvus server.
By default Milvus runs on localhost in port 19530, so you can use default value to connect to Milvus.

>>> host = '127.0.0.1'
>>> port = '19530'
>>> connections.add_connection(default={"host": host, "port": port})
>>> connections.connect(alias='default')

After connecting, we can communicate with Milvus in the following ways. If you are confused about the
terminology, see `Milvus Terminology <https://milvus.io/docs/v2.0.0/glossary.md>`_ for explanations.


Collection
==========

Now let's create a new collection. Before we start, we can list all the collections already exist. For a brand
new Milvus running instance, the result should be empty.

>>> from pymilvus import list_collections
>>> list_collections()
[]

Create Collection
=================

To create collection, we could provide the schema for it.

In this tutorial, we will create a collection with three fields: `id`, `year` and `embedding`.

The type of 'id' field is `int64`, and it is set as primary field.
The type of `year` field is `int64`, and the type of `embedding` is `FLOAT_VECTOR` whose `dim` is 128.

Now we can create a collection:

>>> from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
>>> dim = 128
>>> id_field = FieldSchema(name="id", dtype=DataType.INT64, description="primary_field")
>>> year_field = FieldSchema(name="year", dtype=DataType.INT64, description="year")
>>> embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
>>> schema = CollectionSchema(fields=[id_field, year_field, embedding_field], primary_field='id', auto_id=True, description='desc of collection')
>>> collection_name = "tutorial"
>>> collection = Collection(name=collection_name, schema=schema)

Then you can list collections and 'tutorial' will be in the result.

>>> list_collections()
['tutorial']

You can also get info of the collection.

>>> collection.description
"desc of collection"


This tutorial is a basic intro tutorial, building index won't be covered by this tutorial.
If you want to go further into Milvus with indexes, it's recommended to check our
`index examples <https://github.com/milvus-io/pymilvus/blob/main/examples/example_index.py>`_.

If you're already known about indexes from ``index examples``, and you want a full lists of params supported
by PyMilvus, you check out `Index <https://milvus.io/api-reference/pymilvus/v2.0.0rc5/param.html>`_
chapter of the PyMilvus documentation.

Further more, if you want to get a thorough view of indexes, check our official website for
`Vector Index <https://milvus.io/docs/index.md>`_.

Create Partition
================

If you don't create a partition, there will be a default one called "``_default``", all the entities will be
inserted into the "``_default``" partition. You can check it by ``Collection.partitions()``

>>> collection.partitions
[{"name": "_default", "description": "", "num_entities": 0}]

You can provide a partition name to create a new partition.

>>> collection.create_partition("new_partition")
>>> collection.partitions
[{"name": "_default", "description": "", "num_entities": 0}, {"name": "new_partition", "description": "", "num_entities": 0}]

Insert Entities
========

An entity is a group of fields that corresponds to real world objects. In this tutorial, collection has three fields.
Here is an example of 30 entities structured in list of list.
.. note:
   The field `id` was set as primary and auto_id above, so we shall not input the value for it when inserting.

>>> import random
>>> nb = 30
>>> years = [i for i in range(nb)]
>>> embeddings = [[random.random() for _ in range(dim)] for _ in range(nb)]
>>> entities = [years, embeddings]
>>> collection.insert(entities)

.. note:
   If ``partition_name`` isn't provided, these entities will be inserted into the "``_default``" partition,
   otherwise, them will be inserted into specified partition.


Search
======

Search Entities by Vector Similarity
------------------------------------

You can get entities by vector similarity. Assuming we have a ``embedding_A`` like below, and we want to get top 2 records whose year is greater than 20
that are most similar with it.

In below example, we search the collection on ``embedding`` field.
.. note:
    Before searching, we need to load data into memory.

>>> nq = 10
>>> embedding_A = [[random.random() for _ in range(dim)] for _ in range(nq)]
>>> anns_field = "embedding"
>>> search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
>>> limit = 2
>>> expr = "year > 20"
>>> collection.load()
>>> results = collection.search(embedding_A, anns_field, search_params, limit, expr)

.. note::
    For more about the parameter expr, please refer to: https://github.com/milvus-io/milvus/blob/master/docs/design_docs/query_boolean_expr.md

.. note::
    If the collection is index-built, user need to specify search param, and pass parameter `search_params` like: `collection.search(..., search_params={...})`.
    You can refer to `Index params <https://milvus.io/cn/api-reference/pymilvus/v2.0.0rc5/param.html>`_ for more details.

.. note::
    If parameter `partition_names` is specified, milvus executes search request on these partition instead of whole collection.

The returned ``results`` is a 2-D like structure, 1 for 1 entity querying, 2 for top 2. For more clarity, we obtain
the film as below. If you want to know how to deal with search result in a better way, you can refer to
`search result <https://milvus.io/cn/api-reference/pymilvus/v2.0.0rc5/results.html>`_ in PyMilvus doc.

>>> result = results[0]
>>> embedding_1 = result[0]
>>> embedding_2 = result[1]

Then how do we get ids, distances and fields? It's as below.

.. note::
   Because vectors are randomly generated, so the retrieved vector id and distance may differ.

>>> embedding_1.id  # id
1615279498011637002

>>> embedding_1.distance  # distance
1.0709768533706665


Drop a Partition
----------------

You can also drop a partition.

.. Danger::
   Once you drop a partition, all the data in this partition will be deleted too.

>>> collection.drop_partition("new_partition")


Drop a Collection
-----------------

Finally, you can drop an entire collection.

.. Danger::
   Once you drop a collection, all the data in this collection will be deleted too.

>>> collection.drop()


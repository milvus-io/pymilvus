========
Tutorial
========

This is a basic introduction to Milvus by PyMilvus.

For a runnable python script,
checkout `example.py <https://github.com/milvus-io/pymilvus/blob/1.1/examples/example.py>`_ on PyMilvus Github,
or `hello milvus <https://www.milvus.io/docs/example_code.md>`_ on Milvus official website. It's a good recommended
start to get started with Milvus and PyMilvus as well.


.. note::
   Here we use float vectors as example vector field data, if you want to learn example about binary vectors, see
   `binary vector example <https://github.com/milvus-io/pymilvus/blob/1.1/examples/example_binary.py>`_.


Prerequisites
=============

Before we start, there are some prerequisites.

Make sure that:

- You have a running Milvus instance.
- PyMilvus is correctly `installed <https://pymilvus.readthedocs.io/en/1.1/install.html>`_.

Connect to Milvus
=================

First of all, we need to import PyMilvus.

>>> from milvus import Milvus, DataType, MetricType

Then, we can make connection with Milvus server.
By default Milvus runs on localhost in port 19530, so you can use default value to connect to Milvus.

>>> host = '127.0.0.1'
>>> port = '19530'
>>> client = Milvus(host, port)

After connecting, we can communicate with Milvus in the following ways. If you are confused about the
terminology, see `Milvus Terminology <https://milvus.io/docs/terms.md>`_ for explanations.


Collection
==========

Now let's create a new collection. Before we start, we can list all the collections already exist. For a brand
new Milvus running instance, the result should be empty.

>>> client.list_collections()
(Status(code=0, message='Show collections successfully!'), [])

Create Collection
=================

To create collection, we need to provide collection parameters.
``collection_param`` consists of 4 components, they are ``collection_name``, ``dimension``, ``index_file_size``
and ``metric_type``.

collection_name:
    The name of collection should be a unique string to collections already exist.

dimension:
    For a float vector, dimension should be equal to the length of a vector; for a binary vector, dimension should
    be equal to bit size of a vector.

index_file_size:
    Milvus controls the size of data segment according to the `index_file_size`, you can refer to
    `Storage Concepts <https://milvus.io/docs/storage_concept.md>`_ for more information about segments and `index_file_size`.

metric_type:
    Milvus compute distance between two vectors, you can refer to `Distance Metrics <https://milvus.io/docs/metric.md>`_
    for more information.

Now we can create a collection:

>>> collection_name = 'demo_film_tutorial'
>>> collection_param = {
...     "collection_name": collection_name,
...     "dimension": 8,
...     "index_file_size": 2048,
...     "metric_type": MetricType.L2
... }
>>> client.create_collection(collection_param)
Status(code=0, message='Create collection successfully!')

Then you can list collections and 'demo_film_tutorial' will be in the result.

>>> client.list_collections()
(Status(code=0, message='Show collections successfully!'), ['demo_film_tutorial'])

You can also get info of the collection.

>>> status, info = client.get_collection_info(collection_name)
>>> info
CollectionSchema(collection_name='demo_film_tutorial', dimension=8, index_file_size=2048, metric_type=<MetricType: L2>)

The attributes of collection can be extracted from `info`.

>>> info.collection_name
'demo_film_tutorial'

>>> info.dimension
8

>>> info.index_file_size
2048

>>> info.metric_type
<MetricType: L2>


This tutorial is a basic intro tutorial, building index won't be covered by this tutorial.
If you want to go further into Milvus with indexes, it's recommended to check our
`index examples <https://github.com/milvus-io/pymilvus/tree/1.1/examples/indexes>`_.

If you're already known about indexes from ``index examples``, and you want a full lists of params supported
by PyMilvus, you check out `Index <https://pymilvus.readthedocs.io/en/1.1/param.html>`_
chapter of the PyMilvus documentation.

Further more, if you want to get a thorough view of indexes, check our official website for
`Vector Index <https://milvus.io/docs/index.md>`_.

Create Partition
================

If you don't create a partition, there will be a default one called "``_default``", all the entities will be
inserted into the "``_default``" partition. You can check it by ``list_partitions()``

>>> client.list_partitions(collection_name)
(Status(code=0, message='Success'), [(collection_name='demo_film_tutorial', tag='_default')])

You can provide a partition tag to create a new partition.

>>> client.create_partition(collection_name, "films")
Status(code=0, message='OK')
>>> client.list_partitions(collection_name)
(Status(code=0, message='Success'), [(collection_name='demo_film_tutorial', tag='_default'), (collection_name='demo_film_tutorial', tag='films')])

Entities
========

An entity is a group of fields that corresponds to real world objects. In current version, Milvus only contains a vector field.
Here is an example of 3 entities structured in list of list.

>>> import random
>>> entities = [[random.random() for _ in range(8)] for _ in range(3)]


Insert Entities
===============

>>>> status, ids = client.insert(collection_name, entities)

If the entities inserted successfully, ``ids`` we provided will be returned.

>>> ids
[1615279498011637000, 1615279498011637001, 1615279498011637002]

Or you can also provide entity ids

>>> entity_ids = [0, 1, 2]
>>> status, ids = client.insert(collection_name, entities, entity_ids)

.. warning::
   If the first time when `insert()` is invoked `ids` is not passed into this method, each of the rest time
   when `inset()` is invoked `ids` is not permitted to pass, otherwise server will return an error and the
   insertion process will fail. And vice versa.

.. note:
   If ``partition_tag`` isn't provided, these entities will be inserted into the "``_default``" partition,
   otherwise, them will be inserted into specified partition.



Flush
=====

After successfully inserting 3 entities into Milvus, we can ``Flush`` data from memory to disk so that we can
retrieve them. Milvus also performs an automatic flush with a fixed interval(configurable, default 1 second),
see `Data Flushing <https://milvus.io/docs/flush_python.md>`_.

You can flush multiple collections at one time, so be aware the parameter is a list.

>>> client.flush([collection_name])
Status(code=0, message='OK')

Get Detailed information
========================

After insertion, we can get the detail of collection statistics information by ``get_collection_stats()``

.. note::
   For a better output format, we are using ``pprint`` to provide a better format.

>>> from pprint import pprint
>>> status, stats = client.get_collection_stats(collection_name)
>>> pprint(stats)
{'partitions': [{'row_count': 3,
                 'segments': [{'data_size': 120,
                               'index_name': 'IDMAP',
                               'name': '1615279498038473000',
                               'row_count': 3}],
                 'tag': '_default'},
                {'row_count': 0, 'segments': None, 'tag': 'films'}],
 'row_count': 3}


Count Entities
==============

We can also count how many entities are there in the collection.

>>> client.count_entities(collection_name)
(Status(code=0, message='Success!'), 3)

Get
===

Get Entities by ID
------------------

You can get entities by their ids.

>>> status, films = client.get_entity_by_id(collection_name, [0, 1615279498011637001])
>>> films
[[], [0.8309633731842041, 0.7896093726158142, 0.09463301301002502, 0.7827594876289368, 0.5261889100074768, 0.8051634430885315, 0.18777835369110107, 0.28041353821754456]]

If id exists, an entity will be returned. If id doesn't exist, ``[]`` will be return. For the example above,
the result ``films`` will only have one entity, the other is ``[]``. Because vector id are generated by server, so the value of id may differ.


Search
======

Search Entities by Vector Similarity
------------------------------------

You can get entities by vector similarity. Assuming we have a ``film_A`` like below, and we want to get top 2 films
that are most similar with it.

>>> film_A = [random.random() for _ in range(8)]
>>> status, results = client.search(collection_name, 2, [film_A])

.. note::
    If the collection is index-built, user need to specify search param, and pass parameter `params` like: `client.search(..., params={...})`.
    You can refer to `Index params <https://pymilvus.readthedocs.io/en/1.1/param.html>`_ for more details.

.. note::
    If parameter `partition_tags` is specified, milvus executes search request on these partition instead of whole collection.

The returned ``results`` is a 2-D like structure, 1 for 1 entity querying, 2 for top 2. For more clarity, we obtain
the film as below. If you want to know how to deal with search result in a better way, you can refer to
`search result <https://pymilvus.readthedocs.io/en/1.1/results.html>`_ in PyMilvus doc.

>>> result = results[0]
>>> film_1 = result[0]
>>> film_2 = result[1]

Then how do we get ids, distances and fields? It's as below.

.. note::
   Because vectors are randomly generated, so the retrieved vector id and distance may differ.

>>> film_1.id  # id
1615279498011637002

>>> film_1.distance  # distance
1.0709768533706665


Deletion
========

Finally, let's move on to deletion in Milvus.
We can delete entities by ids, drop a whole partition, or drop the entire collection.

Delete Entities by id
---------------------

You can delete entities by their ids.

>>> client.delete_entity_by_id(collection_name, [0, 1615279498011637002])
Status(code=0, message='OK')

.. note::
    If one entity corresponding to a specified id doesn't exist, milvus ignore it and execute next deletion.
    In this case, client always return ok status except any exception occurs.

>>> client.count_entities(collection_name)
(Status(code=0, message='Success!'), 2)

Drop a Partition
----------------

You can also drop a partition.

.. Danger::
   Once you drop a partition, all the data in this partition will be deleted too.

>>> client.drop_partition(collection_name, "films")
Status(code=0, message='OK')


Drop a Collection
-----------------

Finally, you can drop an entire collection.

.. Danger::
   Once you drop a collection, all the data in this collection will be deleted too.

>>> client.drop_collection(collection_name)
Status(code=0, message='OK')

.. sectionauthor::
   `Yangxuan@milvus <https://github.com/XuanYang-cn>`_

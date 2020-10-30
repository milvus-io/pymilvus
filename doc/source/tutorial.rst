========
Tutorial
========

This is a basic introduction to Milvus by PyMilvus.

For a runnable python script,
checkout `example.py <https://github.com/milvus-io/pymilvus/blob/master/examples/example.py>`_ on PyMilvus Github,
or `hello milvus <https://milvus.io/docs/v0.11.0/example_code.md>`_ on milvus offical website. It's a good recommended start to get started with Milvus and PyMilvus as well.


Prerequisites
=============

Before we start, there are some prerequisites.

Make sure that:

- You have a running Milvus instance.
- PyMilvus is correctly `installed <https://pymilvus.readthedocs.io/en/latest/install.html>`_.

Connect to Milvus
=================

First of all, we need to import PyMilvus.

>>> from milvus import Milvus, DataType

Then, we can make connection with Milvus server.
By default Milvus runs on localhost in port 19530, so you can use default value to connect to Milvus.

>>> host = '127.0.0.1'
>>> port = '19530'
>>> client = Milvus(host, port)

After connecting, we can communicate with Milvus in the following ways. If you are confused about the
terminology, see `Milvus Terminology <https://milvus.io/docs/v0.11.0/terms.md>`_ for explanations.


Collection
==========

Now let's create a new collection. Before we start, we can list all the collections already exist. For a brand
new Milvus running instance, the result should be empty.

>>> client.list_collections()
[]

To create collection, we need to provide a unique collection name and some other parameters.
``collection_name`` should be a unique string to collections already exist. ``collection_param`` consists of 3 components, 
they are ``fields``, ``segment_row_limit`` and ``auto_id``.

>>> collection_name = 'demo_film_tutorial'
>>> collection_param = {
...     "fields": [
...         {
...             "name": "release_year",
...             "type": DataType.INT32
...         },
...         {
...             "name": "embedding",
...             "type": DataType.FLOAT_VECTOR,
...             "params": {"dim": 8}
...         },
...     ],
...     "segment_row_limit": 4096,
...     "auto_id": False
... }

collection_param
----------------

In the ``collection_param``, there are 2 fields in ``fields``: ``embedding`` and ``release_year``.

release_year:
    The name of the first field is ``release_year``, and the type of it is ``DataType.INT32``,
    it's a field to store release year of a film.

embedding:
    The name of the second field is ``embedding``, the type of it is ``DataType.FLOAT_VECTOR``.
    It also has an extra parameter "params" with dimension 8. It's a float vector field to store
    embedding of a film. For a FLOAT_VECTOR, the **"dim" in "params" is required**. You can also add
    ``params`` in other types of field as in FLOAT_VECTOR field.

segment_row_limit
-----------------

Milvus controls the size of data segment according to the ``segment_row_limit``, you can refer
`Storage Concepts <https://milvus.io/docs/v0.11.0/storage_concept.md>`_ for more information about segment and
``segment_row_limit``.
 
auto_id
-------
``auto_id`` is used to tell Milvus if we want ids auto-generated or ids user provided for each entity.
If ``False``, it means we'll have to provide our own ids for entities while inserting, on the contrary,
if ``True``, Milvus will generate ids automatically and we can't provide our own ids for entities.


Create Collection
=================

Now we can create a collection:

>>> client.create_collection(collection_name, collection_param)

Then you can list collections and 'demo_film_tutorial' will be in the result.

>>> client.list_collections()
['demo_film_tutorial']

You can also get info of the collection.

.. note::
   For a better output format, we use ``pprint`` to print the result.

>>> from pprint import pprint
>>> info = client.get_collection_info(collection_name)
>>> pprint(info)
{'auto_id': False,
 'fields': [{'indexes': [{}],
             'name': 'release_year',
             'params': {},
             'type': <DataType.INT32: 4>},
            {'indexes': [{}],
             'name': 'embedding',
             'params': {'dim': 8},
             'type': <DataType.FLOAT_VECTOR: 101>}],
 'segment_row_limit': 4096}

You can see from the output, all the infos are the same as we provide, but there's one more called ``indexes``.

This tutorial is a basic intro tutorial, building index won't be covered by this tutorial.
If you want to go further into Milvus with indexes, it's recommended to check our
`example_index.py <https://github.com/milvus-io/pymilvus/blob/master/examples/example_index.py>`_.

If you're already known about indexes from ``example_index.py``, and you want a full lists of params supported
by PyMilvus, you check out `Index <https://pymilvus.readthedocs.io/en/latest/param.html>`_
chapter of the PyMilvus documentation.

Further more, if you want to get a thorough view of indexes, check our official website for
`Vector Index <https://milvus.io/docs/v0.11.0/index.md>`_.

Create Partition
================

If you don't create a partition, there will be a default one called "``_default``", all the entities will be
inserted into the "``_default``" partition. You can check it by ``list_partitions()``

>>> client.list_partitions(collection_name)
['_default']

You can provide a partition tag to create a new partition.

>>> client.create_partition(collection_name, "American")
>>> client.list_partitions(collection_name)
['American', '_default']

Entities
========

An entities is a group of fields that correspond to real world objects. Here is an example of 3 entities
structured in list of dictionary.

>>> import random
>>> The_Lord_of_the_Rings = [
...     {
...         "id": 1,
...         "title": "The_Fellowship_of_the_Ring",
...         "release_year": 2001,
...         "embedding": [random.random() for _ in range(8)]
...     },
...     {
...         "id": 2,
...         "title": "The_Two_Towers",
...         "release_year": 2002,
...         "embedding": [random.random() for _ in range(8)]
...     },
...     {
...         "id": 3,
...         "title": "The_Return_of_the_King",
...         "release_year": 2003,
...         "embedding": [random.random() for _ in range(8)]
...     }
... ]


Insert Entities
===============

To insert entities into Milvus, we need to group data from the same field like below.

>>> ids = [k.get("id") for k in The_Lord_of_the_Rings]
>>> release_years = [k.get("release_year") for k in The_Lord_of_the_Rings]
>>> embeddings = [k.get("embedding") for k in The_Lord_of_the_Rings]

Then we can create hybrid entities to insert into Milvus.

>>> hybrid_entities = [
...     # Milvus doesn't support string type yet,
...     # so we cannot insert "title".
...     {
...         "name": "release_year",
...         "values": release_years,
...         "type": DataType.INT32
...     },
...     {
...         "name": "embedding",
...         "values": embeddings,
...         "type": DataType.FLOAT_VECTOR
...     },
... ]

If the hybrid entities inserted successfully, ``ids`` we provided will be returned.

.. note::
   If we create collection with ``auto_id = True``, we can't provide ids of our own, and the returned
   ``ids`` is automatically generated by Milvus. If ``partition_tag`` isn't provided, these entities will
   be inserted into the "``_default``" partition.

>>> client.insert(collection_name, hybrid_entities, ids, partition_tag="American")
[1, 2, 3]

Flush
=====

After successfully inserting 3 entities into Milvus, we can ``Flush`` data from memory to disk so that we can
retrieve them. Milvus also performs an automatic flush with a fixed interval(1 second),
see `Data Flushing <https://milvus.io/docs/v0.11.0/flush_python.md>`_.

You can flush multiple collections at one time, so be aware the parameter is a list.

>>> client.flush([collection_name])

Get Detailed information
========================

After insert, we can get the detail of collection statistics info by ``get_collection_stats()``

.. note::
   Again, we are using ``pprint`` to provide a better format.

>>> info = client.get_collection_stats(collection_name)
>>> pprint(info)
{'data_size': 18156,
 'partition_count': 2,
 'partitions': [{'data_size': 0,
                 'id': 13,
                 'row_count': 0,
                 'segment_count': 0,
                 'segments': None,
                 'tag': '_default'},
                {'data_size': 18156,
                 'id': 14,
                 'row_count': 3,
                 'segment_count': 1,
                 'segments': [{'data_size': 18156,
                               'files': [{'data_size': 4124,
                                          'field': '_id',
                                          'name': '_raw',
                                          'path': '/C_7/P_14/S_7/F_49'},
                                         {'data_size': 5724,
                                          'field': '_id',
                                          'name': '_blf',
                                          'path': '/C_7/P_14/S_7/F_53'},
                                         {'data_size': 4112,
                                          'field': 'release_year',
                                          'name': '_raw',
                                          'path': '/C_7/P_14/S_7/F_51'},
                                         {'data_size': 4196,
                                          'field': 'embedding',
                                          'name': '_raw',
                                          'path': '/C_7/P_14/S_7/F_50'}],
                               'id': 7,
                               'row_count': 3}],
                 'tag': 'American'}],
 'row_count': 3}


Count Entities
==============

We can also count how many entities are there in the collection.

>>> client.count_entities(collection_name)
3

Get
===

Get Entities by ID
------------------

You can get entities by their ids.

>>> films = client.get_entity_by_id(collection_name, ids=[1, 200])

If id exists, an entity will be returned. If id doesn't exist, ``None`` will be return. For the example above,
collection "``demo_film_tutorial``" has an entity(id = 1), but doesn't have an entity(id = 200), so the result
``films`` will only have one entity, the other is ``None``. You can get the entity fields like below.
Because embeddings are random generated, so the value of embedding may differ.

>>> for film in films:
...     if film is not None:
...         print(film.id, film.get("release_year"), film.get("embedding"))
... 
1 2001 [0.5146051645278931, 0.9257888197898865, 0.8659316301345825, 0.8082002401351929, 0.33681046962738037, 0.7135953307151794, 0.14593836665153503, 0.9224222302436829]

If you want to know all the fields names, you can get them by:

>>> for film in films:
...     if film is not None:
...         film.fields
... 
['release_year', 'embedding']

Search
======

Search Entities by Vector Similarity
------------------------------------

You can get entities by vector similarity. Assuming we have a ``film_A`` like below, and we want to get top 2 films
that are most similar with it.

>>> film_A = {
...     "title": "random_title",
...     "release_year": 2002,
...     "embedding": [random.random() for _ in range(8)]
... }

We need to prepare query DSL(Domain Specific Language) for this search, for more information about does and
don'ts for Query DSL , please refer to PyMilvus documentation
`Query DSL <https://pymilvus.readthedocs.io/en/latest/query.html>`_ chapter.

.. todo:
   change dsl structure.

>>> dsl = {
...     "bool": {
...         "must": [
...             {
...                 "vector": {
...                     "embedding": {
...                         "topk": 2,
...                         "query": [film_A.get("embedding")],
...                         "metric_type": "L2"
...                     }
...                 }
...             }
...         ]
...     }
... }

Then we can search by this dsl.

.. note::
   If we don't provide anything in "``fields``", there will only be ids and distances in the results.
   Only what we have provided in the "``fields``" can be obtained finally.

>>> results = client.search(collection_name, dsl, fields=["release_year"])

The returned ``results`` is a 1 * 2 structure, 1 for 1 entity querying, 2 for top 2. For more clarity, we obtain
the film as below. If you want to know how to deal with search result in a better way, you can refer to
`search result <https://pymilvus.readthedocs.io/en/latest/results.html>`_ in PyMilvus doc.

>>> entities = results[0]
>>> film_1 = entities[0]
>>> film_2 = entities[1]

Then how do we get ids, distances and fields? It's as below.

.. note::
   Because embeddings are randomly generated, so the retrieved film id, distance and field may differ.

>>> film_1.id  # id
3

>>> film_1.distance  # distance
0.3749755918979645


>>> film_1.entity.get("release_year")  # fields
2003

Search Entities filtered by fields.
-----------------------------------

Milvus can also search entities back by vector similarity combined with fields filtering. Again we will be
using query DSL, please refer to PyMilvus documentation
`Query DSL <https://pymilvus.readthedocs.io/en/latest/query.html>`_ for more information.

For the same ``film_A``, now we also want to search back top 2 most similar films, but with one more condition: 
the release year of films searched back should be the same as ``film_A``. Here is how we organize query DSL.

.. todo:
   change dsl structure.

>>> dsl_hybrid = {
...     "bool": {
...         "must": [
...             {
...                 "term": {"release_year": [film_A.get("release_year")]}
...             },
...             {
...                 "vector": {
...                     "embedding": {
...                         "topk": 2,
...                         "query": [film_A.get("embedding")],
...                         "metric_type": "L2"
...                     }
...                 }
...             }
...         ]
...     }
... }

Then we'll do search as above. This time we will only get 1 film back because there is only 1 film whose release
year is 2002, the same as ``film_A``. You can confirm it by ``len()``.

>>> results = client.search(collection_name, dsl_hybrid, fields=["release_year"])
>>> len(results[0])
1

We can also search back entities with fields in specific range like below, we want to get top 2 films
that are most similar with ``film_A``, and we want the ``release_year`` of entities returned must be larger than
``film_A``'s ``release_year``

.. todo:
   change dsl structure.

>>> dsl_hybrid = {
...     "bool": {
...         "must": [
...             {
...                 # "GT" for greater than
...                 "range": {"release_year": {"GT": film_A.get("release_year")}}
...             },
...             {
...                 "vector": {
...                     "embedding": {"topk": 2, "query": [film_A.get("embedding")], "metric_type": "L2"}
...                 }
...             }
...         ]
...     }
... }

This query will only get 1 film back too, because there is only 1 film whose release year is larger than 2002.

Again, for more information about query DSL, please refer to our documentation 
`Query DSL <https://pymilvus.readthedocs.io/en/latest/query.html>`_.

Deletion
========

Finally, let's move on to deletion in Milvus.
We can delete entities by ids, drop a whole partition, or drop the entire collection.

Delete Entities by id
---------------------

You can delete entities by their ids.

>>> client.delete_entity_by_id(collection_name, ids=[1, 2])
Status(code=0, message='OK')

>>> client.count_entities(collection_name)
1

Drop a Partition
----------------

You can also drop a partition.

.. Danger::
   Once you drop a partition, all the data in this partition will be deleted too.

>>> client.drop_partition(collection_name, "American")
Status(code=0, message='OK')

>>> client.count_entities(collection_name)
0

Drop a Collection
-----------------

Finally, you can drop an entire collection.

.. Danger::
   Once you drop a collection, all the data in this collection will be deleted too.

>>> client.drop_collection(collection_name)
Status(code=0, message='OK')

.. sectionauthor::
   `Yangxuan@milvus <https://github.com/XuanYang-cn>`_

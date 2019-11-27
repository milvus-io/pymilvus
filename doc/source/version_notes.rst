Version notes
=============

Guide
-------------
This page notes brief information about version

Contents
-------------

* **v0.6.0 (developing)**

  * support **with** key

    Now you can use **with** to connect and disconnect automatically. You need specify server host and ip or uri when instantiating a Milvus object. ::

      with Milvus(host=${HOST}, port=${PORT}) as client:
          client.create_table(...)

    or ::

      with Milvus(uri=${URI}) as client:
          client.create_table(...)

  * support search hook to customize search behavior

    Milvus class add a new method **set_hook** to set method hook, support customize behavior of client. Search hook is only supported currently. To do this, you must customize a class inherit from **BaseSearchHook**: ::

      class CustomizedSearchHook(BaseSerchHook):
             .......
             .......

      search_hook = CustomizedSearchHook()
      client.set_hook(search=search_hook)

    `Here <https://milvus-io.github.io/milvus-sdk-python/pythondoc/index.html>`_ is a simple example on how to use search hook.
    You can find more details about API `Here <https://www.milvus.io>`_.

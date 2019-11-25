Version notes
=============

Guide
-------------
This page notes brief information about version

Contents
-------------

* **v0.6.0 (developing)**

  * support **with** key

    Now you can use **with** to connect and disconnect automatically
    ::
      with Milvus(host=${HOST}, port=${PORT}) as client:
          client.create_table(...)

  * support search hook to customize search behavior

    Milvus class add a new method **set_hook**, support customize behavior of client. Search hook is only supported currently. To do this, you must customize a class inherit from **BaseSearchHook**:
    ::
      class CustomizedSearchHook(BaseSerchHook):
             .......
             .......

      search_hook = CustomizedSearchHook()
      client.set_hook(search=search_hook)



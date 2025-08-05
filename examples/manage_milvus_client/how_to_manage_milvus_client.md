# How to Manage MilvusClient in PyMilvus

This guide provides instructions on managing connections with the Milvus server using the `MilvusClient` of PyMilvus. It includes default behavior, advanced usage with aliases, and best practices.

A `MilvusClient` holds an alias to a Milvus server connection. This alias represents a connection to a server or a specific database within the server. Let's first take a look at the default behavior.

## Default Behavior

### MilvusClient Shares Connections

Multiple `MilvusClient` objects with the same Milvus **uri**, **authentication**, and **database** reuse the same connection to the Milvus server.

The following code snippet demonstrate the reusing of connections in a single thread:
```python
TEST_DB = "test_DB"
URI = "http://localhost:19530"

c = MilvusClient(uri=URI)

# Multiple MilvusClient objects reuse the same connection to Milvus server
c_shared = []
for i in range(10):
    tmp = MilvusClient(uri=URI)
    c_shared.append(tmp)
    print(f"alias for {i}th MilvusClient: {tmp._using}, results of list_collections: {tmp.list_collections()}")
```

If you close one of the `MilvusClient` objects, the others won't work anymore.
```python
c.close() # close one of the MilvusClient objects
for tmp in c_shared:
    try:
        tmp.list_collections()
    except Exception as ex:
        print("MilvusClient sharing the same connection will be unable to use, exception: %s", ex)
```

The following code snippet demonstrate the reusing of connections in multiple threads:
```python
def multi_thread_init_milvus_client():
    """Multiple MilvusClient objects in multiple threads share the same connection to Milvus server"""
    threads = []
    thread_count = 10
    for k in range(thread_count):
        x = threading.Thread(target=worker, args=(MilvusClient(uri=URI),))
        threads.append(x)
        x.start()

    for th in threads:
        th.join()


def multi_thread_copy_milvus_client():
    """MilvusClient objects are safe to copy across threads"""
    c_main = MilvusClient(uri=URI)

    threads = []
    thread_count = 10
    for k in range(thread_count):
        x = threading.Thread(target=worker, args=(c_main,))
        threads.append(x)
        x.start()

    for th in threads:
        th.join()


def worker(c: MilvusClient):
    got = c.list_collections()
    print(f"Worker, alias to the server connection: {c._using}, results of list_collections: {got}")
```

Whether copy or not, they all share the same connection to Milvus server underneath.

### MilvusClient Doesn't Share Connections

**`MilvusClient` objects with different uri, authentication, and database don't share connections to the Milvus server by default.**

The following code snippet demonstrate the *c_testdb* and *c* don't share the same connection to Milvus server:
```python
c = MilvusClient(uri=URI)
c.create_database(TEST_DB)

c_testdb = MilvusClient(uri=URI, db_name=TEST_DB)

# c and c_testdb don't share the same connection, the same as different authentications.
print(f"alias for c:        {c._using}, results of c.list_collections: {c.list_collections()}")
print(f"alias for c_testdb: {c_testdb._using}, results of c_testdb.list_collections: {c_testdb.list_collections()}")

c_testdb.close()
try:
    c_testdb.list_collections()
except Exception as ex:
    print(f"c_testdb has been closed, exception: {ex}")

# close of c_testdb won't affect c
print(f"results of c.list_collections: {c.list_collections()}")
```

## Advanced usage: customized aliases

If single connection doesn't meet your performance needs, you can create multiple connections with customized aliases.
The following code snippet demonstrate how to create unique connectionswith "c1-alias" and "c2-alias":

```python
def advanced_unique_connections():
  c1 = MilvusClient(uri=URI, alias="c1-alias")
  c2 = MilvusClient(uri=URI, alias="c2-alias")
```
Notes:

- **Avoid Conflict**: Manage alias names carefully to avoid conflicts in connection management.
- **Resource Management**: Ensure to close MilvusClient of customized aliases when no longer needed to free resources.

### Best Practices

- If you use the default bahavior, that different MilvusClient might share the same connection to Milvus server. Always ensure to **never close the MilvusClient** to avoid influencing shared MilvusClient.
- If you use customized aliases, be aware to manage them carefully.
    - Ensure MilvusClient are closed when no longer needed to free resources.
    - Ensure no other clients require the connection before closing the MilvusClient.
    - Aviod using short-lived connections, reuse them as much as possible.

By following these guidelines, you can effectively manage Milvus client connections for various application scenarios.

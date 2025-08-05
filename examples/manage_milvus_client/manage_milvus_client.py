""" Before running this example, you need to start Milvus server first. """

import logging
import threading

from pymilvus import MilvusClient

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s (%(lineno)s) (%(threadName)s)')

URI = "http://localhost:19530"
TEST_DB = "test_db"

def worker(c: MilvusClient):
    LOGGER.info(f"Worker, alias to the server connection: {c._using}, results of list_collections: {c.list_collections()}")

def multi_thread_init_milvus_client():
    """Multiple MilvusClient objects in multiple threads share the same connection to Milvus server"""
    LOGGER.info("Multiple MilvusClient objects in multiple threads share the same connection to Milvus server")
    threads = []
    thread_count = 10
    for k in range(thread_count):
        x = threading.Thread(target=worker, args=(MilvusClient(uri=URI),))
        threads.append(x)
        x.start()
        LOGGER.debug(f"Thread-{k} '{x.name}' started")

    for th in threads:
        th.join()
        LOGGER.debug(f"Thread '{th.name}' finished")


def multi_thread_copy_milvus_client():
    """MilvusClient objects are safe to copy across threads"""
    LOGGER.info("Multiple MilvusClient objects are safe to copy across threads, they all shared the same connection to Milvus server")
    c_main = MilvusClient(uri=URI)

    threads = []
    thread_count = 10
    for k in range(thread_count):
        x = threading.Thread(target=worker, args=(c_main,))
        threads.append(x)
        x.start()
        LOGGER.debug(f"Thread-{k} '{x.name}' started")

    for th in threads:
        th.join()
        LOGGER.debug(f"Thread '{th.name}' finished")


def shared_connections():
    # A MilvusClient holds an alias to a Milvus server connection
    c = MilvusClient(uri=URI)
    LOGGER.info(f"Alias to the server connection: {c._using}, results of list_collections: {c.list_collections()}")
    if TEST_DB not in c.list_databases():
        c.create_database(TEST_DB)

    # Multiple MilvusClient objects reuse the same connection to Milvus server
    LOGGER.info("By default, multiple MilvusClient objects reuse the same connection to Milvus server")
    c_shared = []
    for i in range(10):
        tmp = MilvusClient(uri=URI)
        c_shared.append(tmp)
        LOGGER.info(f"  {i}th MilvusClient, alias to the server connection: {tmp._using}, results of list_collections: {tmp.list_collections()}")

    # MilvusClient to differen database or with different user and token
    # will create a new connection to Milvus server.
    LOGGER.info("=============================")
    c_testdb = MilvusClient(uri=URI, db_name=TEST_DB)
    LOGGER.info(f"MilvusClient to test_db, alias to the server connection: {c_testdb._using}, results of list_collections: {c_testdb.list_collections()}")

    # MilvusClient object is safe to copy across threads
    LOGGER.info("=============================")
    multi_thread_copy_milvus_client()

    # MilvusClient in multiple threads share the same connection to Milvus server
    LOGGER.info("=============================")
    multi_thread_init_milvus_client()

    # Never close a client if you're absolutely sure no one else is using it.
    # Close of one MilvusClient closes all other MilvusClient's access to MilvusServer
    LOGGER.info("=============================")
    c.close()
    LOGGER.info("Closed MilvusClient c, all MilvusClient sharing the same connection will be unable to use")
    for tmp in c_shared:
        try:
            tmp.list_collections()
        except Exception as ex:
            LOGGER.warning("    MilvusClient sharing the same connection will be unable to use, exception: %s", ex)


def advanced_unique_connections():
    # Use different alias to create MilvusClient doesn't share the connection to Milvus server
    LOGGER.info("=============================")
    LOGGER.info("Use different alias to create MilvusClient doesn't share the connection to Milvus server")
    c1 = MilvusClient(uri=URI, alias="c1-alias")
    LOGGER.info(f"Alias of c1 to the server connection: {c1._using}, results of c1.list_collections: {c1.list_collections()}")
    c2 = MilvusClient(uri=URI, alias="c2-alias")
    LOGGER.info(f"Alias of c2 to the server connection: {c2._using}, results of c2.list_collections: {c2.list_collections()}")

    c1.close()
    LOGGER.info("Closed MilvusClient c1, c1 cannot use")
    try:
        c1.list_collections()
    except Exception as ex:
        LOGGER.warning("Closed MilvusClient c1, it cannot use, exception: %s", ex)
    LOGGER.info(f"Closed MilvusClient c1, c2 can still use MilvusServer, results of c2.list_collections: {c2.list_collections()}")

    LOGGER.info("Be sure to manage your own alias's MilvusClient, close it when you're done")
    c2.close()


if __name__ == "__main__":
    shared_connections()
    advanced_unique_connections()

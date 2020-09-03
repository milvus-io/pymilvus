from milvus import *
from multiprocessing import Manager, Process
from time import sleep


MILVUS_SERVER_IP = "127.0.0.1"
MILVUS_SERVER_PORT = "19530"


class MilvusClient:
    def __init__(self, serverIP, serverPort):
        print("start connect")
        self._milvusConn = self.connectMilvus(serverIP, serverPort)
        print("over connect")

    @staticmethod
    def connectMilvus(serverIP, serverPort):
        __conn__  = Milvus(serverIP, serverPort)

        if not __conn__:
            print("failed to connect to milvus server with ip {0} and port {1}".format(serverIP, serverPort))
            exit(-1)

        return __conn__


def worker(workerID):
    print("workerID {0} start".format(workerID))
    milvusClient = MilvusClient(MILVUS_SERVER_IP, MILVUS_SERVER_PORT)
    while True:
        sleep(1)

    print("workerID {0} exit".format(workerID))


def runWorkers(workerNum = 4):
    for processIndex in range(workerNum):
        _process = Process(target=worker, args=(processIndex,))
        _process.start()


if __name__ == '__main__':
    # runWorkers()
    client = Milvus(MILVUS_SERVER_IP, MILVUS_SERVER_PORT)
    r = client.get_config("wal", "recovery_error_ignore")
    print(r)
    client.set_config("cache", "cache_size", "4GB")

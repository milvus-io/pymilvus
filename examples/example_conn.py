import sys
import itertools
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(".")

from milvus import Milvus

if __name__ == '__main__':
    stub = Milvus()
    stub.connect()

    i = 0
    for _ in range(100):
        for _h, _p in itertools.product(
                ["123.0.0.1", "192.115.112.7", "123.0.45.6", "127.0.0.1"],
                ["678", "999", "19599"]):
            try:
                logger.info("Try connect: round {}".format(i))
                client = Milvus()
                client.connect(host=_h, port=_p)
                i += 1
                # client.connect(host="127.0.0.1", port="19530")
                # break
            except Exception:
                # pass
                # del client
                continue

            logger.info("Connect successfully exceptedly on round {} | host = {}, port = {}".format(i, _h, _p))
            sys.exit(1)

    logger.info("Done.")

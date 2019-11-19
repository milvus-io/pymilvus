import sys

sys.path.append(".")

from milvus import Milvus

if __name__ == '__main__':
    # client = Milvus()

    for i in range(1000):
        try:
            print("Try connect: round {}".format(i))
            client = Milvus()
            client.connect(host="123.0.0.1", port="19530")
            # client.connect(host="127.0.0.1", port="19530")
            # break
        except Exception:
            # pass
            # del client
            continue

        print("Connect successfully exceptedly on round {}".format(i))
        sys.exit(1)

    print("Done.")

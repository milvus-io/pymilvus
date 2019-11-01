import sys

from .. import Milvus, IndexType, MetricType


def _create_table(_args):
    _dict = _args.__dict__
    _ip = _dict.get("host", None)
    if not _ip:
        print("Please input host address")
        sys.exit(1)
    print("Host {}".format(_ip))

    _port = _dict.get("port", None)
    if not _port:
        print("Please input port")
        sys.exit(1)
    print("Port {}".format(_port))

    _name = _dict.get("name", None)
    if not _name:
        print("Please input port")
        sys.exit(1)

    client = Milvus()
    status = client.connect(host=str(_ip), port=str(_port))
    if not status.OK():
        print(status.message)
        exit(1)

    print(status.message)

    param = {
        'table_name': _name,
        'dimension': 16,
        'index_file_size': 1024,
        'metric_type': MetricType.L2
    }

    status = client.create_table(param)
    print(status)

    client.disconnect()


def table(args):
    if args.table == 'create':
        _create_table(args)


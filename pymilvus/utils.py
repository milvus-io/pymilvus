from . import DataType


def entities_slice(entities):
    row_length = 0
    row_count = len(entities[0]["values"])
    for field in entities:
        ftype = field["type"]
        if ftype in (DataType.BOOL,):
            row_length += 1
        elif ftype in (DataType.INT32, DataType.FLOAT):
            row_length += 4
        elif ftype in (DataType.INT64, DataType.DOUBLE):
            row_length += 8
        elif ftype in (DataType.FLOAT_VECTOR,):
            row_length += 4 * len(field["values"][0])
        elif ftype in (DataType.BINARY_VECTOR,):
            row_length += len(field["values"][0])

    max_slice_row_count = 256 << 20 // row_length

    for i in range(0, row_count, max_slice_row_count):
        end = min(i + max_slice_row_count, row_count)
        slice_entities = list()
        for f in entities:
            slice_entities.append({"name": f["name"], "values": f["values"][i: end], "type": f["type"]})
        yield slice_entities

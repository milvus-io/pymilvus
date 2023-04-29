"""Default MilvusClient args."""

DEFAULT_SEARCH_PARAMS = {
    "IVF_FLAT": {"metric_type": "L2", "params": {"nprobe": 10}},
    "IVF_SQ8": {"metric_type": "L2", "params": {"nprobe": 10}},
    "IVF_PQ": {"metric_type": "L2", "params": {"nprobe": 10}},
    "HNSW": {"metric_type": "L2", "params": {"ef": 10}},
    "RHNSW_FLAT": {"metric_type": "L2", "params": {"ef": 10}},
    "RHNSW_SQ": {"metric_type": "L2", "params": {"ef": 10}},
    "RHNSW_PQ": {"metric_type": "L2", "params": {"ef": 10}},
    "AUTOINDEX": {"metric_type": "L2", "params": {}},
}

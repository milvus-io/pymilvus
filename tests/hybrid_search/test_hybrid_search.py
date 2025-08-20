#herer
#MILVUS_HOST = "10.104.17.119"
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530

import numpy as np
from pymilvus import (
    connections,
    utility,
    Collection,
    AnnSearchRequest, RRFRanker, WeightedRanker,
)
from ml_dtypes import bfloat16

# params:
#       nq: 10
#       top_k: 10
#       timeout: 300
#       random_data: true
#       output_fields: [id, binary_vector, float16_vector, bfloat16_vector, float_vector]
#       check_task: check_search_output
#       check_items:
#         output_fields: [id, binary_vector, float16_vector, bfloat16_vector, float_vector]
#       rerank:
#         WeightedRanker: [0.4, 0.3, 0.3]
#       reqs:
#         - anns_field: binary_vector
#           search_param:
#             nprobe: 32
#           top_k: 10
#           expr: "int32_1 > 3000000"
#         - anns_field: float16_vector
#           search_param:
#             nprobe: 32
#           top_k: 5
#           expr: "float32_1 < 5000000.1"
#         - anns_field: bfloat16_vector
#           search_param:
#             nprobe: 32
#           top_k: 12

def convert_bool_list_to_bytes(bool_list):
    if len(bool_list) % 8 != 0:
        raise ValueError("The length of a boolean list must be a multiple of 8")

    byte_array = bytearray(len(bool_list) // 8)
    for i, bit in enumerate(bool_list):
        if bit == 1:
            index = i // 8
            shift = i % 8
            byte_array[index] |= (1 << shift)
    return bytes(byte_array)

def test_hybrid_search_weighted_ranker():
    """测试使用WeightedRanker的hybrid search"""
    print("=== 测试WeightedRanker的hybrid search ===")
    
    # 连接Milvus
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    
    # 获取collection
    collection_name = "fouram_uRAi5NhA"
    if not utility.has_collection(collection_name):
        print(f"Collection {collection_name} 不存在")
        return
    
    collection = Collection(collection_name)
    
    # 参数配置
    nq = 10
    top_k = 10
    timeout = 300
    output_fields = ["id", "binary_vector", "float16_vector", "bfloat16_vector", "float_vector"]
    
    # 生成随机搜索向量
    rng = np.random.default_rng(seed=19530)
    
    # 循环进行1000次hybrid search
    total_tests = 10000
    success_count = 0
    fail_count = 0
    
    print(f"开始进行 {total_tests} 次hybrid search测试...")
    # 构建搜索请求列表
    req_list = []
    
    # 第一个请求：binary_vector
    # 128维binary vector = 16字节，每个bit表示一个维度
    binary_vectors = []
    for _ in range(nq):
        # 生成128个随机bit
        bool_list = (rng.random(128) > 0.5).astype(int).tolist()
        # 转换为字节
        byte_data = convert_bool_list_to_bytes(bool_list)
        binary_vectors.append(byte_data)
    
    search_param = {
        "data": binary_vectors,
        "anns_field": "binary_vector",
        "param": {"nprobe": 32, "metric_type": "JACCARD"},  # 二进制向量使用HAMMING距离
        "limit": 10,
        "expr": "int32_1 > 3000000000"
    }
    req = AnnSearchRequest(**search_param)
    req_list.append(req)
    
    # 第二个请求：float16_vector
    float16_vectors = rng.random((nq, 128)).astype(np.float16)  # 使用float16类型
    search_param = {
        "data": float16_vectors,
        "anns_field": "float16_vector",
        "param": {"nprobe": 32, "metric_type": "COSINE"},
        "limit": 5,
        "expr": "float32_1 < 5000000.1"
    }
    req = AnnSearchRequest(**search_param)
    req_list.append(req)
    
    # 第三个请求：bfloat16_vector
    bfloat16_vectors = []
    for _ in range(nq):
        raw_vector = [rng.random() for _ in range(128)]
        bf16_vector = np.array(raw_vector, dtype=bfloat16)
        bfloat16_vectors.append(bf16_vector)
    
    search_param = {
        "data": bfloat16_vectors,
        "anns_field": "bfloat16_vector",
        "param": {"nprobe": 32, "metric_type": "IP"},
        "limit": 12
    }
    req = AnnSearchRequest(**search_param)
    req_list.append(req)
    
    # 使用WeightedRanker进行hybrid search
    weights = [0.4, 0.3, 0.3]
    ranker = WeightedRanker(*weights, norm_score=True)
    for test_round in range(total_tests):
        try:
            # 执行hybrid search
            hybrid_res = collection.hybrid_search(
                req_list, 
                ranker, 
                top_k, 
                output_fields=output_fields,
                timeout=timeout
            )
            
            # 检查每个结果的output_fields集合
            _result_check = True
            for o in [set(hit.fields.keys()) for hits in hybrid_res for hit in hits]:
                if o != set(output_fields):
                    _result_check = False
                    break
            
            if _result_check:
                success_count += 1
            else:
                fail_count += 1
                print(f"第 {test_round + 1} 次测试: output_fields不匹配")
                
        except Exception as e:
            fail_count += 1
            print(f"第 {test_round + 1} 次测试异常: {e}")
        
        # 每100次测试打印一次进度
        if (test_round + 1) % 100 == 0:
            print(f"已完成 {test_round + 1} 次测试, 成功: {success_count}, 失败: {fail_count}")
    
    # 打印最终结果
    print(f"\n=== 测试完成 ===")
    print(f"总测试次数: {total_tests}")
    print(f"成功次数: {success_count}")
    print(f"失败次数: {fail_count}")
    print(f"成功率: {success_count/total_tests*100:.2f}%")
    
    connections.disconnect("default")
    print("WeightedRanker测试完成\n")


if __name__ == "__main__":
    # 运行所有测试
    test_hybrid_search_weighted_ranker()
    print("所有测试完成！")


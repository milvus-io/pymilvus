from ..grpc_gen import status_pb2
from ..grpc_gen.milvus_pb2 import TopKQueryResult as Grpc_Result
from ..client.abstract import TopKQueryResult
from ..client.exceptions import ParamError


def merge_results(results_list, topk, *args, **kwargs):
    """
    merge query results
    """

    def _reduce(source_ids, ids, source_diss, diss, k, reverse):
        """

        """
        if source_diss[k - 1] <= diss[0]:
            return source_ids, source_diss
        if diss[k - 1] <= source_diss[0]:
            return ids, diss

        source_diss.extend(diss)
        diss_t = enumerate(source_diss)
        diss_m_rst = sorted(diss_t, key=lambda x: x[1], reverse=reverse)[:k]
        diss_m_out = [id_ for _, id_ in diss_m_rst]

        source_ids.extend(ids)
        id_m_out = [source_ids[i] for i, _ in diss_m_rst]

        return id_m_out, diss_m_out

    status = status_pb2.Status(error_code=status_pb2.SUCCESS,
                               reason="Success")

    reverse = kwargs.get('reverse', False)
    raw = kwargs.get('raw', False)

    if not results_list:
        return status, [], []

    merge_id_results = []
    merge_dis_results = []
    row_num = 0

    for files_collection in results_list:
        if not isinstance(files_collection, Grpc_Result) and \
                not isinstance(files_collection, TopKQueryResult):
            return ParamError("Result type is unknown.")

        row_num = files_collection.row_num
        if not row_num:
            continue

        ids = files_collection.ids
        diss = files_collection.distances  # distance collections
        # Notice: batch_len is equal to topk, may need to compare with topk
        batch_len = len(ids) // row_num

        for row_index in range(row_num):
            id_batch = ids[row_index * batch_len: (row_index + 1) * batch_len]
            dis_batch = diss[row_index * batch_len: (row_index + 1) * batch_len]

            if len(merge_id_results) < row_index:
                raise ValueError("merge error")
            if len(merge_id_results) == row_index:
                merge_id_results.append(id_batch)
                merge_dis_results.append(dis_batch)
            else:
                merge_id_results[row_index], merge_dis_results[row_index] = \
                    _reduce(merge_id_results[row_index], id_batch,
                            merge_dis_results[row_index], dis_batch,
                            batch_len,
                            reverse)

    id_mrege_list = []
    dis_mrege_list = []

    for id_results, dis_results in zip(merge_id_results, merge_dis_results):
        id_mrege_list.extend(id_results)
        dis_mrege_list.extend(dis_results)

    raw_result = Grpc_Result(
        status=status,
        row_num=row_num,
        ids=id_mrege_list,
        distances=dis_mrege_list
    )

    return raw_result if raw else TopKQueryResult(raw_result)

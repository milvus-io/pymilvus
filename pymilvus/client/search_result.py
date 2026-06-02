import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import orjson

from pymilvus.client.types import DataType
from pymilvus.exceptions import MilvusException
from pymilvus.grpc_gen import common_pb2, schema_pb2

from . import entity_helper, field_data_extractors
from .type_info import (
    get_array_element_attr,
    is_byte_vector_type,
    is_dense_vector_type,
    is_scalar_type,
    is_sparse_vector_type,
    row_width,
)

logger = logging.getLogger(__name__)


def _is_eager_scalar(dtype: DataType) -> bool:
    return is_scalar_type(dtype) and dtype not in {DataType.JSON, DataType.ARRAY}


def _is_lazy_result_field(dtype: DataType) -> bool:
    return (
        is_dense_vector_type(dtype)
        or is_sparse_vector_type(dtype)
        or dtype in {DataType.JSON, DataType._ARRAY_OF_STRUCT, DataType._ARRAY_OF_VECTOR}
    )


def _dense_result_slice_width(dtype: DataType, dim: int) -> Optional[int]:
    if not is_dense_vector_type(dtype):
        return None
    if not is_byte_vector_type(dtype):
        return dim
    return row_width(dtype, dim)


class HybridHits(list):
    ids: List[Union[str, int]]
    distances: List[float]
    lazy_field_data: List[schema_pb2.FieldData]
    has_materialized: bool
    dynamic_fields: List[str]
    start: int

    def __init__(
        self,
        start: int,
        end: int,
        all_pks: List[Union[str, int]],
        all_scores: List[float],
        fields_data: List[schema_pb2.FieldData],
        output_fields: List[str],
        highlight_results: List[common_pb2.HighlightResult],
        pk_name: str,
    ):
        self.ids = all_pks[start:end]
        self.distances = all_scores[start:end]
        self.dynamic_fields = set(output_fields) - {
            field_data.field_name for field_data in fields_data
        }
        self.lazy_field_data = []
        self.has_materialized = False
        self.start = start
        self._prefix_sum_cache = {}

        col_results = {}

        for field_data in fields_data:
            field_name = field_data.field_name

            if _is_eager_scalar(field_data.type) or field_data.type == DataType.ARRAY:
                col_results[field_name] = field_data_extractors.decode_range(field_data, start, end)

            elif _is_lazy_result_field(field_data.type):
                self.lazy_field_data.append(field_data)
            else:
                msg = f"Unsupported field type: {field_data.type}"
                raise MilvusException(msg)

        count = end - start
        entities = [{} for _ in range(count)]

        for field_name, values in col_results.items():
            for i, value in enumerate(values):
                entities[i][field_name] = value

        top_k_res = [
            Hit(
                {pk_name: self.ids[i], "distance": self.distances[i], "entity": entities[i]},
                pk_name=pk_name,
            )
            for i in range(count)
        ]

        if len(highlight_results) > 0:
            for i, hit in enumerate(top_k_res):
                hit["highlight"] = {
                    result.field_name: {
                        "fragments": list(result.datas[i + start].fragments),
                        "scores": list(result.datas[i + start].scores),
                    }
                    for result in highlight_results
                }

        super().__init__(top_k_res)

    def __str__(self) -> str:
        """Only print at most 10 query results"""
        reminder = f" ... and {len(self) - 10} entities remaining" if len(self) > 10 else ""
        return f"{self[:10]}{reminder}"

    def __getitem__(self, key: int):
        self.materialize()
        return super().__getitem__(key)

    def get_raw_item(self, idx: int):
        """Get the item at index without triggering materialization"""
        return list.__getitem__(self, idx)

    def __iter__(self):
        self.materialize()
        return super().__iter__()

    def _get_physical_index(self, field_data: Any, logical_index: int) -> int:
        """Calculate physical index for nullable vectors with sparse storage.

        Uses prefix sum for O(1) lookup instead of O(n) iteration.
        Caches prefix sum in instance variable using field_data id as key.
        """
        field_id = id(field_data)
        if field_id not in self._prefix_sum_cache:
            if len(field_data.valid_data) == 0:
                self._prefix_sum_cache[field_id] = None
            else:
                self._prefix_sum_cache[field_id] = np.cumsum(
                    [0] + [1 if v else 0 for v in field_data.valid_data]
                )
        prefix_sum = self._prefix_sum_cache[field_id]
        if prefix_sum is None:
            return logical_index
        return int(prefix_sum[logical_index])

    def materialize(self):
        if not self.has_materialized:
            n = len(self)
            for field_data in self.lazy_field_data:
                field_name = field_data.field_name

                if is_dense_vector_type(field_data.type) or field_data.type in (
                    DataType.SPARSE_FLOAT_VECTOR,
                    DataType._ARRAY_OF_VECTOR,
                ):
                    for i in range(n):
                        item = self.get_raw_item(i)
                        actual_idx = self.start + i
                        item["entity"][field_name] = field_data_extractors.decode_cell(
                            field_data,
                            actual_idx,
                            physical_index_override=self._get_physical_index(
                                field_data, actual_idx
                            ),
                        )
                elif field_data.type == DataType.JSON:
                    idx = self.start
                    for i in range(n):
                        item = self.get_raw_item(i)
                        json_dict_list = field_data_extractors.decode_cell(field_data, idx)
                        if json_dict_list is None:
                            item["entity"][field_name] = None
                        elif not field_data.is_dynamic:
                            item["entity"][field_data.field_name] = json_dict_list
                        elif not self.dynamic_fields:
                            item["entity"].update(json_dict_list)
                        else:
                            item["entity"].update(
                                {
                                    k: v
                                    for k, v in json_dict_list.items()
                                    if k in self.dynamic_fields
                                }
                            )
                        idx += 1
                elif field_data.type == DataType._ARRAY_OF_STRUCT:
                    # Process struct arrays - convert column format back to array of structs
                    idx = self.start
                    struct_arrays = field_data_extractors.get_field_data(field_data)
                    if struct_arrays and hasattr(struct_arrays, "fields"):
                        for i in range(n):
                            item = self.get_raw_item(i)
                            item["entity"][field_name] = (
                                entity_helper.extract_struct_array_from_column_data(
                                    struct_arrays, idx
                                )
                            )
                            idx += 1
                    else:
                        for i in range(n):
                            item = self.get_raw_item(i)
                            item["entity"][field_name] = None
                else:
                    msg = f"Unsupported field type: {field_data.type}"
                    raise MilvusException(msg)

        self.has_materialized = True

    __repr__ = __str__


class SearchResult(list):
    """A list[list[dict]] Contains nq * limit results.

    The first level is the results for each nq, and the second level
    is the top-k(limit) results for each query.

    Examples:
        >>> nq_res = client.search()
        >>> for topk_res in nq_res:
        >>>     for one_res in topk_res:
        >>>         print(one_res)
        {"id": 1, "distance": 0.1, "entity": {"vector": [1.0, 2.0, 3.0], "name": "a"}}
        ...

        >>> res[0][0]
        {"id": 1, "distance": 0.1, "entity": {"vector": [1.0, 2.0, 3.0], "name": "a"}}

        >>> res.recalls
        [0.9, 0.9, 0.9]

        >>> res.extra
        {"cost": 1}

    Attributes:
        recalls(List[float], optional): The recalls of the search result, one for each query.
        extra(Dict, optional): The extra information of the search result.
    """

    def __init__(
        self,
        res: schema_pb2.SearchResultData,
        round_decimal: Optional[int] = None,
        status: Optional[common_pb2.Status] = None,
        session_ts: Optional[int] = 0,
    ):
        _data = self._parse_search_result_data(res, round_decimal)
        super().__init__(_data)

        # set recalls
        self.recalls = res.recalls if len(res.recalls) > 0 else None

        # set extra info
        self.extra = {}
        if status and status.extra_info:
            if "report_value" in status.extra_info:
                self.extra["cost"] = int(status.extra_info["report_value"])
            if "scanned_remote_bytes" in status.extra_info:
                self.extra["scanned_remote_bytes"] = int(status.extra_info["scanned_remote_bytes"])
            if "scanned_total_bytes" in status.extra_info:
                self.extra["scanned_total_bytes"] = int(status.extra_info["scanned_total_bytes"])
            if "cache_hit_ratio" in status.extra_info:
                self.extra["cache_hit_ratio"] = float(status.extra_info["cache_hit_ratio"])

        # iterator related
        self._session_ts = session_ts
        self._search_iterator_v2_results = res.search_iterator_v2_results

    def __str__(self) -> str:
        """Only print at most 10 results"""
        result_msg = f"data: {self[:10]}"

        # Optional messages
        recall_msg = f",recalls: {self.recalls[:10]}" if self.recalls else ""
        extra_msg = f",{self.extra}" if self.extra else ""
        reminder = f" ... and {len(self) - 10} results remaining" if len(self) > 10 else ""

        return f"{result_msg}{recall_msg}{reminder}{extra_msg}"

    __repr__ = __str__

    def materialize(self):
        for i in range(len(self)):
            self[i].materialize()

    def _parse_search_result_data(
        self,
        res: schema_pb2.SearchResultData,
        round_decimal: Optional[int],
    ) -> List[List[Dict[str, Any]]]:
        _pk_name = res.primary_field_name or "id"

        all_pks: List[Union[str, int]] = []
        all_scores: List[float] = []
        if res.ids.HasField("int_id"):
            all_pks = res.ids.int_id.data
        elif res.ids.HasField("str_id"):
            all_pks = res.ids.str_id.data

        if isinstance(round_decimal, int) and round_decimal > 0:
            all_scores = [round(x, round_decimal) for x in res.scores]
        else:
            all_scores = res.scores

        data = []
        nq_thres = 0

        for topk in res.topks:
            start, end = nq_thres, nq_thres + topk
            data.append(
                HybridHits(
                    start,
                    end,
                    all_pks,
                    all_scores,
                    res.fields_data,
                    res.output_fields,
                    res.highlight_results,
                    _pk_name,
                )
            )

            nq_thres += topk
        return data

    def _get_fields_by_range(
        self, start: int, end: int, all_fields_data: List[schema_pb2.FieldData]
    ) -> Dict[str, Tuple[List[Any], schema_pb2.FieldData]]:
        field2data: Dict[str, Tuple[List[Any], schema_pb2.FieldData]] = {}

        for field in all_fields_data:
            name, dtype = field.field_name, field.type
            field_meta = schema_pb2.FieldData(
                type=dtype,
                field_name=name,
                field_id=field.field_id,
                is_dynamic=field.is_dynamic,
            )
            if _is_eager_scalar(dtype):
                field2data[name] = field_data_extractors.decode_range(field, start, end), field_meta
                continue

            if dtype == DataType.JSON:
                field2data[name] = field_data_extractors.decode_range(field, start, end), field_meta
                continue

            if dtype == DataType.ARRAY:
                field2data[name] = field_data_extractors.decode_range(field, start, end), field_meta
                continue

            if dtype == DataType._ARRAY_OF_STRUCT:
                struct_array_data = []

                if hasattr(field, "struct_arrays") and field.struct_arrays:
                    for row_idx in range(start, end):
                        struct_array_data.append(
                            entity_helper.extract_struct_array_from_column_data(
                                field.struct_arrays, row_idx
                            )
                        )

                field2data[name] = (struct_array_data, field_meta)
                continue

            # vectors
            dim = field.vectors.dim
            field_meta.vectors.dim = dim
            if is_dense_vector_type(dtype):
                data = field_data_extractors.get_field_data(field)
                width = _dense_result_slice_width(dtype, dim)
                if dtype == DataType.FLOAT_VECTOR and start == 0 and end * width >= len(data):
                    # If the range equals to the length of vectors.float_vector.data, direct return
                    # it to avoid a copy. This logic improves performance by 25% for the case
                    # retrival 1536 dim embeddings with topk=16384.
                    field2data[name] = data, field_meta
                else:
                    field2data[name] = (
                        data[start * width : end * width],
                        field_meta,
                    )
                continue

            # TODO(SPARSE): do we want to allow the user to specify the return format?
            if is_sparse_vector_type(dtype):
                field2data[name] = field_data_extractors.decode_range(field, start, end), field_meta
                continue
        return field2data

    def get_session_ts(self):
        """Iterator related inner method"""
        # TODO(Goose): change it into properties
        return self._session_ts

    def get_search_iterator_v2_results_info(self):
        """Iterator related inner method"""
        # TODO(Goose): Change it into properties
        return self._search_iterator_v2_results


class Hits(list):
    """List[Dict] Topk search result with pks, distances, and output fields.

        [
            {"id": 1, "distance": 0.3, "entity": {"vector": [1, 2, 3]}},
            {"id": 2, "distance": 0.2, "entity": {"vector": [4, 5, 6]}},
            {"id": 3, "distance": 0.1, "entity": {"vector": [7, 8, 9]}},
        ]

    Examples:
        >>> res = client.search()
        >>> hits = res[0]
        >>> for hit in hits:
        >>>     print(hit)
        {"id": 1, "distance": 0.3, "entity": {"vector": [1, 2, 3]}}
        {"id": 2, "distance": 0.2, "entity": {"vector": [4, 5, 6]}}
        {"id": 3, "distance": 0.1, "entity": {"vector": [7, 8, 9]}}

    Attributes:
        ids(List[Union[str, int]]): topk primary keys
        distances(List[float]): topk distances
    """

    ids: List[Union[str, int]]
    distances: List[float]

    def __init__(
        self,
        topk: int,
        pks: List[Union[int, str]],
        distances: List[float],
        fields: Dict[str, Tuple[List[Any], schema_pb2.FieldData]],
        output_fields: List[str],
        pk_name: str,
    ):
        """
        Args:
            fields(Dict[str, Tuple[List[Any], schema_pb2.FieldData]]):
                field name to a tuple of topk data and field meta
        """
        self.ids = pks
        self.distances = distances

        all_fields = list(fields.keys())
        dynamic_fields = list(set(output_fields) - set(all_fields))

        top_k_res = []
        for i in range(topk):
            entity = {}
            for fname, (data, field_meta) in fields.items():
                if len(data) <= i:
                    entity[fname] = None
                # Get dense vectors
                if is_dense_vector_type(field_meta.type):
                    dim = _dense_result_slice_width(field_meta.type, field_meta.vectors.dim)
                    entity[fname] = data[i * dim : (i + 1) * dim]
                    continue

                # Get dynamic fields
                if field_meta.type == DataType.JSON and field_meta.is_dynamic:
                    if len(dynamic_fields) > 0:
                        entity.update({k: v for k, v in data[i].items() if k in dynamic_fields})
                        continue

                    if fname in output_fields:
                        entity.update(data[i])
                        continue

                # sparse float vector and other fields
                entity[fname] = data[i]
            top_k_res.append(
                Hit({pk_name: pks[i], "distance": distances[i], "entity": entity}, pk_name=pk_name)
            )

        super().__init__(top_k_res)

    def __str__(self) -> str:
        """Only print at most 10 query results"""
        reminder = f" ... and {len(self) - 10} entities remaining" if len(self) > 10 else ""
        return f"{self[:10]}{reminder}"

    __repr__ = __str__


class Hit(dict):
    """Enhanced result in dict that can get data in dict[dict]

    Examples:
        >>> res = {
        >>>     "my_id": 1,
        >>>     "distance": 0.3,
        >>>     "entity": {
        >>>         "emb": [1, 2, 3],
        >>>         "desc": "a description"
        >>>     }
        >>> }
        >>> h = Hit(res, pk_name="my_id")
        >>> h
        {"my_id": 1, "distance": 0.3, "entity": {"emb": [1, 2, 3], "desc": "a description"}}
        >>> h["my_id"]
        1
        >>> h["distance"]
        0.3
        >>> h["entity"]["emb"]
        [1, 2, 3]
        >>> h["entity"]["desc"]
        "a description"
        >>> h.get("emb")
        [1, 2, 3]
    """

    def __init__(self, *args, pk_name: str = "", **kwargs):
        super().__init__(*args, **kwargs)

        self._pk_name = pk_name

    def __getattr__(self, item: str):
        """Patch for orm, will be deprecated soon"""

        # hit.entity return self
        if item == "entity":
            return self

        try:
            return self.__getitem__(item)
        except KeyError as exc:
            raise AttributeError from exc

    def to_dict(self) -> Dict[str, Any]:
        """Patch for orm, will be deprecated soon"""
        return self

    @property
    def id(self) -> Union[str, int]:
        """Patch for orm, will be deprecated soon"""
        return self.get(self._pk_name)

    @property
    def distance(self) -> float:
        """Patch for orm, will be deprecated soon"""
        return self.get("distance")

    @property
    def pk(self) -> Union[str, int]:
        """Alias of id, will be deprecated soon"""
        return self.id

    @property
    def score(self) -> float:
        """Alias of distance, will be deprecated soon"""
        return self.distance

    @property
    def highlight(self) -> Dict[str, Any]:
        return self.get("highlight")

    @property
    def fields(self) -> Dict[str, Any]:
        """Patch for orm, will be deprecated soon"""
        return self.get("entity")

    def __getitem__(self, key: str):
        try:
            return super().__getitem__(key)
        except KeyError:
            return super().__getitem__("entity")[key]

    def get(self, key: Any, default: Any = None):
        try:
            return self.__getitem__(key)
        except KeyError:
            pass
        return default


def extract_array_row_data(
    scalars: List[schema_pb2.ScalarField], element_type: DataType
) -> List[List[Any]]:
    attr = get_array_element_attr(element_type)
    if attr is None:
        raise MilvusException(message=f"Unsupported data type: {element_type}")

    row = []
    for ith_array in scalars:
        if ith_array is None:
            row.append(None)
        else:
            row.append(getattr(ith_array, attr).data)
    return row


def apply_valid_data(data: List[Any], valid_data: Union[None, List[bool]]) -> List[Any]:

    if valid_data:
        return [d if valid else None for d, valid in zip(data, valid_data)]
    return data


def extract_struct_field_value(field_data: schema_pb2.FieldData, index: int) -> Any:
    """Extract a single value from a struct field at the given index."""
    if field_data.type == DataType.BOOL:
        if index < len(field_data.scalars.bool_data.data):
            return field_data.scalars.bool_data.data[index]
    elif field_data.type in (DataType.INT8, DataType.INT16, DataType.INT32):
        if index < len(field_data.scalars.int_data.data):
            return field_data.scalars.int_data.data[index]
    elif field_data.type == DataType.INT64:
        if index < len(field_data.scalars.long_data.data):
            return field_data.scalars.long_data.data[index]
    elif field_data.type == DataType.FLOAT:
        if index < len(field_data.scalars.float_data.data):
            return np.single(field_data.scalars.float_data.data[index])
    elif field_data.type == DataType.DOUBLE:
        if index < len(field_data.scalars.double_data.data):
            return field_data.scalars.double_data.data[index]
    elif field_data.type == DataType.VARCHAR:
        if index < len(field_data.scalars.string_data.data):
            return field_data.scalars.string_data.data[index]
    elif field_data.type == DataType.JSON:
        if index < len(field_data.scalars.json_data.data):
            return orjson.loads(field_data.scalars.json_data.data[index])
    elif field_data.type == DataType.FLOAT_VECTOR:
        dim = field_data.vectors.dim
        start_idx = index * dim
        end_idx = start_idx + dim
        if end_idx <= len(field_data.vectors.float_vector.data):
            return field_data.vectors.float_vector.data[start_idx:end_idx]
    elif field_data.type == DataType.BINARY_VECTOR:
        dim = field_data.vectors.dim // 8
        start_idx = index * dim
        end_idx = start_idx + dim
        if end_idx <= len(field_data.vectors.binary_vector):
            return field_data.vectors.binary_vector[start_idx:end_idx]
    return None

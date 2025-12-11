import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import orjson

from pymilvus.client.types import DataType
from pymilvus.exceptions import MilvusException
from pymilvus.grpc_gen import common_pb2, schema_pb2
from pymilvus.grpc_gen.schema_pb2 import FieldData

from .type_handlers import get_type_registry

logger = logging.getLogger(__name__)


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

        # Create initial hits
        top_k_res = [
            Hit({pk_name: all_pks[i], "distance": all_scores[i], "entity": {}}, pk_name=pk_name)
            for i in range(start, end)
        ]

        # Process fields data (separate lazy fields and extract scalar fields immediately)
        self._process_fields_data(fields_data, top_k_res, start)

        # Apply highlights
        self._apply_highlights(highlight_results, top_k_res, start)

        super().__init__(top_k_res)

    def _process_fields_data(
        self,
        fields_data: List[schema_pb2.FieldData],
        hits: List["Hit"],
        start: int,
    ) -> None:
        """Process fields data: separate lazy fields and extract scalar fields immediately."""
        for field_data in fields_data:
            try:
                handler = get_type_registry().get_handler(field_data.type)
            except ValueError as err:
                # Handler not found
                msg = f"Unsupported field type: {field_data.type}"
                raise MilvusException(msg) from err

            # Use handler.is_lazy_field() to determine if field should be lazy loaded
            if handler.is_lazy_field():
                self.lazy_field_data.append(field_data)
            elif field_data.type == DataType.ARRAY:
                # ARRAY type needs special handling using extract_from_field_data
                for i, hit in enumerate(hits):
                    idx = i + start
                    handler.extract_from_field_data(field_data, idx, hit["entity"], None)
            else:
                # Scalar types: extract immediately
                self._extract_and_fill_scalar_field(field_data, handler, hits, start)

    def _extract_and_fill_scalar_field(
        self,
        field_data: schema_pb2.FieldData,
        handler: Any,
        hits: List["Hit"],
        start: int,
    ) -> None:
        """Extract scalar field data and fill into hits."""
        data = handler.get_raw_data(field_data)
        has_valid = len(field_data.valid_data) > 0

        for i, hit in enumerate(hits):
            idx = i + start
            # Check data length to avoid IndexError
            if idx >= len(data):
                value = None
            elif has_valid and idx < len(field_data.valid_data):
                # Check valid_data index bounds to avoid IndexError
                value = data[idx] if field_data.valid_data[idx] else None
            elif has_valid:
                # valid_data length insufficient, treat as invalid
                value = None
            else:
                value = data[idx]
            hit["entity"][field_data.field_name] = value

    def _apply_highlights(
        self,
        highlight_results: List[common_pb2.HighlightResult],
        hits: List["Hit"],
        start: int,
    ) -> None:
        """Apply highlight results to hits."""
        if not highlight_results:
            return

        for i, hit in enumerate(hits):
            hit["highlight"] = {
                result.field_name: list(result.datas[i + start].fragments)
                for result in highlight_results
            }

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

    def materialize(self):
        if not self.has_materialized:
            for field_data in self.lazy_field_data:
                # Use type handler for all types
                try:
                    handler = get_type_registry().get_handler(field_data.type)
                    context = {
                        "dynamic_fields": self.dynamic_fields,  # For JSON handler
                    }
                    for i in range(len(self)):
                        item = self.get_raw_item(i)
                        # Note: index needs to account for self.start offset
                        actual_index = self.start + i
                        handler.extract_from_field_data(
                            field_data, actual_index, item["entity"], context
                        )
                except ValueError as err:
                    # Handler not found
                    msg = f"Unsupported field type: {field_data.type}"
                    raise MilvusException(msg) from err

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
        """Extract field data for a range of rows.

        This method handles batch extraction directly instead of delegating to handlers,
        because batch extraction logic (slicing, valid_data handling) is not type-specific.
        """
        field2data: Dict[str, Tuple[List[Any], schema_pb2.FieldData]] = {}

        for field in all_fields_data:
            name, dtype = field.field_name, field.type
            field_meta = schema_pb2.FieldData(
                type=dtype,
                field_name=name,
                field_id=field.field_id,
                is_dynamic=field.is_dynamic,
            )

            try:
                handler = get_type_registry().get_handler(dtype)
            except ValueError:
                # Handler not found, skip this field
                continue

            # Extract data using handler's get_raw_data and direct slicing
            try:
                extracted_data = self._extract_batch_data(field, start, end, handler, field_meta)
                field2data[name] = (extracted_data, field_meta)
            except Exception as e:
                logger.warning(f"Failed to extract batch data for {name} ({dtype}): {e}")
                continue

        return field2data

    def _extract_batch_data(
        self,
        field_data: schema_pb2.FieldData,
        start: int,
        end: int,
        handler: Any,
        field_meta: schema_pb2.FieldData,
    ) -> List[Any]:
        """Extract batch data from a field.

        Handles type-specific extraction via handler, and generic logic
        (slicing, valid_data) directly.
        """
        from pymilvus.client import entity_helper  # noqa: PLC0415

        from .type_handlers import get_type_registry  # noqa: PLC0415

        dtype = field_data.type

        if dtype == DataType.FLOAT_VECTOR:
            dim = field_data.vectors.dim
            field_meta.vectors.dim = dim
            raw_data = handler.get_raw_data(field_data)
            return raw_data[start * dim : end * dim]

        if dtype in (
            DataType.FLOAT16_VECTOR,
            DataType.BFLOAT16_VECTOR,
            DataType.INT8_VECTOR,
            DataType.BINARY_VECTOR,
        ):
            dim = field_data.vectors.dim
            field_meta.vectors.dim = dim
            raw_data = handler.get_raw_data(field_data)
            bytes_per_vector = handler.get_bytes_per_vector(dim)
            return raw_data[start * bytes_per_vector : end * bytes_per_vector]

        if dtype == DataType.SPARSE_FLOAT_VECTOR:
            return entity_helper.sparse_proto_to_rows(
                field_data.vectors.sparse_float_vector, start, end
            )

        if dtype == DataType.JSON:
            raw_data = handler.get_raw_data(field_data)[start:end]
            json_list = []
            for item in raw_data:
                if item is not None:
                    try:
                        json_list.append(orjson.loads(item))
                    except Exception as e:
                        logger.error(f"Failed to load JSON: {e}")
                        raise
                else:
                    json_list.append(item)
            return self._apply_valid_data(json_list, field_data, start, end)

        if dtype == DataType.ARRAY:
            raw_data = handler.get_raw_data(field_data)[start:end]
            element_type = field_data.scalars.array_data.element_type
            element_handler = get_type_registry().get_handler(element_type)
            extracted = []
            for scalar_field in raw_data:
                if scalar_field is None:
                    extracted.append(None)
                else:
                    try:
                        element_data = element_handler.extract_from_scalar_field(scalar_field)
                        extracted.append(element_data if element_data else None)
                    except NotImplementedError:
                        extracted.append(None)
            return self._apply_valid_data(extracted, field_data, start, end)

        if dtype == DataType._ARRAY_OF_STRUCT:
            struct_array_data = []
            if hasattr(field_data, "struct_arrays") and field_data.struct_arrays:
                for row_idx in range(start, end):
                    struct_array_data.append(
                        entity_helper.extract_struct_array_from_column_data(
                            field_data.struct_arrays, row_idx
                        )
                    )
            return struct_array_data

        raw_data = handler.get_raw_data(field_data)
        sliced = raw_data[start:end]
        if not isinstance(sliced, list):
            sliced = list(sliced) if hasattr(sliced, "__iter__") else [sliced]
        return self._apply_valid_data(sliced, field_data, start, end)

    def _apply_valid_data(
        self, data: List[Any], field_data: schema_pb2.FieldData, start: int, end: int
    ) -> List[Any]:
        """Apply valid_data mask to extracted data."""
        if hasattr(field_data, "valid_data") and field_data.valid_data:
            result = list(data)
            for i, valid in enumerate(field_data.valid_data[start:end]):
                if not valid:
                    result[i] = None
            return result
        return data

    def get_session_ts(self):
        """Iterator related inner method"""
        # TODO(Goose): change it into properties
        return self._session_ts

    def get_search_iterator_v2_results_info(self):
        """Iterator related inner method"""
        # TODO(Goose): Change it into properties
        return self._search_iterator_v2_results


def get_field_data(field_data: FieldData):
    """Get raw data field from FieldData using type handler."""
    try:
        handler = get_type_registry().get_handler(field_data.type)
        return handler.get_raw_data(field_data)
    except ValueError as err:
        msg = f"Unsupported field type: {field_data.type}"
        raise MilvusException(msg) from err


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
                    continue

                # Extract field value
                field_value = self._extract_field_value_for_hit(
                    fname, data, field_meta, i, dynamic_fields, output_fields
                )

                # Handle field value assignment
                # For JSON dynamic fields, None means exclude (don't add to entity)
                # For other fields, None is a valid value and should be added
                if field_value is None:
                    # Check if this is a JSON dynamic field exclusion
                    if field_meta.type == DataType.JSON and field_meta.is_dynamic:
                        # JSON dynamic field excluded, skip
                        continue
                    # For non-JSON fields, None is a valid value
                    entity[fname] = None
                elif (
                    isinstance(field_value, dict)
                    and field_meta.type == DataType.JSON
                    and field_meta.is_dynamic
                ):
                    # JSON dynamic fields: merge into entity
                    entity.update(field_value)
                else:
                    # Regular field value (including non-dynamic JSON fields)
                    entity[fname] = field_value

            top_k_res.append(
                Hit({pk_name: pks[i], "distance": distances[i], "entity": entity}, pk_name=pk_name)
            )

        super().__init__(top_k_res)

    def _extract_field_value_for_hit(
        self,
        fname: str,
        data: List[Any],
        field_meta: schema_pb2.FieldData,
        index: int,
        dynamic_fields: List[str],
        output_fields: List[str],
    ) -> Any:
        """
        Extract field value for a hit.

        For JSON dynamic fields, returns:
        - Filtered dict if dynamic_fields specified
        - Full dict if field_name in output_fields
        - None to indicate exclusion

        For other fields, returns the raw value.
        """
        try:
            handler = get_type_registry().get_handler(field_meta.type)
        except ValueError:
            # Handler not found, return raw value
            return data[index]

        # Extract raw value
        if handler.is_lazy_field():
            # Lazy fields: use dimension-based slicing or direct access
            dim = field_meta.vectors.dim if hasattr(field_meta, "vectors") else None
            bytes_per_vector = handler.get_bytes_per_vector(dim) if dim else 0
            if bytes_per_vector > 0:
                value = data[index * bytes_per_vector : (index + 1) * bytes_per_vector]
            else:
                value = data[index]
        else:
            # Scalar fields: direct access
            value = data[index]
            # Apply valid_data if present
            if (
                len(field_meta.valid_data) > 0
                and index < len(field_meta.valid_data)
                and not field_meta.valid_data[index]
            ):
                value = None

        # Handle JSON dynamic field filtering
        if field_meta.type == DataType.JSON and field_meta.is_dynamic and isinstance(value, dict):
            if dynamic_fields:
                # Filter to include only requested dynamic fields
                filtered = {k: v for k, v in value.items() if k in dynamic_fields}
                return filtered if filtered else None
            if fname in output_fields:
                # Include all fields
                return value
            # Exclude this field
            return None

        return value

    def __str__(self) -> str:
        """Only print at most 10 query results"""
        reminder = f" ... and {len(self) - 10} entities remaining" if len(self) > 10 else ""
        return f"{self[:10]}{reminder}"

    __repr__ = __str__


from collections import UserDict


class Hit(UserDict):
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
        return self.data.get(self._pk_name)

    @property
    def distance(self) -> float:
        """Patch for orm, will be deprecated soon"""
        return self.data.get("distance")

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
        return self.data.get("highlight")

    @property
    def fields(self) -> Dict[str, Any]:
        """Patch for orm, will be deprecated soon"""
        return self.get("entity")

    def __getitem__(self, key: str):
        try:
            return self.data[key]
        except KeyError:
            pass
        return self.data["entity"][key]

    def get(self, key: Any, default: Any = None):
        try:
            return self.__getitem__(key)
        except KeyError:
            pass
        return default


def _extract_array_element_data(
    scalar_field: schema_pb2.ScalarField, element_type: DataType
) -> Any:
    """Extract array element data using handler for element type."""
    try:
        handler = get_type_registry().get_handler(element_type)
        # Use handler's extract_from_scalar_field method
        return handler.extract_from_scalar_field(scalar_field)
    except (ValueError, NotImplementedError):
        # Handler not found or method not implemented, return None
        return None


def extract_array_row_data(
    scalars: List[schema_pb2.ScalarField], element_type: DataType
) -> List[List[Any]]:
    row = []
    for ith_array in scalars:
        if ith_array is None:
            row.append(None)
            continue

        element_data = _extract_array_element_data(ith_array, element_type)
        if element_data is not None:
            row.append(element_data)
    return row


def apply_valid_data(
    data: List[Any], valid_data: Union[None, List[bool]], start: int, end: int
) -> List[Any]:
    if valid_data:
        for i, valid in enumerate(valid_data[start:end]):
            if not valid:
                data[i] = None
    return data


def extract_struct_field_value(field_data: schema_pb2.FieldData, index: int) -> Any:
    """Extract a single value from a struct field at the given index using type handler."""
    try:
        handler = get_type_registry().get_handler(field_data.type)
        row_data = {}
        handler.extract_from_field_data(field_data, index, row_data)
        # Return the extracted value if found
        if field_data.field_name in row_data:
            return row_data[field_data.field_name]
    except (ValueError, KeyError):
        # Handler not found or field not extracted, return None
        pass
    return None

from typing import Dict, Tuple

from pymilvus.exceptions import ParamError


class IndexParam:
    def __init__(self, field_name: str, index_type: str, index_name: str, **kwargs):
        """
        Examples:

            >>> IndexParam(
            >>>     field_name="embeddings",
            >>>     index_type="HNSW",
            >>>     index_name="hnsw_index",
            >>>     metric_type="COSINE",
            >>>     M=64,
            >>>     efConstruction=100,
            >>> )
        """
        self._field_name = field_name
        self._index_type = index_type
        self._index_name = index_name

        # index configs are unique to each index,
        # if params={} is passed in, it will be flattened and merged
        # with other configs. None is treated as an empty dict so callers
        # can omit the value explicitly.
        params = kwargs.pop("params", None) or {}
        if not isinstance(params, dict):
            msg = f"params must be a dict or None, got {type(params).__name__}: {params!r}"
            raise ParamError(message=msg)

        self._configs = {}
        self._configs.update(params)
        self._configs.update(kwargs)

    @property
    def field_name(self):
        return self._field_name

    @property
    def index_name(self):
        return self._index_name

    @property
    def index_type(self):
        return self._index_type

    def get_index_configs(self) -> Dict:
        """return index_type and index configs in a dict

        Examples:

            {
                "index_type": "HNSW",
                "metrics_type": "COSINE",
                "M": 64,
                "efConstruction": 100,
            }
        """
        configs = self._configs
        # Omit index_type if it's empty
        if self.index_type:
            configs["index_type"] = self.index_type
        return configs

    def to_dict(self) -> Dict:
        """All params"""
        return {
            "field_name": self.field_name,
            "index_type": self.index_type,
            "index_name": self.index_name,
            **self._configs,
        }

    def __str__(self):
        return str(self.to_dict())

    __repr__ = __str__

    def __eq__(self, other: None):
        if isinstance(other, self.__class__):
            return dict(self) == dict(other)

        if isinstance(other, dict):
            return dict(self) == other
        return False


class IndexParams(list):
    """List of indexs of a collection"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_index(self, field_name: str, index_type: str = "", index_name: str = "", **kwargs):
        index_param = IndexParam(field_name, index_type, index_name, **kwargs)
        super().append(index_param)


def extract_bound_index_param(
    output_field_name: str, index_params: "IndexParams"
) -> Tuple[str, Dict]:
    """Validate the bound index of a function output field and extract it.

    The bound index meta is mandatory for add_function_field: without it,
    backfill compaction of the new vector output field would stall on a
    missing index. Rejects invalid input before any RPC (the server enforces
    the same rules) and returns ``(index_name, index_configs)``.
    """
    if not isinstance(index_params, IndexParams):
        raise ParamError(
            message=(
                "wrong type of argument index_params, "
                f"expected: IndexParams, got: {type(index_params).__name__}"
            )
        )
    if len(index_params) != 1:
        raise ParamError(
            message="index_params must contain exactly one index for the function output field"
        )
    index_param = index_params[0]
    if index_param.field_name not in ("", output_field_name):
        raise ParamError(
            message=(
                f"index_params field_name {index_param.field_name!r} does not match "
                f"the function output field {output_field_name!r}"
            )
        )
    if not index_param.index_type:
        raise ParamError(
            message="an explicit index_type is required for the bound index of the function output field"
        )
    # Copy: get_index_configs() exposes IndexParam's internal dict.
    return index_param.index_name, dict(index_param.get_index_configs())

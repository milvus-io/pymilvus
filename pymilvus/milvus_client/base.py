"""Base class for Milvus clients."""

from typing import Dict, List

from pymilvus.orm.collection import CollectionSchema, FieldSchema
from pymilvus.orm.connections import connections
from pymilvus.orm.schema import StructFieldSchema
from pymilvus.orm.types import DataType

from .check import validate_param
from .index import IndexParams


class BaseMilvusClient:
    """Base class for Milvus clients (synchronous and asynchronous)."""

    @classmethod
    def create_schema(cls, **kwargs):
        """Create a collection schema.

        Args:
            **kwargs: Additional keyword arguments for schema creation.

        Returns:
            CollectionSchema: The created collection schema.
        """
        kwargs["check_fields"] = False  # do not check fields for now
        return CollectionSchema([], **kwargs)

    @classmethod
    def create_struct_field_schema(cls) -> StructFieldSchema:
        """Create a struct field schema.

        Returns:
            StructFieldSchema: The created struct field schema.
        """
        return StructFieldSchema()

    @classmethod
    def create_field_schema(
        cls, name: str, data_type: DataType, desc: str = "", **kwargs
    ) -> FieldSchema:
        """Create a field schema. Wrapping orm.FieldSchema.

        Args:
            name (str): The name of the field.
            data_type (DataType): The data type of the field.
            desc (str): The description of the field.
            **kwargs: Additional keyword arguments.

        Returns:
            FieldSchema: the FieldSchema created.
        """
        return FieldSchema(name, data_type, desc, **kwargs)

    @classmethod
    def prepare_index_params(cls, field_name: str = "", **kwargs) -> IndexParams:
        """Prepare index parameters.

        Args:
            field_name (str): The name of the field to create index for.
            **kwargs: Additional keyword arguments for index creation.

        Returns:
            IndexParams: The created index parameters.
        """
        index_params = IndexParams()
        if field_name:
            validate_param("field_name", field_name, str)
            index_params.add_index(field_name, **kwargs)
        return index_params

    def get_server_type(self) -> str:
        """Get the server type.

        Returns:
            str: The server type (e.g., "milvus", "zilliz").
        """
        return self._get_connection().get_server_type()

    def _get_connection(self):
        """Get the connection handler.

        Returns:
            The connection handler instance.
        """
        return connections._fetch_handler(self._using)

    def _extract_primary_field(self, schema_dict: Dict) -> dict:
        """Extract the primary field from a schema dictionary.

        Args:
            schema_dict (Dict): The schema dictionary.

        Returns:
            dict: The primary field dictionary, or empty dict if not found.
        """
        fields = schema_dict.get("fields", [])
        if not fields:
            return {}

        for field_dict in fields:
            if field_dict.get("is_primary", None) is not None:
                return field_dict

        return {}

    def _pack_pks_expr(self, schema_dict: Dict, pks: List) -> str:
        """Pack primary keys into an expression string.

        Args:
            schema_dict (Dict): The schema dictionary.
            pks (List): List of primary key values.

        Returns:
            str: The expression string for filtering by primary keys.
        """
        primary_field = self._extract_primary_field(schema_dict)
        pk_field_name = primary_field["name"]
        data_type = primary_field["type"]

        # Varchar pks need double quotes around the values
        if data_type == DataType.VARCHAR:
            ids = ["'" + str(entry) + "'" for entry in pks]
            expr = f"""{pk_field_name} in [{",".join(ids)}]"""
        else:
            ids = [str(entry) for entry in pks]
            expr = f"{pk_field_name} in [{','.join(ids)}]"
        return expr

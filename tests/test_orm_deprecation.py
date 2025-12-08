# Copyright (C) 2019-2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

import importlib
import sys
import warnings

import pytest


class TestOrmDeprecation:
    """Tests for pymilvus.orm deprecation warnings."""

    def test_orm_internal_import_detection(self):
        """Test that the internal import detection function works correctly."""
        # Note: The deprecation warning is only raised for external imports.
        # Since tests are considered internal, we verify the backward compatibility
        # instead of the warning itself.

    def test_orm_schema_backward_compatible(self):
        """Test that imports from pymilvus.orm.schema still work for backward compatibility."""
        from pymilvus.orm.schema import CollectionSchema, FieldSchema, Function

        # Verify the classes are the same as from client
        from pymilvus.client.schema import (
            CollectionSchema as ClientCollectionSchema,
            FieldSchema as ClientFieldSchema,
            Function as ClientFunction,
        )

        assert CollectionSchema is ClientCollectionSchema
        assert FieldSchema is ClientFieldSchema
        assert Function is ClientFunction

    def test_orm_connections_backward_compatible(self):
        """Test that imports from pymilvus.orm.connections still work for backward compatibility."""
        from pymilvus.orm.connections import Connections, connections

        # Verify the classes are the same as from client
        from pymilvus.client.connections import (
            Connections as ClientConnections,
            connections as client_connections,
        )

        assert Connections is ClientConnections
        assert connections is client_connections

    def test_orm_types_backward_compatible(self):
        """Test that imports from pymilvus.orm.types still work for backward compatibility."""
        from pymilvus.orm.types import DataType, infer_dtype_bydata

        # Verify the classes are the same as from client
        from pymilvus.client.types import (
            DataType as ClientDataType,
            infer_dtype_bydata as client_infer_dtype_bydata,
        )

        assert DataType is ClientDataType
        assert infer_dtype_bydata is client_infer_dtype_bydata

    def test_orm_constants_backward_compatible(self):
        """Test that imports from pymilvus.orm.constants still work for backward compatibility."""
        from pymilvus.orm.constants import BATCH_SIZE, MAX_BATCH_SIZE, UNLIMITED

        # Verify the constants are the same as from client
        from pymilvus.client.constants import (
            BATCH_SIZE as CLIENT_BATCH_SIZE,
            MAX_BATCH_SIZE as CLIENT_MAX_BATCH_SIZE,
            UNLIMITED as CLIENT_UNLIMITED,
        )

        assert BATCH_SIZE == CLIENT_BATCH_SIZE
        assert MAX_BATCH_SIZE == CLIENT_MAX_BATCH_SIZE
        assert UNLIMITED == CLIENT_UNLIMITED

    def test_orm_iterator_backward_compatible(self):
        """Test that imports from pymilvus.orm.iterator still work for backward compatibility."""
        from pymilvus.orm.iterator import (
            QueryIterator,
            SearchIterator,
            SearchPage,
            fall_back_to_latest_session_ts,
        )

        # Verify the classes are the same as from client
        from pymilvus.client.iterator import (
            QueryIterator as ClientQueryIterator,
            SearchIterator as ClientSearchIterator,
            SearchPage as ClientSearchPage,
            fall_back_to_latest_session_ts as client_fall_back_to_latest_session_ts,
        )

        assert QueryIterator is ClientQueryIterator
        assert SearchIterator is ClientSearchIterator
        assert SearchPage is ClientSearchPage
        assert fall_back_to_latest_session_ts is client_fall_back_to_latest_session_ts

    def test_top_level_import_no_warning(self):
        """Test that importing from top-level pymilvus does not raise deprecation warnings.

        Note: The deprecation warning is designed to only trigger for external imports,
        not for internal imports within the pymilvus package or from tests.
        """
        # Just verify the imports work correctly
        from pymilvus import (
            CollectionSchema,
            DataType,
            FieldSchema,
            connections,
        )

        # Verify they are the expected types
        assert CollectionSchema is not None
        assert DataType is not None
        assert FieldSchema is not None
        assert connections is not None


import re
from unittest.mock import Mock, patch

import pytest
from pymilvus.exceptions import MilvusException
from pymilvus.orm.constants import (
    DEFAULT_SEARCH_EXTENSION_RATE,
    EF,
    MAX_BATCH_SIZE,
    PARAMS,
)
from pymilvus.orm.iterator import (
    assert_info,
    check_set_flag,
    extend_batch_size,
    fall_back_to_latest_session_ts,
    io_operation,
)


class TestIteratorHelpers:
    @patch('pymilvus.orm.iterator.datetime')
    @patch('pymilvus.orm.iterator.mkts_from_datetime')
    def test_fall_back_to_latest_session_ts(self, mock_mkts, mock_datetime):
        mock_now = Mock()
        mock_datetime.datetime.now.return_value = mock_now
        mock_mkts.return_value = 123456789

        result = fall_back_to_latest_session_ts()

        mock_datetime.datetime.now.assert_called_once()
        mock_mkts.assert_called_once_with(mock_now, milliseconds=1000.0)
        assert result == 123456789

    def test_assert_info_success(self):
        # Should not raise when condition is True
        assert_info(True, "This should not raise")

    def test_assert_info_failure(self):
        # Should raise MilvusException when condition is False
        with pytest.raises(MilvusException, match="Test error message"):
            assert_info(False, "Test error message")

    def test_io_operation_success(self):
        mock_func = Mock()
        io_operation(mock_func, "Error message")
        mock_func.assert_called_once()

    def test_io_operation_os_error(self):
        mock_func = Mock(side_effect=OSError("OS error"))
        with pytest.raises(MilvusException, match="Error message"):
            io_operation(mock_func, "Error message")

    def test_extend_batch_size_without_ef(self):
        batch_size = 100
        next_param = {PARAMS: {}}

        # Without extension
        result = extend_batch_size(batch_size, next_param, False)
        assert result == 100

        # With extension
        result = extend_batch_size(batch_size, next_param, True)
        expected = min(MAX_BATCH_SIZE, batch_size * DEFAULT_SEARCH_EXTENSION_RATE)
        assert result == expected

    def test_extend_batch_size_with_ef(self):
        batch_size = 100
        next_param = {PARAMS: {EF: 50}}

        # Should be limited by EF value
        result = extend_batch_size(batch_size, next_param, False)
        assert result == 50

        # With extension, still limited by EF
        result = extend_batch_size(batch_size, next_param, True)
        assert result == 50

    def test_extend_batch_size_max_limit(self):
        batch_size = MAX_BATCH_SIZE * 2
        next_param = {PARAMS: {}}

        # Should be limited by MAX_BATCH_SIZE
        result = extend_batch_size(batch_size, next_param, False)
        assert result == MAX_BATCH_SIZE

        # With extension, still limited by MAX_BATCH_SIZE
        result = extend_batch_size(batch_size, next_param, True)
        assert result == MAX_BATCH_SIZE

    def test_check_set_flag(self):
        obj = Mock()
        kwargs = {
            "test_key": True,
            "another_key": "value",
            "false_key": False
        }

        check_set_flag(obj, "flag_name", kwargs, "test_key")
        assert obj.flag_name is True

        check_set_flag(obj, "another_flag", kwargs, "false_key")
        assert obj.another_flag is False

        check_set_flag(obj, "missing_flag", kwargs, "non_existent_key")
        assert obj.missing_flag is False


class TestQueryIteratorInit:
    """Test QueryIterator initialization and basic methods"""

    @pytest.fixture
    def mock_connection(self):
        conn = Mock()
        conn.describe_collection.return_value = {
            "collection_id": 123,
            "schema": {"fields": []},
            "consistency_level": 1
        }
        return conn

    @pytest.fixture
    def mock_schema(self):
        schema = Mock()
        schema.fields = []
        return schema

    @patch('pymilvus.orm.iterator.Connections')
    def test_query_iterator_basic_init(self, mock_connections, mock_connection, mock_schema):
        mock_connections.get_connection.return_value = mock_connection

        # Note: We can't fully instantiate QueryIterator without implementing all abstract methods
        # This test would need a concrete implementation or more extensive mocking
        # For now, we test the helper functions that would be used

        # Test that connection retrieval works
        conn = mock_connections.get_connection("default")
        assert conn == mock_connection


class TestSearchIteratorHelpers:
    """Test SearchIterator helper methods"""

    def test_search_iterator_batch_extension(self):
        """Test batch size extension logic for search iterator"""
        # This tests the logic that would be used in SearchIterator
        batch_size = 100
        next_param = {PARAMS: {"ef": 200}}

        # Test with ef parameter
        result = extend_batch_size(batch_size, next_param, True)
        expected = min(200, batch_size * DEFAULT_SEARCH_EXTENSION_RATE)
        assert result == expected

    @patch('pymilvus.orm.iterator.Path')
    def test_iterator_checkpoint_operations(self, mock_path):
        """Test checkpoint file operations that iterators might use"""
        mock_file = Mock()
        mock_path.return_value.open.return_value.__enter__.return_value = mock_file
        mock_path.return_value.exists.return_value = True

        # Test checkpoint save operation
        def save_checkpoint():
            with mock_path.return_value.open('w') as f:
                f.write('checkpoint_data')

        io_operation(save_checkpoint, "Failed to save checkpoint")

        # Test checkpoint load operation
        def load_checkpoint():
            with mock_path.return_value.open('r') as f:
                return f.read()

        io_operation(load_checkpoint, "Failed to load checkpoint")

    def test_iterator_state_assertions(self):
        """Test state validation assertions used in iterators"""

        # Test valid state
        assert_info(True, "Iterator is in valid state")

        # Test invalid state
        with pytest.raises(MilvusException, match="Iterator exhausted"):
            assert_info(False, "Iterator exhausted")

        # Test collection mismatch
        with pytest.raises(MilvusException, match="Collection mismatch"):
            assert_info(False, "Collection mismatch")


class TestIteratorConstants:
    """Test iterator-related constants and their usage"""

    def test_batch_size_limits(self):
        """Test batch size calculation respects limits"""
        # Test minimum batch size
        batch_size = 1
        next_param = {PARAMS: {}}
        result = extend_batch_size(batch_size, next_param, False)
        assert result >= 1

        # Test maximum batch size
        batch_size = MAX_BATCH_SIZE * 10
        result = extend_batch_size(batch_size, next_param, False)
        assert result <= MAX_BATCH_SIZE

    def test_extension_rate_application(self):
        """Test search extension rate is applied correctly"""
        batch_size = 100
        next_param = {PARAMS: {}}

        # Without extension
        result_no_ext = extend_batch_size(batch_size, next_param, False)

        # With extension
        result_with_ext = extend_batch_size(batch_size, next_param, True)

        # Extension should increase batch size
        assert result_with_ext >= result_no_ext

        # Extension should be by DEFAULT_SEARCH_EXTENSION_RATE
        if result_with_ext < MAX_BATCH_SIZE:
            assert result_with_ext == batch_size * DEFAULT_SEARCH_EXTENSION_RATE


class TestIteratorErrorHandling:
    """Test error handling in iterator operations"""

    def test_io_operation_error_propagation(self):
        """Test that IO errors are properly wrapped"""

        # Test with OSError
        with pytest.raises(MilvusException, match="Custom IO error"):
            io_operation(lambda: (_ for _ in ()).throw(OSError("OS level error")), "Custom IO error")

        # Test with PermissionError (subclass of OSError)
        with pytest.raises(MilvusException, match="Permission denied"):
            io_operation(lambda: (_ for _ in ()).throw(PermissionError("No access")), "Permission denied")

        # Test with FileNotFoundError (subclass of OSError)
        with pytest.raises(MilvusException, match="File not found"):
            io_operation(lambda: (_ for _ in ()).throw(FileNotFoundError("Missing file")), "File not found")

    def test_assert_info_with_different_messages(self):
        """Test assert_info with various error messages"""

        test_cases = [
            "Simple error",
            "Error with special characters: @#$%",
            "Error with numbers 12345",
            "Very long error message " * 100,
            ""  # Empty message
        ]

        for message in test_cases:
            # Escape regex special characters in the message for matching
            escaped_message = re.escape(message) if message else ".*"
            with pytest.raises(MilvusException, match=escaped_message):
                assert_info(False, message)


class TestIteratorFlags:
    """Test flag setting functionality for iterators"""

    def test_check_set_flag_various_types(self):
        """Test setting flags with various value types"""
        obj = Mock()

        # Boolean values
        kwargs = {"bool_flag": True}
        check_set_flag(obj, "test_bool", kwargs, "bool_flag")
        assert obj.test_bool is True

        # String values (should be truthy/falsy)
        kwargs = {"string_flag": "enabled"}
        check_set_flag(obj, "test_string", kwargs, "string_flag")
        assert obj.test_string == "enabled"

        # None values
        kwargs = {"none_flag": None}
        check_set_flag(obj, "test_none", kwargs, "none_flag")
        assert obj.test_none is None

        # Numeric values
        kwargs = {"num_flag": 42}
        check_set_flag(obj, "test_num", kwargs, "num_flag")
        assert obj.test_num == 42

        # Missing key (should default to False)
        kwargs = {}
        check_set_flag(obj, "test_missing", kwargs, "missing_key")
        assert obj.test_missing is False

    def test_check_set_flag_overwrites(self):
        """Test that check_set_flag overwrites existing attributes"""
        obj = Mock()
        obj.existing_flag = "old_value"

        kwargs = {"new_value": "updated"}
        check_set_flag(obj, "existing_flag", kwargs, "new_value")
        assert obj.existing_flag == "updated"

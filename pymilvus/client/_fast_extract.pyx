# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from cpython.dict cimport PyDict_SetItem
from cpython.list cimport PyList_GET_ITEM, PyList_GET_SIZE
from cpython.object cimport PyObject


cpdef void assign_scalar_fast(
    list entity_rows,
    str field_name,
    list data_list,
    list valid_data,
    bint has_valid
):
    """Fast Cython implementation of scalar field assignment.

    Args:
        entity_rows: List of dictionaries to populate
        field_name: Name of the field to set
        data_list: List of values to assign (already converted from protobuf)
        valid_data: List of validity flags (empty if no validity checks)
        has_valid: Whether to check validity
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t row_count = PyList_GET_SIZE(entity_rows)
    cdef PyObject* field_name_obj = <PyObject*>field_name
    cdef dict row
    cdef object value

    if has_valid:
        for i in range(row_count):
            row = <dict>PyList_GET_ITEM(entity_rows, i)
            if <bint>PyList_GET_ITEM(valid_data, i):
                value = <object>PyList_GET_ITEM(data_list, i)
            else:
                value = None
            PyDict_SetItem(row, field_name, value)
    else:
        for i in range(row_count):
            row = <dict>PyList_GET_ITEM(entity_rows, i)
            value = <object>PyList_GET_ITEM(data_list, i)
            PyDict_SetItem(row, field_name, value)


cpdef void assign_array_fast(
    list entity_rows,
    str field_name,
    list array_data,
    list valid_data,
    bint has_valid,
    int element_type
):
    """Fast Cython implementation of array field assignment.

    Args:
        entity_rows: List of dictionaries to populate
        field_name: Name of the field to set
        array_data: List of array proto objects
        valid_data: List of validity flags
        has_valid: Whether to check validity
        element_type: DataType of array elements
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t row_count = PyList_GET_SIZE(entity_rows)
    cdef dict row
    cdef object array_item
    cdef object value

    # element_type values from DataType enum
    # INT64=5, BOOL=1, INT8/16/32=2/3/4, FLOAT=10, DOUBLE=11, STRING/VARCHAR=20/21

    if has_valid:
        for i in range(row_count):
            row = <dict>PyList_GET_ITEM(entity_rows, i)
            if not <bint>PyList_GET_ITEM(valid_data, i):
                PyDict_SetItem(row, field_name, None)
            else:
                array_item = <object>PyList_GET_ITEM(array_data, i)
                # Extract based on element type
                if element_type == 5:  # INT64
                    value = array_item.long_data.data
                elif element_type == 1:  # BOOL
                    value = array_item.bool_data.data
                elif element_type in (2, 3, 4):  # INT8/16/32
                    value = array_item.int_data.data
                elif element_type == 10:  # FLOAT
                    value = array_item.float_data.data
                elif element_type == 11:  # DOUBLE
                    value = array_item.double_data.data
                else:  # STRING/VARCHAR
                    value = array_item.string_data.data
                PyDict_SetItem(row, field_name, value)
    else:
        for i in range(row_count):
            row = <dict>PyList_GET_ITEM(entity_rows, i)
            array_item = <object>PyList_GET_ITEM(array_data, i)
            # Extract based on element type
            if element_type == 5:  # INT64
                value = array_item.long_data.data
            elif element_type == 1:  # BOOL
                value = array_item.bool_data.data
            elif element_type in (2, 3, 4):  # INT8/16/32
                value = array_item.int_data.data
            elif element_type == 10:  # FLOAT
                value = array_item.float_data.data
            elif element_type == 11:  # DOUBLE
                value = array_item.double_data.data
            else:  # STRING/VARCHAR
                value = array_item.string_data.data
            PyDict_SetItem(row, field_name, value)

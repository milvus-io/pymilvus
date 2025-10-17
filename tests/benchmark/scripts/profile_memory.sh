#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <test_name>"
    echo "Example: $0 test_search_bench.py::test_search_float32_basic_scalars"
    exit 1
fi

TEST_NAME=$1
OUTPUT_BIN="mem_profile_$(echo $TEST_NAME | tr ':/' '__').bin"
OUTPUT_HTML="mem_profile_$(echo $TEST_NAME | tr ':/' '__').html"

echo "Memory profiling $TEST_NAME..."
echo "Output files: $OUTPUT_BIN, $OUTPUT_HTML"

memray run -o "$OUTPUT_BIN" pytest "tests/benchmark/$TEST_NAME" -v

echo ""
echo "Generating flamegraph..."
memray flamegraph "$OUTPUT_BIN"

echo ""
echo "Generating summary..."
memray summary "$OUTPUT_BIN"

echo ""
echo "Profiling complete!"
echo "View flamegraph: open $OUTPUT_HTML"
echo ""
echo "Additional commands:"
echo "  memray table $OUTPUT_BIN    # View top allocators"
echo "  memray tree $OUTPUT_BIN     # View call stack tree"

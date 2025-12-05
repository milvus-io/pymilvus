#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <test_name>"
    echo "Example: $0 test_search_bench.py::test_search_float32_basic_scalars"
    exit 1
fi

TEST_NAME=$1
OUTPUT_SVG="cpu_profile_$(echo $TEST_NAME | tr ':/' '__').svg"
OUTPUT_SPEEDSCOPE="cpu_profile_$(echo $TEST_NAME | tr ':/' '__').speedscope.json"

echo "Profiling $TEST_NAME..."
echo "Output files: $OUTPUT_SVG, $OUTPUT_SPEEDSCOPE"

py-spy record -o "$OUTPUT_SVG" --native -- pytest "tests/benchmark/$TEST_NAME" -v

py-spy record -o "$OUTPUT_SPEEDSCOPE" -f speedscope -- pytest "tests/benchmark/$TEST_NAME" -v

echo ""
echo "Profiling complete!"
echo "View SVG: open $OUTPUT_SVG"
echo "View Speedscope: Upload $OUTPUT_SPEEDSCOPE to https://www.speedscope.app/"

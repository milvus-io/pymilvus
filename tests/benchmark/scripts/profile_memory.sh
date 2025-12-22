#!/bin/bash
# Memory profiling script using pytest-memray
# This allows profiling existing benchmark tests without modification

if [ -z "$1" ]; then
    echo "Usage: $0 <test_pattern> [allocations_to_show]"
    echo ""
    echo "Examples:"
    echo "  $0 test_search_bench.py::TestSearchBench::test_search_float32_varying_topk"
    echo "  $0 'test_search_bench.py::TestSearchBench::test_search_float32_varying_topk[10000]'"
    echo "  $0 test_access_patterns_bench.py 20"
    echo ""
    echo "Note: Use quotes for test names with brackets"
    exit 1
fi

TEST_PATTERN=$1
MOST_ALLOCS=${2:-10}
OUTPUT_DIR=".memray_profiles"

mkdir -p "$OUTPUT_DIR"

echo "🔍 Memory profiling: $TEST_PATTERN"
echo "📊 Showing top $MOST_ALLOCS allocators"
echo "📁 Binary dumps: $OUTPUT_DIR"
echo ""

# Run with pytest-memray
pytest "tests/benchmark/$TEST_PATTERN" \
    --memray \
    --memray-bin-path="$OUTPUT_DIR" \
    --most-allocations="$MOST_ALLOCS" \
    --stacks=10 \
    -v

echo ""
echo "✅ Memory profiling complete!"
echo ""
echo "📊 Binary dumps saved to: $OUTPUT_DIR/"
echo ""
echo "🔥 Generate flamegraphs:"
echo "  memray flamegraph $OUTPUT_DIR/memray-*.bin"
echo ""
echo "📋 Additional analysis:"
echo "  memray table $OUTPUT_DIR/memray-*.bin    # Top allocators table"
echo "  memray tree $OUTPUT_DIR/memray-*.bin     # Call stack tree"
echo "  memray summary $OUTPUT_DIR/memray-*.bin  # Summary statistics"

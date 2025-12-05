# pymilvus MilvusClient Benchmarking Suite

This benchmark suite measures client-side performance of pymilvus MilvusClient API operations (search, query, hybrid search) without requiring a running Milvus server.

## Overview

We benchmark **client-side code only** by mocking gRPC calls:
- ✅ Request preparation (parameter validation, serialization)
- ✅ Response parsing (deserialization, type conversion)
- ❌ Network I/O (excluded via mocking)
- ❌ Server-side processing (excluded via mocking)

## Directory Structure

```
tests/benchmark/
├── README.md                # This file - complete guide
├── conftest.py              # Mock gRPC stubs & shared fixtures
├── mock_responses.py        # Fake protobuf response builders
├── test_search_bench.py     # Search timing benchmarks
└── scripts/
    ├── profile_cpu.sh       # CPU profiling wrapper
    └── profile_memory.sh    # Memory profiling wrapper
```

### Installation

```bash
pip install -e ".[dev]"
```

---

## 1. Timing Benchmarks (pytest-benchmark)
### Usage

```bash
# Run all benchmarks
pytest tests/benchmark/ --benchmark-only

# Run specific benchmark
pytest tests/benchmark/test_search_bench.py::TestSearchBench::test_search_float32_varying_output_fields --benchmark-only

# Save baseline for comparison
pytest tests/benchmark/ --benchmark-only --benchmark-save=baseline

# Compare against baseline
pytest tests/benchmark/ --benchmark-only --benchmark-compare=baseline

# Generate histogram
pytest tests/benchmark/ --benchmark-only --benchmark-histogram
```

## 2. CPU Profiling (py-spy)
### Usage

#### Option A: Profile entire benchmark run

```bash
# Generate flamegraph (SVG)
py-spy record -o cpu_profile.svg --native -- pytest tests/benchmark/test_search_bench.py::TestSearchBench::test_search_float32 -v

# Generate speedscope format (interactive viewer)
py-spy record -o cpu_profile.speedscope.json -f speedscope -- pytest tests/benchmark/test_search_bench.py::TestSearchBench::test_search_float32 -v

# View speedscope: Upload to https://www.speedscope.app/
```

#### Option B: Use helper script

```bash
./tests/benchmark/scripts/profile_cpu.sh test_search_bench.py::test_search_float32
```

#### Option C: Profile specific function

```bash
# Top functions by CPU time
py-spy top -- python -m pytest tests/benchmark/test_search_bench.py::test_search_float32 -v
```

## 3. Memory Profiling (memray)

### What it Measures
- Memory allocation over time
- Peak memory usage
- Allocation flamegraphs
- Memory leaks
- Allocation call stacks

### Usage

#### Option A: Profile and generate reports

```bash
# Run with memray
memray run -o search_bench.bin pytest tests/benchmark/test_search_bench.py::test_search_float32 -v

# Generate flamegraph (HTML)
memray flamegraph search_bench.bin

# Generate table view (top allocators)
memray table search_bench.bin

# Generate tree view (call stack)
memray tree search_bench.bin

# Generate summary stats
memray summary search_bench.bin
```

#### Option B: Live monitoring

```bash
# Real-time memory usage in terminal
memray run --live pytest tests/benchmark/test_search_bench.py::test_search_float32 -v
```

#### Option C: Use helper script

```bash
./tests/benchmark/scripts/profile_memory.sh test_search_bench.py::test_search_float32
```

## 6. Complete Workflow

```bash
# Step 1: Install dependencies
pip install -e ".[dev]"

# Step 2: Run timing benchmarks (fast, ~minutes)
pytest tests/benchmark/ --benchmark-only

# Step 3: Identify slow tests from benchmark results

# Step 4: CPU profile specific slow tests
py-spy record -o cpu_slow_test.svg -- pytest tests/benchmark/test_search_bench.py::test_slow_one -v

# Step 5: Memory profile tests with large results
memray run -o mem_large.bin pytest tests/benchmark/test_search_bench.py::test_large_results -v
memray flamegraph mem_large.bin

# Step 6: Analyze results and fix bottlenecks

# Step 7: Re-run benchmarks and compare with baseline
pytest tests/benchmark/ --benchmark-only --benchmark-compare=baseline
```

## Expected Bottlenecks

Based on code analysis, we expect to find:

1. **Protobuf deserialization** - Large responses with many fields
2. **Vector data conversion** - Bytes → numpy arrays
3. **Type conversions** - Protobuf types → Python types
4. **Field iteration** - Processing many output fields
5. **Memory copies** - Unnecessary data duplication

These benchmarks will help us validate and quantify these hypotheses.

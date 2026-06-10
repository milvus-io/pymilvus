# PyMilvus Test Suites

PyMilvus uses directory-targeted Make commands to classify maintained tests:

- `make unittest` runs deterministic unit tests in `tests/unit/` with coverage.
- `make integration-lite` runs embedded Milvus Lite integration tests in `tests/integration/lite/` without coverage upload.
- `make benchmark` runs benchmark workloads in `tests/benchmark/`.

Examples remain user-facing sample programs and are not part of these default test commands.

UV ?= uv
UV_RUN_DEV = $(UV) run --group dev
UV_RUN_DEV_PYTHONPATH = PYTHONPATH=$(CURDIR) $(UV_RUN_DEV)
UV_RUN_LITE = $(UV) run --group dev --extra milvus-lite
UV_RUN_LITE_PYTHONPATH = PYTHONPATH=$(CURDIR) $(UV_RUN_LITE)
# vcs-versioning 2.x changes .dev0 handling away from PyMilvus' intended
# master dev line, so keep all local/workflow version reads on <2.
VERSION_DEPS = --with "hatchling>=1.27,<1.28" --with "setuptools-scm[toml]>=8" --with "vcs-versioning<2"

sync:
	$(UV) sync --group dev

unittest:
	$(UV_RUN_DEV_PYTHONPATH) pytest tests/unit --cov=pymilvus -v

integration-lite:
	$(UV_RUN_LITE_PYTHONPATH) pytest tests/integration/lite -v

benchmark:
	$(UV_RUN_DEV_PYTHONPATH) pytest tests/benchmark --benchmark-only

lint:
	$(UV_RUN_DEV_PYTHONPATH) black pymilvus tests --check --diff
	$(UV_RUN_DEV_PYTHONPATH) ruff check pymilvus tests

format:
	$(UV_RUN_DEV_PYTHONPATH) black pymilvus tests
	$(UV_RUN_DEV_PYTHONPATH) ruff check pymilvus tests --fix

coverage:
	$(UV_RUN_DEV_PYTHONPATH) pytest tests/unit --cov=pymilvus --cov-report=xml -v

example:
	$(UV_RUN_DEV_PYTHONPATH) python examples/example.py

example_index:
	$(UV_RUN_DEV_PYTHONPATH) python examples/example_index.py

package:
	$(UV) build --sdist --wheel --out-dir dist/ .

get_proto:
	git submodule update --init

gen_proto:
	cd pymilvus/grpc_gen && PYTHON_CMD='$(UV) run --group dev python' ./python_gen.sh

check_proto_product: gen_proto
	./check_proto_product.sh

version:
	@$(UV) run --no-project $(VERSION_DEPS) python -m hatchling version

install:
	$(UV) sync

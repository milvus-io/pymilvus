UV ?= uv
UV_RUN_DEV = $(UV) run --group dev
UV_RUN_DEV_PYTHONPATH = PYTHONPATH=$(CURDIR) $(UV_RUN_DEV)

sync:
	$(UV) sync --group dev

unittest:
	$(UV_RUN_DEV_PYTHONPATH) pytest tests --ignore=tests/benchmark --cov=pymilvus -v

lint:
	$(UV_RUN_DEV_PYTHONPATH) black pymilvus tests --check --diff
	$(UV_RUN_DEV_PYTHONPATH) ruff check pymilvus tests

format:
	$(UV_RUN_DEV_PYTHONPATH) black pymilvus tests
	$(UV_RUN_DEV_PYTHONPATH) ruff check pymilvus tests --fix

coverage:
	$(UV_RUN_DEV_PYTHONPATH) pytest --cov=pymilvus --ignore=tests/benchmark tests --cov-report=xml

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
	$(UV) run --no-project --with "hatchling>=1.27,<1.28" --with "setuptools-scm[toml]>=8" python -m hatchling version

install:
	$(UV) sync

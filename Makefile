UV ?= uv
UV_RUN_DEV = $(UV) run --extra dev

sync:
	$(UV) sync --extra dev

unittest:
	PYTHONPATH=$(CURDIR) $(UV_RUN_DEV) python -m pytest tests --ignore=tests/benchmark --cov=pymilvus -v

lint:
	PYTHONPATH=$(CURDIR) $(UV_RUN_DEV) python -m black pymilvus tests --check --diff
	PYTHONPATH=$(CURDIR) $(UV_RUN_DEV) python -m ruff check pymilvus tests

format:
	$(UV) sync --extra dev
	PYTHONPATH=$(CURDIR) $(UV_RUN_DEV) python -m black pymilvus tests
	PYTHONPATH=$(CURDIR) $(UV_RUN_DEV) python -m ruff check pymilvus tests --fix

coverage:
	PYTHONPATH=$(CURDIR) $(UV_RUN_DEV) python -m pytest --cov=pymilvus --ignore=tests/benchmark tests --cov-report=xml

example:
	PYTHONPATH=$(CURDIR) $(UV_RUN_DEV) python examples/example.py

example_index:
	PYTHONPATH=$(CURDIR) $(UV_RUN_DEV) python examples/example_index.py

package:
	$(UV) build --sdist --wheel --out-dir dist/ .

get_proto:
	git submodule update --init

gen_proto:
	$(UV) sync --extra dev
	cd pymilvus/grpc_gen && PYTHON_CMD='$(UV) run --extra dev python' ./python_gen.sh

check_proto_product: gen_proto
	./check_proto_product.sh

version:
	$(UV_RUN_DEV) python -c "import _version_helper; print(_version_helper.version)"

install:
	$(UV) sync

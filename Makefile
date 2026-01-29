unittest:
	PYTHONPATH=`pwd` python3 -m pytest tests --ignore=tests/benchmark --cov=pymilvus -v

lint:
	PYTHONPATH=`pwd` python3 -m black pymilvus tests --check --diff
	PYTHONPATH=`pwd` python3 -m ruff check pymilvus tests

format:
	pip install -e ".[dev]"
	PYTHONPATH=`pwd` python3 -m black pymilvus tests
	PYTHONPATH=`pwd` python3 -m ruff check pymilvus tests --fix

coverage:
	PYTHONPATH=`pwd` pytest --cov=pymilvus --ignore=tests/benchmark tests --cov-report=xml

example:
	PYTHONPATH=`pwd` python examples/example.py

example_index:
	PYTHONPATH=`pwd` python examples/example_index.py

package:
	python3 -m build --sdist --wheel --outdir dist/ .

get_proto:
	git submodule update --init

gen_proto:
	pip install -e ".[dev]"
	cd pymilvus/grpc_gen && ./python_gen.sh

check_proto_product: gen_proto
	./check_proto_product.sh

version:
	python -m setuptools_scm

install:
	pip install -e .

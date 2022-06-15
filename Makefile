unittest:
	PYTHONPATH=`pwd` python3 -m pytest tests --cov=pymilvus -v

lint:
	PYTHONPATH=`pwd` pylint --rcfile=pylint.conf pymilvus

codecov:
	PYTHONPATH=`pwd` pytest --cov=pymilvus --cov-report=xml tests -x -v -rxXs

example:
	PYTHONPATH=`pwd` python examples/example.py

example_index:
	PYTHONPATH=`pwd` python examples/example_index.py

package:
	python3 -m build --sdist --wheel --outdir dist/ .

get_proto:
	git submodule update --init

gen_proto:
	cd pymilvus/grpc_gen && ./python_gen.sh

check_proto_product: gen_proto
	./check_proto_product.sh

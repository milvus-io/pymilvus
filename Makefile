unittest:
	PYTHONPATH=`pwd` python3 -m pytest tests --cov=pymilvus

lint:
	PYTHONPATH=`pwd` pylint --rcfile=pylint.conf pymilvus

codecov:
	PYTHONPATH=`pwd` pytest --cov=pymilvus --cov-report=xml tests -x -rxXs

example:
	PYTHONPATH=`pwd` python examples/example.py

example_index:
	PYTHONPATH=`pwd` python examples/example_index.py

package:
	python3 -m build --sdist --wheel --outdir dist/ .

gen_proto:
	cd pymilvus/grpc_gen/proto && ./python_gen.sh

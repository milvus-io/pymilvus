
unittest:
	PYTHONPATH=`pwd` python3 -m pytest tests
	PYTHONPATH=`pwd` python3 -m pytest pymilvus/orm/tests -x -rxXs

lint:
	PYTHONPATH=`pwd` pylint --rcfile=pylint.conf pymilvus

codecov:
	PYTHONPATH=`pwd` pytest --cov=pymilvus --cov-report=xml pymilvus/orm/tests -x -rxXs

example:
	PYTHONPATH=`pwd` python examples/example.py

example_index:
	PYTHONPATH=`pwd` python examples/example_index.py

package:
	python3 -m build --sdist --wheel --outdir dist/ .



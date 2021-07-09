
unittest:
	PYTHONPATH=`pwd` python3 -m pytest tests

package:
	python3 -m build --sdist --wheel --outdir dist/ .

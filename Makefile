
unittest:
	python -m pytest tests

package:
	python -m build --sdist --wheel --outdir dist/.

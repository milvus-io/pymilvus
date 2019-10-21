import pathlib
import setuptools
import io
import re

HERE = pathlib.Path(__file__).parent

README = (HERE / 'README.md').read_text()

with io.open("milvus/client/__init__.py", "rt", encoding="utf8") as f:
    version = re.search(r'__version__ = "(.*?)"', f.read()).group(1)

setuptools.setup(
    name="pymilvus-test",
    version=version,
    description="Python Sdk for Milvus",
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/milvus-io/pymilvus',
    license="Apache-2.0",
    packages=["milvus.client", 'milvus.grpc_gen', 'milvus'],
    include_package_data=True,
    install_requires=["grpcio>=1.22.0", "grpcio-tools>=1.22.0"],
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],

    python_requires='>=3.5'
)

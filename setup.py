import pathlib
import setuptools
from datetime import datetime
import re

HERE = pathlib.Path(__file__).parent

README = (HERE / 'README.md').read_text()

setuptools.setup(
    name="pymilvus-perf",
    author='Milvus Team',
    author_email='milvus-team@zilliz.com',
    setup_requires=['setuptools_scm'],
    use_scm_version={'local_scheme': 'no-local-version'},
    description="Python Sdk for Milvus",
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/milvus-io/pymilvus',
    license="Apache-2.0",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "grpcio==1.37.1",
        "grpcio-tools==1.37.1",
        "ujson>=2.0.0,<=5.1.0",
        "mmh3>=2.0,<=3.0.0",
        "pandas==1.1.5; python_version<'3.7'",
        "pandas>=1.2.4,<=1.3.5; python_version>'3.6'",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],

    python_requires='>=3.6'
)

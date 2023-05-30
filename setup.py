import pathlib
import setuptools

HERE = pathlib.Path(__file__).parent

README = (HERE / 'README.md').read_text()

setuptools.setup(
    name="pymilvus",
    author='Milvus Team',
    author_email='milvus-team@zilliz.com',
    setup_requires=['setuptools_scm'],
    use_scm_version={'local_scheme': 'no-local-version', 'version_scheme': 'release-branch-semver'},
    description="Python Sdk for Milvus",
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/milvus-io/pymilvus',
    license="Apache-2.0",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "grpcio>=1.49.1,<=1.53.0",
        "protobuf>=3.20.0",
        "environs<=9.5.0",
        "ujson>=2.0.0",
        "pandas>=1.2.4",
        "numpy!=1.25.0rc1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],

    python_requires='>=3.7'
)

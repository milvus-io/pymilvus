import pathlib
import setuptools

HERE = pathlib.Path(__file__).parent

README = (HERE / 'README.md').read_text()

setuptools.setup(
    name="pymilvus-test",
    version="0.2.7",

    description="Python Sdk for Milvus; Alpha version",
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/BossZou/pymilvus',
    license="Apache-2.0",
    packages=["milvus.client", 'milvus.grpc_gen', 'milvus'],
    include_package_data=True,
    install_requires=["grpcio==1.22.0", "grpcio-tools==1.22.0"],
    classifiers=[
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],

    python_requires='>=3.4'
)

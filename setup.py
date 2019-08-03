import pathlib
import setuptools

HERE = pathlib.Path(__file__).parent

README = (HERE / 'README.md').read_text()

setuptools.setup(
    name="pymilvus",
    version="0.1.25",
    description="Python Sdk for Milvus",
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/milvus-io/pymilvus',
    license="Apache-2.0",
    packages=["milvus.client", 'milvus.thrift', 'milvus'],
    include_package_data=True,
    install_requires=["thrift"],
    classifiers=[
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],


    python_requires='>=3.4'
)

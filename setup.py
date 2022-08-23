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
        "grpcio>=1.47.0,<=1.48.0",
        "grpcio-tools>=1.47.0, <=1.48.0",
        "ujson>=2.0.0,<=5.4.0",
        "mmh3>=2.0,<=3.0.0",
        "pandas==1.1.5; python_version<'3.7'",
        "pandas>=1.2.4; python_version>'3.6'",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],

    python_requires='>=3.6'
)

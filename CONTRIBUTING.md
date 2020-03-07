# Contributing to pymilvus

We welcome all kinds of contributions. Simply file an issue stating your reason
and plans for making the change, update CHANGELOG.md, and create a pull request 
to the current active branch. Make sure to refer to the issue you filed in your 
PR's description. Cheers!


## What contributions can I make?

Any contributions are allowed without changing the project architecture and interfaces here.
You are welcome to make contributions.


## How can I contribute?

### Development environment

You are recommended to develop in a virtual environment launched by python environment management tool,
for example, virtualenv, conda and etc. When your virtual environment is activate, run command `pip install -r requirements.txt`
 to install dependent packages.


### Coding Style
Before submitting a pull request, make sure the coding style is qualified. run command `pylint --rcfile=pylint.conf milvus/client` 
to check it.


## Run unit test with code coverage

Before submitting your PR, make sure you have run unit test, and your code coverage rate is >= 90%.

```shell 
$ pytest --cov=milvus/client --cov-report=html
```

You may need a milvus server which is running when you run unit test. See more details on [Milvus server](https://github.com/milvus-io/milvus).


## Update CHANGLOG.md

Add issue tips into CHANGLOG.md, make sure all issue tips are sorted by issue number in ascending order.




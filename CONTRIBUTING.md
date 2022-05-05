# Contributing to PyMilvus

PyMilvus is the Python SDK for [Milvus](https://github.com/milvus-io/milvus), an open-source vector database project on GitHub. Growing along with the development of Milvus, PyMilvus is one of the most popular sub-projects of Milvus. Until Sept. 10th, 2021, PyMilvus has gained more than 300 stars on GitHub and attracted 40 contributors.

Many people interested in Milvus start by using PyMilvus as their first contact with Milvus. PyMilvus 1.x, which is compatible with Milvus 1.x, is a long-term support (LTS) version. PyMilvus 2.x is compatible with Milvus 2.x, and is now under active development.

Projects in the Milvus community all welcome your contributions, and we welcome you to help build this community. PyMilvus differs from other projects of Milvus community because:

- It is a pure Python project;
- It supports Milvus E2E, Benchmark, and Milvus Bootcamp;
- It supports significantly more application scenarios as a Python package.

We are committed to building a collaborative, exuberant open-source community for PyMilvus. Therefore, contributions to PyMilvus are welcome from everyone. Anyone who is familiar with the code and usage of PyMilvus is welcome to contribute to the community, help newcomers, and pass on the open-source, collaborative, and open spirit.


## What contributions can you make?

Issues with label [good-first-issue](https://github.com/milvus-io/pymilvus/labels/good%20first%20issue) and [help-wanted](https://github.com/milvus-io/pymilvus/labels/help%20wanted) in this repo are entry-level issues. They are the perfect starting points if you are trying to get familiar with this project.

If you want to challenge yourself, you may wish to look for issues with the label [Hacktoberfest](https://github.com/milvus-io/pymilvus/labels/Hacktoberfest).

If you identify any problems, you can:
1. [File an issue](https://github.com/milvus-io/pymilvus/issues/new/choose) to report the problem;
2. Describe how to reproduce this problem (optional);
3. Provide any possible solutions to this problem (optional);
4. Submit a Pull Request (PR) to solve this problem (optional).

If you are interested in existing problems, you can:
- Answer questions to offer help in issues with [question](https://github.com/milvus-io/pymilvus/labels/Issue%20%7C%20question) labels;
- In issues with [bug](https://github.com/milvus-io/pymilvus/labels/kind%2Fbug), [enhancement](https://github.com/milvus-io/pymilvus/labels/enhancement) labels:
  - Provide details on the problem, reproducing steps, and solutions;
  - Submit a PR to tackle the problem.

If you want to request more features for PyMilvus, you can:
- [File an issue](https://github.com/milvus-io/pymilvus/issues/new/choose) to to describe the new features and explain why;
- Provide implementation design and test design (optional);
- Submit a PR to implement the feature (optional).

If you are interested in existing PRs, you can:
- Review the code and offer advice;
- Instruct new contributors to complete the PR process.

Note: the problems, features, and questions mentioned here are not limited to Python code. They also refer to all kinds of documents (technical documents, API references, contributing guide, etc.)

## PyMilvus Code Structure
`docs/`: Contains source documentation (except for API Reference) that created by sphinx; most documentation are stored as `.rst` files under `docs/source`.

`examples/`: Contains Python scripts, which can be run directly, for introducing the usage of PyMiluvs API through examples.

`pymilvus/`: Contains PyMilvus source codes.

`tests/`: Contains unit tests.

`CONTRIBUTING.md`: Contributing guidelines.

`CONTRIBUTING_CN.md`: Contributing guidelines in Chinese.

`LICENSE`: Open Source License that PyMilvus follows.

`Makefile`: Scripts for Github action.

`OWNERS`: This file designates reviewers and approvers for the current directory. They are chosen according to their participation and code contribution. Active contributors are listed as reviewers, responsible for code reviews. Reviewers who have been active and reviewing codes for a period of time are listed as approvers. They are in charge of reviewing content apart from codes. If you submit a PR and do not know who can help you review the code, you can assign reviewers and approvers from this file to review your PR.

`README.md`: Readme.

`requirements.txt`: Dependencies for developing PyMilvus.

`setup.py`: Package script for PyMilvus.

## Congratulations! You are now the contributor to the Milvus community!

Apart from dealing with codes and machines, you are always welcome to communicate with any member from the Milvus community. New faces join us every day, and they may as well encounter the same challenges as you faced beore. Feel free to help them. You can pass on the collaborative spirit from the assistance you acquired when you first joined the community. Let us build a collaborative, open-source, exuberant, and tolerant community together!

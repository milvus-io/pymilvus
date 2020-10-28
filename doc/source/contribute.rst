============
Contributing
============

- `Open Issues`_
- `Submit Pull Requests`_
- `Github workflow`_
- `Contribution Guideline`_

Contributing is warmly welcomed. You can contribute to PyMilvus project by opening issues and submitting pull
requests on `PyMilvus Github page <https://github.com/milvus-io/pymilvus>`_.

Open Issues
===========
To request a new feature, report a bug or ask a question, it's recommended for you to **open an issue**.

For a feature
    you can tell us why you need it and we will decide whether to implement it soon.
    If we think it's a good improvement, we will make it a feature request and start to work on it. It's
    also welcomed for you to open an issue with your PR as a solution.

For a bug
    you need to tell us as much information as possible, better start with our
    `bug report template <https://github.com/milvus-io/pymilvus/issues/new?assignees=&labels=&template=bug_report.md&title=%5BBUG%5D>`_.
    With information, we can reproduce the bug easily and solve it later.

For a question
    it's welcomed to ask any questions about PyMilvus and Milvus, we are pleased to communicate with you.

Submit Pull Requests
====================

If you have improvements to PyMilvus, please submit pull requests(PR) to master, see workflow below.

**PR for codes**, you need to tell us why we need it, mentioning an existing issue would be better.

**PR for docs**, you also need to tell us why we need it.

Your PRs will be reviewed and checked, merged into our project if approved.

Github workflow
===============

This is a brief instruction of Github workflow for beginners.

* **Fork** the `PyMilvus repository <https://github.com/milvus-io/pymilvus>`_ on Github.

* **Clone** your fork to your local machine with ``git clone git@github.com:<your_user_name>/pymilvus.git``.

* Create a new branch with ``git checkout -b my_working_branch``.

* Make your changes, commit, then push to your forked repository.

* Visit Github and make you PR.

If you already have an existing local repository, always update it before you start to make changes like below:

.. code-block:: shell
   
   $ git remote add upstream git@github.com:milvus-io/pymilvus.git
   $ git checkout master
   $ git pull upstream master
   $ git checkout -b my_working_branch


Contribution guideline
======================

.. todo:
   More details aout tests and pylint check .

1. Add unit tests for your codes
2. Pass pylint check
3. For documentations
You need to enter the ``doc`` directory and run ``make html``, please refer to
`About this documentations <https://pymilvus.readthedocs.io/en/latest/about.html>`_.

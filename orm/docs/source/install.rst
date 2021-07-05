============
Installation
============

Installing via pip
==================

PyMilvus-ORM is in the `Python Package Index <https://pypi.org/project/pymilvus-orm/>`_.

PyMilvus-ORM only support python3(>= 3.6), usually, it's ok to install PyMilvus-ORM like below.

.. code-block:: shell
   
   $ python3 -m pip install pymilvus-orm==2.0.0rc1

Installing in a virtual environment
====================================

It's recommended to use PyMilvus-ORM in a virtual environment, using virtual environment allows you to avoid
installing Python packages globally which could break system tools or other projects.
We use ``virtualenv`` as an example to demonstrate how to install and using PyMilvus-ORM in a virtual environment.
See `virtualenv <https://virtualenv.pypa.io/en/latest/>`_ for more information about why and how.


.. code-block:: shell
   
   $ python3 -m pip install virtualenv
   $ virtualenv venv
   $ source venv/bin/activate
   (venv) $ pip install pymilvus-orm==2.0.0rc1

If you want to exit the virtualenv ``venv``, you can use ``deactivate``.


.. code-block:: shell
   
   (venv) $ deactivate
   $ 


Installing a specific PyMilvus-ORM version
======================================

Here we assume you are already in a virtual environment.

Suitable PyMilvus-ORM version depends on Milvus version you are using. See `install pymilvus-orm <https://github.com/milvus-io/pymilvus-orm#install-pymilvus-orm>`_ for recommended pymilvus-orm version.

If you want to install a specific version of PyMilvus-ORM:

.. code-block:: shell
   
   (venv) $ pip install pymilvus-orm==0.0.1

If you want to upgrade PyMilvus-ORM into the latest version published:

.. code-block:: shell
   
   (venv) $ pip install --upgrade pymilvus-orm


Installing from source
======================

This will install the latest PyMilvus-ORM into your virtual environment.

.. code-block:: shell
   
   (venv) $ pip install git+https://github.com/milvus-io/pymilvus-orm.git

Verifying installation
======================

Your installation is correct if the following command in the Python shell doesn't raise an exception.

.. code-block:: shell
   
   (venv) $ python -c "from pymilvus_orm import Milvus, DataType"


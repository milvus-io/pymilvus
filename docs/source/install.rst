============
Installation
============

Installing via pip
==================

PyMilvus is in the `Python Package Index <https://pypi.org/project/pymilvus/>`_.

PyMilvus only support python3(>= 3.6), usually, it's ok to install PyMilvus like below.

.. code-block:: shell
   
   $ python3 -m pip install pymilvus==2.0.0

Installing in a virtual environment
====================================

It's recommended to use PyMilvus in a virtual environment, using virtual environment allows you to avoid
installing Python packages globally which could break system tools or other projects.
We use ``virtualenv`` as an example to demonstrate how to install and using PyMilvus in a virtual environment.
See `virtualenv <https://virtualenv.pypa.io/en/latest/>`_ for more information about why and how.


.. code-block:: shell
   
   $ python3 -m pip install virtualenv
   $ virtualenv venv
   $ source venv/bin/activate
   (venv) $ pip install pymilvus==2.0.0

If you want to exit the virtualenv ``venv``, you can use ``deactivate``.


.. code-block:: shell
   
   (venv) $ deactivate
   $ 


Installing a specific PyMilvus version
======================================

Here we assume you are already in a virtual environment.

Suitable PyMilvus version depends on Milvus version you are using. See `install pymilvus <https://github.com/milvus-io/pymilvus#install-pymilvus>`_ for recommended pymilvus version.

If you want to install a specific version of PyMilvus:

.. code-block:: shell
   
   (venv) $ pip install pymilvus==2.0.0

If you want to upgrade PyMilvus into the latest version published:

.. code-block:: shell
   
   (venv) $ pip install --upgrade pymilvus


Installing from source
======================

This will install the latest PyMilvus into your virtual environment.

.. code-block:: shell
   
   (venv) $ pip install git+https://github.com/milvus-io/pymilvus.git

Verifying installation
======================

Your installation is correct if the following command in the Python shell doesn't raise an exception.

.. code-block:: shell
   
   (venv) $ python -c "from pymilvus import Collection"


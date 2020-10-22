====================
Installing/Upgrading
====================

Installing via pip
==================

PyMilvus is in `PYPI <https://pypi.org/project/pymilvus/>`_.

PyMilvus only suport python3(>= 3.5), usually it's ok to install PyMilvus like below.

.. code-block:: shell
   
   $ python3 -m pip install pymilvus

Installing in an virtual environment
====================================

It is recommended to use PyMilvus in an virtul environment, so modules and environment won't be conflicted.
We use ``virtualenv`` as an example to demenstrate how to install and using PyMilvus in an virtual environment.
See `virtualenv <https://virtualenv.pypa.io/en/latest/>`_ for more information about why and how.


.. code-block:: shell
   
   $ python3 -m pip install virtualenv
   $ virtualenv venv
   $ source venv/bin/activate
   (venv) $ pip install pymilvus

If you want to exit the virtualenv ``venv``, you can use ``deactivate``.


.. code-block:: shell
   
   (venv) $ deactivate
   $ 


Installing a spacific PyMilvus version
======================================

Here we assume you are already in a virtual environment.

Suitable PyMilvus version is depended on Milvus version your are using. See `install pymilvus <https://github.com/milvus-io/pymilvus#install-pymilvus>`_ for recommended pymilvus version.

If you want to install a spacific verison of PyMilvus:

.. code-block:: shell
   
   (venv) $ pip install pymilvus==0.3.0

If you want to upgrade PyMilvus into newest version:

.. code-block:: shell
   
   (venv) $ pip install --upgrade pymilvus

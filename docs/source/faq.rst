===
FAQ
===

- `I'm getting random "socket operation on non-socket" errors from gRPC.`_
- `How to fix the error when I install PyMilvus on Windows?`_



I'm getting random "socket operation on non-socket" errors from gRPC.
=====================================================================

Make sure to set the environment variable ``GRPC_ENABLE_FORK_SUPPORT=1``.
For reference, see `this post <https://zhuanlan.zhihu.com/p/136619485>`_.

How to fix the error when I install PyMilvus on Windows?
========================================================

Try installing PyMilvus in a Conda environment.

.. sectionauthor::
   `Yangxuan@milvus <https://github.com/XuanYang-cn>`_

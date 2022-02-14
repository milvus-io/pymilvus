===
FAQ
===
This page lists common issues that may occur when using Milvus, as well as possible troubleshooting tips.

What should I do if I got ``AttributeError: module 'google.protobuf.descriptor' has no attribute '_internal_create_key'``?
---------------------------------------------------------------------------------------------------------------------------
You should upgrade Protobuf with ``pip install --upgrade protobuf`` to version later than that PyMilvus required. You can check the version of Protobuf installed on on your device with ``pip3 show protobuf``.

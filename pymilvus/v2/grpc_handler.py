# Copyright (C) 2019-2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.


# TODO
def connect(address: str, timeout: float = None, **kwargs) -> None:
    """ Make a secure connection with Milvus.

    Support tls/ssl one-way and two-ways authentication.

    Args:
    Raises:
        MilvusException: if fail to connect to Milvus.
    """
    pass


# TODO
def insecured_connect(address: str, timeout: float = 10) -> None:
    """ Make an insecure connection with Milvus. """
    pass

# TODO
handler = None

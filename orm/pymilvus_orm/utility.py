# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.

from pymilvus_orm.connections import get_connection

def loading_progress(collection_name, partition_names=[], using="default"):
    """
    Show #loaded entities vs #total entities.

    :param collection_name: The name of collection to show
    :type  collection_name: str

    :param partition_names: The name of partition to show
    :type  partition_names: list 

    :return: Loading progress, contains num of loaded and num of total
    :rtype:  dict
    """
    if len(partition_names) == 0:
        return get_connection(using).load_collection_progress(collection_name)
    else:
        return get_connection(using).load_partitions_progress(collection_name, partition_names)


def wait_for_loading_complete(collection_name, partition_names=[], timeout=None, using="default"):
    """
    Block until loading is done or Raise Exception after timeout.

    :param collection_name: The name of collection to wait
    :type  collection_name: str

    :param partition_names: The name of partition to wait
    :type  partition_names: list

    :param timeout: The timeout for this method, unit: second
    :type  timeout: int
    """
    if len(partition_names) == 0:
        return get_connection(using).wait_for_loading_collection_complete(collection_name, timeout)
    else: 
        return get_connection(using).wait_for_loading_partitions_complete(collection_name, partition_names, timeout)


def index_building_progress(collection_name, index_name="", using="default"):
    """
    Show # indexed entities vs. # total entities.

    :param collection_name: The name of collection to show
    :type  collection_name: str

    :param index_name: The name of index to show
    :type  index_name: str

    :param timeout: The timeout for this method, unit: second
    :type  timeout: int

    :return: Building progress, contains num of indexed entities and num of total entities
    :rtype:  dict
    """
    return get_connection(using).get_index_build_progress(collection_name, index_name)


def wait_for_index_building_complete(collection_name, index_name="", timeout=None, using="default"):
    """
    Block until building is done or Raise Exception after timeout.

    :param collection_name: The name of collection to wait
    :type  collection_name: str

    :param index_name: The name of index to wait
    :type  index_name: str

    :param timeout: The timeout for this method, unit: second
    :type  timeout: int
    """
    return get_connection(using).wait_for_creating_index(collection_name, index_name, timeout)


def has_collection(collection_name, using="default"):
    """
    Checks whether a specified collection exists.

    :param collection_name: The name of collection to check.
    :type  collection_name: str

    :return: Whether the collection exists.
    :rtype:  bool
    """
    return get_connection(using).has_collection(collection_name)


def has_partition(collection_name, partition_name, using="default"):
    """
    Checks if a specified partition exists in a collection.

    :param collection_name: The collection name of partition to check
    :type  collection_name: str

    :param partition_name: The name of partition to check.
    :type  partition_name: str

    :return: Whether the partition exist.
    :rtype:  bool
    """
    return get_connection(using).has_partition(collection_name, partition_name)


def list_collections(timeout=None, using="default") -> list:
    """
    Returns a list of all collection names.

    :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                    is set to None, client waits until server response or error occur.
    :type  timeout: float

    :return: List of collection names, return when operation is successful
    :rtype: list[str]
    """
    return get_connection(using).list_collections()

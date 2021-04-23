def loading_progress(collection_name, partition_name="", using="default"):
    """
    Show #loaded entities vs #total entities.

    :param collection_name: The name of collection to show
    :type  collection_name: str

    :param partition_name: The name of partition to show
    :type  partition_name: str

    :return: Loading progress, contains num of loaded and num of total
    :rtype:  dict
    """
    pass


def wait_for_loading_complete(collection_name, partition_name="", timeout=None, using="default"):
    """
    Block until loading is done or Raise Exception after timeout.

    :param collection_name: The name of collection to wait
    :type  collection_name: str

    :param partition_name: The name of partition to wait
    :type  partition_name: str

    :param timeout: The timeout for this method, unit: second
    :type  timeout: int
    """
    pass


def index_building_progress(collection_name, index_name, timeout=None, using="default"):
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
    pass


def wait_for_index_building_complete(collection_name, index_name, timeout=None, using="default"):
    """
    Block until building is done or Raise Exception after timeout.

    :param collection_name: The name of collection to wait
    :type  collection_name: str

    :param index_name: The name of index to wait
    :type  index_name: str

    :param timeout: The timeout for this method, unit: second
    :type  timeout: int
    """
    pass


def has_collection(collection_name, using="default"):
    """
    Checks whether a specified collection exists.

    :param collection_name: The name of collection to check.
    :type  collection_name: str

    :return: Whether the collection exists.
    :rtype:  bool
    """
    pass


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
    pass


def list_collections(timeout=None, using="default"):
    """
    Returns a list of all collection names.

    :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                    is set to None, client waits until server response or error occur.
    :type  timeout: float

    :return: List of collection names, return when operation is successful
    :rtype: list[str]
    """
    pass

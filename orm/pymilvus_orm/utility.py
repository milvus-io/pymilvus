def loading_progress(collection_name, partition_name=""):
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


def wait_for_loading_complete(collection_name, partition_name="", timeout=None):
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


def index_building_progress(collection_name, index_name, timeout=None):
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


def wait_for_index_building_complete(collection_name, index_name, timeout=None):
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


def has_collection(collection_name):
    """
    Checks whether a specified collection exists.

    :param collection_name: The name of collection to check.
    :type  collection_name: str

    :return: Whether the collection exists.
    :rtype:  bool
    """
    pass


def has_partition(collection_name, partition_name):
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

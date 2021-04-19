


def loading_progress(collection_name, partition_name=None):
    """
    Show #loaded entities vs #total entities.

    :param collection_name: The name of collection to show.
    :type  collection_name: str

    :param partition_name: The name of partition to show.
    :type  partition_name: str

    :return: Number of loaded entities and total entites
    :rtype:  (int, int)
    """
    pass


def wait_for_loading_complete(collection_name, partition_name=None):
    """
    Block until loading is done or Raise Exception after timeout.

    :param collection_name: The name of collection to wait
    :type  collection_name: str

    :param partition_name: The name of partition to wait
    :type  partition_name: str
    """
    pass


def index_building_progress(collection_name, index_name=None):
    """
    Show # indexed entities vs. # total entities.

    :param collection_name: The name of collection to show.
    :type  collection_name: str

    :param index_name: The name of index to show.
    :type  index_name: str

    :return: Number of indexed entities and total entites
    :rtype:  (int, int)
    """
    pass


def wait_for_index_building_complete(collection_name, index_name=None):
    """
    Block until building is done or Raise Exception after timeout.

    :param collection_name: The name of collection to wait
    :type  collection_name: str

    :param index_name: The name of index to wait.
    :type  index_name: str
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


def has_partition(collecton_name, partition_name):
    """
    Checks if a specified partition exists in a collection.

    :param collecton_name: The collection name of partition to check
    :type  collecton_name: str

    :param partition_name: The name of partition to check.
    :type  partition_name: str

    :return: Whether the partition exist.
    :rtype:  bool
    """
    pass
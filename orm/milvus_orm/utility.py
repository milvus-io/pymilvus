


def loading_progress(cls, collection_name, partition_name=None, using=None):
    pass


def wait_for_loading_complete(cls, collection_name, partition_name=None, using=None):
    pass


def index_building_progress(cls, collection_name, index_name=None, using=None):
    pass


def wait_for_index_building_complete(cls, collection_name, index_name=None, using=None):
    pass


def has_collection(cls, collection_name, using=None):
    pass


def has_partition(cls, collecton_name, partition_name, using=None):
    pass
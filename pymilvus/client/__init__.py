"""client module"""
from pkg_resources import get_distribution, DistributionNotFound

__version__ = '0.0.0.dev'

try:
    __version__ = get_distribution('pymilvus').version
except DistributionNotFound:
    # package is not installed
    pass

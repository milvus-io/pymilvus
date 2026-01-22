import logging
import sys
from os.path import abspath, dirname
from pathlib import Path

logging.getLogger("faker").setLevel(logging.WARNING)
sys.path.append(Path(__file__).absolute().parent.parent)

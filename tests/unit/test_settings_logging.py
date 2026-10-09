import logging
import os
import subprocess
import sys
from pathlib import Path

from pymilvus.settings import init_log

PYMILVUS_LOGGERS = ("pymilvus", "pymilvus.milvus_client", "pymilvus.bulk_writer")


def test_import_does_not_close_existing_handlers():
    # pymilvus is already imported in this process, so check in a fresh interpreter.
    code = """
import logging
import sys

class Probe(logging.Handler):
    closed = False

    def close(self):
        Probe.closed = True
        super().close()

    def emit(self, record):
        sys.stdout.write("delivered\\n")

log = logging.getLogger("host_app")
log.setLevel(logging.INFO)
log.addHandler(Probe())

import pymilvus.settings  # noqa: F401

log.info("after import")
print("closed" if Probe.closed else "open")
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[2])
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.split() == ["delivered", "open"]


def test_pymilvus_loggers_are_configured():
    for name in PYMILVUS_LOGGERS:
        logger = logging.getLogger(name)
        assert logger.propagate is False
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    assert logging.getLogger("pymilvus").level == logging.WARNING
    assert logging.getLogger("pymilvus.milvus_client").level == logging.INFO
    assert logging.getLogger("pymilvus.bulk_writer").level == logging.INFO


def test_init_log_is_idempotent():
    init_log("WARNING")
    init_log("WARNING")
    for name in PYMILVUS_LOGGERS:
        assert len(logging.getLogger(name).handlers) == 1

"""Regression tests for pymilvus logging setup (issue #3794).

``import pymilvus`` used to run ``logging.config.dictConfig()`` which closes
every handler that already exists in the process. Logging configured by the
application before the import must survive untouched.
"""

import io
import logging
import logging.handlers
import subprocess
import sys
from pathlib import Path

import pytest
from pymilvus import settings

PYMILVUS_LOGGERS = ("pymilvus", "pymilvus.milvus_client", "pymilvus.bulk_writer")
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def app_handlers(tmp_path):
    """Handlers an application attached to the root logger before importing pymilvus."""
    buf = io.StringIO()
    stream_handler = logging.StreamHandler(buf)
    file_handler = logging.FileHandler(tmp_path / "app.log")
    memory_handler = logging.handlers.MemoryHandler(capacity=100, target=stream_handler)
    root = logging.getLogger()
    handlers = (stream_handler, file_handler, memory_handler)
    for handler in handlers:
        root.addHandler(handler)
    try:
        yield buf, stream_handler, file_handler, memory_handler
    finally:
        for handler in handlers:
            root.removeHandler(handler)
            handler.close()


def test_init_log_keeps_application_handlers_open(app_handlers):
    buf, stream_handler, file_handler, memory_handler = app_handlers

    settings.init_log("WARNING")

    assert file_handler.stream is not None
    assert not file_handler.stream.closed
    assert memory_handler.target is stream_handler
    root = logging.getLogger()
    assert stream_handler in root.handlers
    assert file_handler in root.handlers
    assert memory_handler in root.handlers

    logging.getLogger("myapp").warning("still logged")
    assert "still logged" in buf.getvalue()
    assert "still logged" in Path(file_handler.baseFilename).read_text()


def test_init_log_keeps_handler_attached_to_pymilvus_logger():
    buf = io.StringIO()
    app_handler = logging.StreamHandler(buf)
    logger = logging.getLogger("pymilvus")
    logger.addHandler(app_handler)
    try:
        settings.init_log("WARNING")

        assert app_handler in logger.handlers
        logger.warning("routed to app")
        assert "routed to app" in buf.getvalue()
    finally:
        logger.removeHandler(app_handler)


def test_init_log_is_idempotent():
    settings.init_log("WARNING")
    settings.init_log("WARNING")

    for name in PYMILVUS_LOGGERS:
        logger = logging.getLogger(name)
        pymilvus_handlers = [
            h for h in logger.handlers if h.get_name() == settings.LOG_HANDLER_NAME
        ]
        assert len(pymilvus_handlers) == 1
        assert not logger.propagate
        assert isinstance(pymilvus_handlers[0], logging.StreamHandler)
        assert pymilvus_handlers[0].formatter._fmt == settings.LOG_FORMAT

    assert logging.getLogger("pymilvus").level == logging.WARNING
    assert logging.getLogger("pymilvus.milvus_client").level == logging.INFO
    assert logging.getLogger("pymilvus.bulk_writer").level == logging.INFO


def test_import_pymilvus_keeps_preconfigured_file_handler(tmp_path):
    """End-to-end: ``import pymilvus`` in a fresh interpreter must not close app handlers."""
    log_file = tmp_path / "app.log"
    script = f"""
import logging, sys
handler = logging.FileHandler({str(log_file)!r})
logging.getLogger().addHandler(handler)
import pymilvus
sys.exit(0 if handler.stream is not None and not handler.stream.closed else 1)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

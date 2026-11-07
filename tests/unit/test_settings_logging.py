"""Regression tests for pymilvus logging setup (issue #3794).

``import pymilvus`` used to run ``logging.config.dictConfig()`` which closes
every handler that already exists in the process. Logging configured by the
application before the import must survive untouched.
"""

import contextlib
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


@contextlib.contextmanager
def only_pymilvus_handlers():
    """Run with only pymilvus' own handlers on the pymilvus loggers, then restore them.

    pytest attaches its log-capture handlers to every non-propagating logger
    once a test starts running, and :func:`settings.init_log` would otherwise
    treat them as application handlers.
    """
    saved = {name: list(logging.getLogger(name).handlers) for name in PYMILVUS_LOGGERS}
    for name, handlers in saved.items():
        logging.getLogger(name).handlers[:] = [
            h for h in handlers if h.get_name() == settings.LOG_HANDLER_NAME
        ]
    try:
        yield
    finally:
        for name, handlers in saved.items():
            logging.getLogger(name).handlers[:] = handlers


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
    with only_pymilvus_handlers():
        logger.addHandler(app_handler)
        settings.init_log("WARNING")

        assert app_handler in logger.handlers
        logger.warning("routed to app")
        assert "routed to app" in buf.getvalue()


def test_init_log_does_not_double_emit_with_application_handler(capsys):
    buf = io.StringIO()
    app_handler = logging.StreamHandler(buf)
    logger = logging.getLogger("pymilvus")
    with only_pymilvus_handlers():
        # Recreate the console handler so that it writes to the captured stderr.
        for name in PYMILVUS_LOGGERS:
            logging.getLogger(name).handlers.clear()
        settings.init_log("WARNING")
        logger.addHandler(app_handler)
        settings.init_log("WARNING")

        assert [h.get_name() for h in logger.handlers] == [app_handler.get_name()]
        logger.warning("only once")
        assert buf.getvalue().count("only once") == 1
        assert "only once" not in capsys.readouterr().err


def test_init_log_is_idempotent():
    with only_pymilvus_handlers():
        settings.init_log("WARNING")
        settings.init_log("WARNING")

        for name in PYMILVUS_LOGGERS:
            logger = logging.getLogger(name)
            assert len(logger.handlers) == 1
            handler = logger.handlers[0]
            assert handler.get_name() == settings.LOG_HANDLER_NAME
            assert isinstance(handler, logging.StreamHandler)
            assert not logger.propagate
            record = logging.LogRecord(name, logging.WARNING, __file__, 1, "msg", None, None)
            assert handler.format(record) == logging.Formatter(settings.LOG_FORMAT).format(record)

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
    result = _run_in_fresh_interpreter(script)
    assert result.returncode == 0, result.stderr


def test_import_pymilvus_keeps_write_mode_file_handler_logging(tmp_path):
    """A ``mode="w"`` FileHandler is never reopened once closed, so records after the import were lost."""
    log_file = tmp_path / "app.log"
    script = f"""
import logging
handler = logging.FileHandler({str(log_file)!r}, mode="w")
root = logging.getLogger()
root.addHandler(handler)
root.setLevel(logging.INFO)
root.info("before import")
import pymilvus
root.info("after import")
"""
    result = _run_in_fresh_interpreter(script)
    assert result.returncode == 0, result.stderr

    content = log_file.read_text()
    assert "before import" in content
    assert "after import" in content


def test_import_pymilvus_keeps_buffered_records_flushed_at_exit(tmp_path):
    """Records still buffered at interpreter exit must be flushed by logging's atexit shutdown."""
    log_file = tmp_path / "app.log"
    script = f"""
import logging, logging.handlers
target = logging.FileHandler({str(log_file)!r})
buffered = logging.handlers.MemoryHandler(
    capacity=1000, flushLevel=logging.CRITICAL, target=target, flushOnClose=True
)
root = logging.getLogger()
root.addHandler(buffered)
root.setLevel(logging.INFO)
root.info("buffered before import")
import pymilvus
root.info("buffered after import")
"""
    result = _run_in_fresh_interpreter(script)
    assert result.returncode == 0, result.stderr

    content = log_file.read_text()
    assert "buffered before import" in content
    assert "buffered after import" in content


def _run_in_fresh_interpreter(script):
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

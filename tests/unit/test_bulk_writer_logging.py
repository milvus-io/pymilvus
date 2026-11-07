"""Importing ``pymilvus.bulk_writer`` must not configure logging for the application.

The bulk writer modules used to call ``logging.basicConfig()`` at import time,
which attached a handler to the root logger and set its level to INFO. A
``logging.basicConfig()`` call made by the application afterwards was then
silently ignored.

Each check runs in a fresh interpreter so that the import really happens.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

IMPORT_BULK_WRITER = """
import pymilvus.bulk_writer
import pymilvus.bulk_writer.bulk_writer
import pymilvus.bulk_writer.endpoint_resolver
import pymilvus.bulk_writer.volume_file_manager
import pymilvus.bulk_writer.volume_manager
"""


def run_python(*parts: str) -> subprocess.CompletedProcess:
    script = "\n".join(textwrap.dedent(part) for part in parts)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_import_leaves_root_logger_unchanged():
    result = run_python(
        """
        import logging
        root = logging.getLogger()
        before = (root.level, list(root.handlers))
        """,
        IMPORT_BULK_WRITER,
        """
        after = (root.level, list(root.handlers))
        print(repr(before))
        print(repr(after))
        """,
    )
    assert result.returncode == 0, result.stderr
    before, after = result.stdout.splitlines()
    assert after == before


def test_application_basic_config_after_import_takes_effect():
    result = run_python(
        IMPORT_BULK_WRITER,
        """
        import logging
        import sys
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format="app: %(message)s")
        logging.getLogger("myapp").debug("hello")
        """,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "app: hello\n"


def test_bulk_writer_loggers_are_under_pymilvus_namespace():
    result = run_python(
        IMPORT_BULK_WRITER,
        """
        import json
        import sys
        loggers = {
            name: [module.logger.name, module.logger.getEffectiveLevel()]
            for name, module in sys.modules.items()
            if name.startswith("pymilvus.bulk_writer.") and hasattr(module, "logger")
        }
        print(json.dumps(loggers))
        """,
    )
    assert result.returncode == 0, result.stderr
    loggers = json.loads(result.stdout)
    assert "pymilvus.bulk_writer.bulk_writer" in loggers
    assert "pymilvus.bulk_writer.endpoint_resolver" in loggers
    for module_name, (logger_name, level) in loggers.items():
        assert logger_name == module_name
        # INFO, inherited from the "pymilvus.bulk_writer" logger set up in settings.py
        assert level == 20, module_name

"""Test that importing pymilvus does not call load_dotenv() (#3666).

Libraries should not call load_dotenv() at import time because it injects
all .env keys into os.environ for the entire process, breaking test isolation.
"""

import importlib
import os
from pathlib import Path

import pymilvus.settings as settings_module


def test_settings_does_not_call_load_dotenv():
    """Verify that pymilvus.settings no longer imports or calls load_dotenv."""
    source_file = Path(settings_module.__file__)
    source = source_file.read_text()
    assert "load_dotenv" not in source, (
        "pymilvus.settings should not call load_dotenv() — "
        "libraries must not load .env files at import time (see #3666)"
    )
    assert "from dotenv" not in source, (
        "pymilvus.settings should not import from dotenv (see #3666)"
    )


def test_importing_pymilvus_does_not_load_env_file(tmp_path, monkeypatch):
    """Verify that importing pymilvus with a .env file present does not
    pollute os.environ."""
    env_file = tmp_path / ".env"
    test_key = "PYMILVUS_TEST_DOTENV_SHOULD_NOT_LOAD"
    test_value = "this_should_not_be_in_os_environ"
    env_file.write_text(f"{test_key}={test_value}\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(test_key, raising=False)

    importlib.reload(settings_module)

    assert test_key not in os.environ, (
        f"Importing pymilvus should not load .env files into os.environ — "
        f"{test_key} was set by load_dotenv() (see #3666)"
    )

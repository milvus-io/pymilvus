import os
import sys
from importlib import reload

def test_settings_dotenv_no_pollution(tmp_path):
    old_uri = os.environ.get("MILVUS_URI")
    old_secret = os.environ.get("MY_APP_SECRET")
    old_disabled = os.environ.get("PYTHON_DOTENV_DISABLED")

    if "MILVUS_URI" in os.environ:
        del os.environ["MILVUS_URI"]
    if "MY_APP_SECRET" in os.environ:
        del os.environ["MY_APP_SECRET"]
    if "PYTHON_DOTENV_DISABLED" in os.environ:
        del os.environ["PYTHON_DOTENV_DISABLED"]

    orig_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        with open(".env", "w") as f:
            f.write("MILVUS_URI=http://localhost:19530\n")
            f.write("MY_APP_SECRET=sensitive_value\n")

        import pymilvus.settings
        reload(pymilvus.settings)

        assert os.environ.get("MILVUS_URI") == "http://localhost:19530"
        assert os.environ.get("MY_APP_SECRET") is None

    finally:
        os.chdir(orig_cwd)
        if old_uri is not None:
            os.environ["MILVUS_URI"] = old_uri
        elif "MILVUS_URI" in os.environ:
            del os.environ["MILVUS_URI"]
        if old_secret is not None:
            os.environ["MY_APP_SECRET"] = old_secret
        elif "MY_APP_SECRET" in os.environ:
            del os.environ["MY_APP_SECRET"]
        if old_disabled is not None:
            os.environ["PYTHON_DOTENV_DISABLED"] = old_disabled
        elif "PYTHON_DOTENV_DISABLED" in os.environ:
            del os.environ["PYTHON_DOTENV_DISABLED"]


def test_settings_dotenv_disabled(tmp_path):
    old_uri = os.environ.get("MILVUS_URI")
    old_disabled = os.environ.get("PYTHON_DOTENV_DISABLED")

    if "MILVUS_URI" in os.environ:
        del os.environ["MILVUS_URI"]
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"

    orig_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        with open(".env", "w") as f:
            f.write("MILVUS_URI=http://localhost:19530\n")

        import pymilvus.settings
        reload(pymilvus.settings)

        # Should NOT load because dotenv is disabled
        assert os.environ.get("MILVUS_URI") is None

    finally:
        os.chdir(orig_cwd)
        if old_uri is not None:
            os.environ["MILVUS_URI"] = old_uri
        elif "MILVUS_URI" in os.environ:
            del os.environ["MILVUS_URI"]
        if old_disabled is not None:
            os.environ["PYTHON_DOTENV_DISABLED"] = old_disabled
        elif "PYTHON_DOTENV_DISABLED" in os.environ:
            del os.environ["PYTHON_DOTENV_DISABLED"]


def test_settings_dotenv_valueless_keys_skipped(tmp_path):
    old_uri = os.environ.get("MILVUS_URI")

    if "MILVUS_URI" in os.environ:
        del os.environ["MILVUS_URI"]

    orig_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        with open(".env", "w") as f:
            # Valueless key in .env syntax
            f.write("MILVUS_URI\n")

        # This should not raise TypeError during load
        import pymilvus.settings
        reload(pymilvus.settings)

        assert os.environ.get("MILVUS_URI") is None

    finally:
        os.chdir(orig_cwd)
        if old_uri is not None:
            os.environ["MILVUS_URI"] = old_uri
        elif "MILVUS_URI" in os.environ:
            del os.environ["MILVUS_URI"]

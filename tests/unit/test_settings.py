import os
import subprocess
import sys
from pathlib import Path


def test_import_does_not_load_dotenv_into_process_environment(tmp_path):
    (tmp_path / ".env").write_text("PYMILVUS_IMPORT_SIDE_EFFECT=unexpected\n")
    environment = os.environ.copy()
    environment.pop("PYMILVUS_IMPORT_SIDE_EFFECT", None)
    environment["PYTHONPATH"] = str(Path(__file__).parents[2])

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; import pymilvus; print(os.getenv('PYMILVUS_IMPORT_SIDE_EFFECT'))",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "None"


def test_import_reads_configuration_from_process_environment(tmp_path):
    environment = os.environ.copy()
    environment.update(
        {
            "MILVUS_URI": "http://configured.example",
            "MILVUS_CONN_ALIAS": "configured",
            "MILVUS_CONN_TIMEOUT": "2.5",
            "PYTHONPATH": str(Path(__file__).parents[2]),
        }
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pymilvus.settings import Config; print(Config.MILVUS_URI, Config.MILVUS_CONN_ALIAS, Config.MILVUS_CONN_TIMEOUT)",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "http://configured.example configured 2.5"

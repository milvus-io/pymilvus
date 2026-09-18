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
            "import os; import pymilvus.settings; print(os.getenv('PYMILVUS_IMPORT_SIDE_EFFECT'))",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "None"

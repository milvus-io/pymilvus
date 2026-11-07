"""Isolated scale check: large manifests must spill instead of retaining Python objects."""

import os
import subprocess
import sys
from pathlib import Path


def test_twenty_thousand_path_index_has_bounded_python_heap(tmp_path):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).absolute()), str(tmp_path)],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).absolute().parents[2])},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS entries=20000" in result.stdout
    assert not list(tmp_path.iterdir())


if __name__ == "__main__":
    import tracemalloc

    from pymilvus.bulk_writer._upload_manifest import _UploadManifest
    from pymilvus.bulk_writer.upload_policy import UploadPolicy

    tracemalloc.start()
    with _UploadManifest(sys.argv[1]) as manifest:
        for i in range(20_000):
            manifest.add(f"nested/shard-{i % 1000}/file-{i}", 3, 0)
        assert manifest.storage == "sqlite"
        for i in range(0, 20_000, 2):
            manifest.match(
                f"nested/shard-{i % 1000}/file-{i}", 3, 0, UploadPolicy.SKIP_IF_SAME_SIZE
            )
        assert manifest.remaining_count == 10_000
        assert manifest.remaining_bytes == 30_000
        assert sum(1 for _ in manifest.entries()) == 10_000
        _, peak = tracemalloc.get_traced_memory()
        assert peak < 16 * 1024 * 1024, peak

    sys.stdout.write(f"PASS entries=20000 peakPythonIndexBytes={peak}\n")

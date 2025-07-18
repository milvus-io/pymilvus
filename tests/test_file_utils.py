import tempfile
from pathlib import Path

from pymilvus.stage.file_utils import FileUtils


def create_temp_file(path: Path, size_in_bytes: int = 10):
    path.write_bytes(b'x' * size_in_bytes)
    return path


def test_process_local_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "test.txt"
        create_temp_file(tmp_file, size_in_bytes=100)

        files, total_size = FileUtils.process_local_path(str(tmp_file))
        assert len(files) == 1
        assert Path(files[0]).resolve() == tmp_file.resolve()
        assert total_size == 100


def test_process_local_directory():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        file1 = create_temp_file(tmp_path / "file1.txt", 50)
        file2 = create_temp_file(tmp_path / "file2.txt", 70)

        subdir = tmp_path / "subdir"
        subdir.mkdir()
        file3 = create_temp_file(subdir / "file3.txt", 80)

        files, total_size = FileUtils.process_local_path(str(tmp_path))
        assert len(files) == 3
        assert total_size == 200
        resolved_files = [Path(f).resolve() for f in files]
        assert all(f.exists() for f in resolved_files)


def test_process_invalid_path():
    try:
        FileUtils.process_local_path("/path/does/not/exist")
    except ValueError as e:
        assert "Path does not exist" in str(e)
    else:
        assert False, "Expected ValueError"

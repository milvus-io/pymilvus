"""Volume planning, LIST-only policies, bounded resources and retry regression tests."""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from threading import Barrier, Event
from unittest.mock import Mock
from urllib.parse import parse_qs, quote, quote_plus, urlsplit

import pytest
import requests
from minio import Minio
from minio.error import S3Error, ServerError
from pymilvus.bulk_writer import UploadPolicy, VolumeFileManager
from pymilvus.bulk_writer._upload_executor import run_bounded
from pymilvus.bulk_writer._upload_manifest import _UploadManifest
from pymilvus.bulk_writer._volume_listing import list_objects_page
from pymilvus.bulk_writer.volume_file_manager import (
    _UploadProgressCallbackError,
    _UploadProgressTracker,
    _VolumeUploadContext,
)
from pymilvus.bulk_writer.volume_restful import apply_volume
from pymilvus.exceptions import MilvusException
from urllib3 import PoolManager
from urllib3.response import HTTPResponse


def s3_error(code, status=400):
    return S3Error(
        response=HTTPResponse(status=status),
        code=code,
        message="test error",
        resource="resource",
        request_id="request",
        host_id="host",
    )


class FakeClient:
    def __init__(self, manager):
        self.manager = manager
        self._http = Mock()
        self.identity = len(manager.clients)

    def fput_object(self, **kwargs):
        m = self.manager
        assert not self._http.clear.called, "Client must remain open while in use"
        m.put_calls.append((self.identity, kwargs["object_name"]))
        if m.put_hook:
            m.put_hook(self, kwargs)
        path = Path(kwargs["file_path"])
        m.objects[kwargs["object_name"]] = (path.stat().st_size, path.stat().st_mtime_ns)
        if m.lose_response:
            m.lose_response = False
            raise TimeoutError("response lost after commit")
        if "progress" in kwargs:
            kwargs["progress"].update(path.stat().st_size)


class FakeManager(VolumeFileManager):
    def __init__(self):
        super().__init__("https://example.com", "test-key", "volume")
        self.objects = {}
        self.clients = []
        self.list_calls = []
        self.put_calls = []
        self.max_bytes = None
        self.max_files = None
        self.list_hook = self.put_hook = None
        self.lose_response = False

    def _create_volume_state(self, path):
        info = {
            "volumePrefix": "prefix/",
            "volumeName": "volume",
            "bucketName": "bucket",
            "credentials": {"expireTime": "2099-01-01T00:00:00Z"},
            "condition": {"maxContentLength": self.max_bytes, "maxFileNumber": self.max_files},
        }
        client = FakeClient(self)
        self.clients.append(client)
        return info, client

    def page(self, client, bucket, prefix, token):
        assert bucket == "bucket"
        assert not client._http.clear.called
        self.list_calls.append((client.identity, prefix, token))
        if self.list_hook:
            self.list_hook(client, prefix, token)
        keys = sorted(k for k in self.objects if k.startswith(prefix))
        start = int(token or 0)
        page = [(k, *self.objects[k]) for k in keys[start : start + 1000]]
        end = start + len(page)
        return page, str(end) if end < len(keys) else None


@pytest.fixture
def manager(monkeypatch):
    m = FakeManager()
    monkeypatch.setattr("pymilvus.bulk_writer.volume_file_manager.list_objects_page", m.page)
    return m


def write(root, name, size=3):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


@pytest.mark.parametrize(
    "policy,puts,lists",
    [
        (UploadPolicy.OVERWRITE, 1, 0),
        (UploadPolicy.SKIP_IF_EXISTS, 0, 1),
        (UploadPolicy.SKIP_IF_SAME_SIZE, 0, 1),
        (UploadPolicy.SIZE_AND_MTIME, 0, 1),
    ],
)
def test_policies_and_final_progress(manager, tmp_path, policy, puts, lists):
    source = write(tmp_path, "data", 3)
    manager.objects["prefix/data/data"] = (3, source.stat().st_mtime_ns)
    progress = []
    result = manager.upload_file_to_volume(
        str(source), "data", upload_policy=policy, progress_callback=progress.append
    )
    assert result["path"] == "data/"
    assert len(manager.put_calls) == puts
    assert len(manager.list_calls) == lists
    assert progress[-1].percent == 100
    assert progress[-1].total_files == puts
    assert all(c._http.clear.call_count == 1 for c in manager.clients)


def test_nested_paths_exact_matches_and_remaining_quota(manager, tmp_path):
    write(tmp_path, "a/b/same", 4)
    write(tmp_path, "other/same", 2)
    manager.objects.update({"prefix/data/a/b/same": (4, 0), "prefix/data/other/same-more": (2, 0)})
    manager.max_bytes, manager.max_files = 2, 1
    manager.upload_file_to_volume(str(tmp_path), "data")
    assert manager.put_calls == [(0, "prefix/data/other/same")]
    assert manager.list_calls == [(0, "prefix/data/", None)]


def test_early_stop_with_size_mismatch_and_zero_bytes(manager, tmp_path):
    write(tmp_path, "a", 3)
    write(tmp_path, "b", 0)
    manager.objects = {"prefix/data/a": (2, 0), "prefix/data/b": (0, 0)}
    manager.objects.update({f"prefix/data/z{i:05d}": (0, 0) for i in range(5000)})
    manager.upload_file_to_volume(str(tmp_path), "data")
    assert len(manager.list_calls) == 1
    assert manager.put_calls == [(0, "prefix/data/a")]


def test_page_retry_preserves_token_and_refreshes_expired_token(manager, tmp_path):
    write(tmp_path, "z", 3)
    manager.objects = {f"prefix/data/a{i:04d}": (0, 0) for i in range(1000)}
    manager.objects["prefix/data/z"] = (3, 0)

    def fail(client, prefix, token):
        if client.identity == 0 and token == "1000":
            raise s3_error("ExpiredToken")

    manager.list_hook = fail
    manager.upload_file_to_volume(str(tmp_path), "data", retry_interval=0)
    assert manager.list_calls == [
        (0, "prefix/data/", None),
        (0, "prefix/data/", "1000"),
        (1, "prefix/data/", "1000"),
    ]
    assert not manager.put_calls
    assert all(c._http.clear.call_count == 1 for c in manager.clients)


@pytest.mark.parametrize(
    "code", ["AccessDenied", "SignatureDoesNotMatch", "NoSuchBucket", "InvalidToken"]
)
def test_auth_and_bucket_errors_fail_without_retry_or_upload(manager, tmp_path, code):
    write(tmp_path, "data")
    manager.list_hook = Mock(side_effect=s3_error(code))
    with pytest.raises(S3Error):
        manager.upload_file_to_volume(str(tmp_path), "data", retry_interval=0)
    assert len(manager.list_calls) == len(manager.clients) == 1
    assert not manager.put_calls


@pytest.mark.parametrize("retries,attempts", [(0, 1), (2, 3)])
def test_expiration_retry_budget(manager, tmp_path, retries, attempts):
    write(tmp_path, "data")
    manager.put_hook = Mock(side_effect=s3_error("SecurityTokenExpired"))
    with pytest.raises(RuntimeError, match=f"after {attempts} attempts"):
        manager.upload_file_to_volume(
            str(tmp_path),
            "data",
            upload_policy=UploadPolicy.OVERWRITE,
            max_retries=retries,
            retry_interval=0,
        )
    assert len(manager.put_calls) == len(manager.clients) == attempts


@pytest.mark.parametrize(
    "policy,puts",
    [
        (UploadPolicy.OVERWRITE, 2),
        (UploadPolicy.SKIP_IF_EXISTS, 1),
        (UploadPolicy.SKIP_IF_SAME_SIZE, 1),
        (UploadPolicy.SIZE_AND_MTIME, 1),
    ],
)
def test_put_response_lost_honors_retry_policy(manager, tmp_path, policy, puts):
    write(tmp_path, "data")
    manager.lose_response = True
    manager.upload_file_to_volume(str(tmp_path), "data", upload_policy=policy, retry_interval=0)
    assert len(manager.put_calls) == puts
    if policy is UploadPolicy.OVERWRITE:
        assert not manager.list_calls
    else:
        assert manager.list_calls[-1][1] == "prefix/data/data"


def test_concurrent_expiration_refresh_is_coalesced(manager, tmp_path):
    write(tmp_path, "a")
    write(tmp_path, "b")
    barrier = Barrier(2, timeout=5)

    def fail(client, kwargs):
        if client.identity == 0:
            barrier.wait()
            assert not client._http.clear.called
            raise s3_error("ExpiredToken")

    manager.put_hook = fail
    manager.upload_file_to_volume(
        str(tmp_path),
        "data",
        upload_concurrency=2,
        upload_policy=UploadPolicy.OVERWRITE,
        retry_interval=0,
    )
    assert len(manager.clients) == 2
    assert all(c._http.clear.call_count == 1 for c in manager.clients)


def test_session_close_waits_for_borrower():
    a, b = Mock(), Mock()
    context = _VolumeUploadContext({}, a, timedelta(seconds=1))
    with context.borrow():
        context.set_state({}, b)
        assert not a._http.clear.called
    assert a._http.clear.call_count == 1
    context.close()
    context.close()
    assert b._http.clear.call_count == 1


def test_symlinks_preserve_keys_and_cycles_fail_before_network(manager, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    external = write(tmp_path, "external/data")
    (source / "file").symlink_to(external)
    (source / "directory").symlink_to(external.parent, target_is_directory=True)
    alias = tmp_path / "root-alias"
    alias.symlink_to(source, target_is_directory=True)
    manager.upload_file_to_volume(str(alias), "data")
    assert set(manager.objects) == {"prefix/data/file", "prefix/data/directory/data"}
    manager.upload_file_to_volume(str(alias), "data")
    assert len(manager.put_calls) == 2
    (source / "cycle").symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="cycle"):
        manager.upload_file_to_volume(str(source), "data")
    assert len(manager.clients) == 2


def test_memory_sqlite_boundary_metadata_matching_and_cleanup(tmp_path):
    with _UploadManifest(str(tmp_path)) as manifest:
        for i in range(10_000):
            manifest.add(str(i), 3, 10)
        assert manifest.storage == "memory"
        manifest.add("overflow", 5, 20)
        assert manifest.storage == "sqlite"
        for i in range(10_000):
            manifest.match(str(i), 3, 10, UploadPolicy.SKIP_IF_SAME_SIZE)
        assert manifest.remaining_count == 1
        assert manifest.remaining_bytes == 5
        assert [(e.path, e.size, e.mtime_ns) for e in manifest.entries()] == [("overflow", 5, 20)]
    assert not list(tmp_path.iterdir())


def test_directory_queue_spills_and_symlink_aliases_survive_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(_UploadManifest, "DIRECTORY_LIMIT", 1)
    source = tmp_path / "source"
    source.mkdir()
    external = write(tmp_path, "external/data")
    (source / "a").symlink_to(external.parent, target_is_directory=True)
    (source / "b").symlink_to(external.parent, target_is_directory=True)
    with _UploadManifest(str(tmp_path)) as manifest:
        manifest.scan(source)
        assert manifest.storage == "sqlite"
        assert [e.path for e in manifest.entries()] == ["a/data", "b/data"]
    assert len(list(tmp_path.iterdir())) == 2


def test_progress_is_bounded_and_callback_does_not_hold_state_lock():
    entered, release = Event(), Event()
    events = []

    def callback(event):
        events.append(event)
        if len(events) == 1:
            entered.set()
            assert release.wait(5)

    tracker = _UploadProgressTracker(10_001, 10_001, callback)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(tracker.update_file, "first", 1, 1)
        try:
            assert entered.wait(5)
            pool.submit(tracker.finish_file, "second", 1).result(timeout=2)
        finally:
            release.set()
        first.result(timeout=2)
    tracker.finish_file("first", 1)
    for i in range(9999):
        tracker.finish_file(str(i), 1)
    tracker.finish_upload()
    assert tracker._file_progress == {}
    assert tracker._completed_files == 10_001
    assert len(events) == 2
    assert events[-1].percent == 100


def test_bounded_executor_stops_reading_on_failure_and_waits_for_peers():
    started, stopped = Event(), Event()
    reads = []
    stop = Event()

    def entries():
        for i in range(100):
            reads.append(i)
            yield i

    def action(value):
        if value == 0:
            assert started.wait(5)
            raise ValueError("failed")
        try:
            started.set()
            assert stop.wait(5)
        finally:
            stopped.set()

    with pytest.raises(ValueError, match="failed"):
        run_bounded(entries(), action, 2, stop)
    assert reads == [0, 1]
    assert stopped.is_set()


@pytest.mark.parametrize("encode_key", [quote, quote_plus])
def test_list_adapter_uses_flat_v2_pages_and_preserves_plus_and_tokens(encode_key):
    key = "prefix/中文 + %.txt"
    xml = f"""<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <EncodingType>url</EncodingType><IsTruncated>true</IsTruncated>
      <NextContinuationToken>opaque+token/==</NextContinuationToken>
      <Contents><Key>{encode_key(key)}</Key><Size>3</Size>
      <LastModified>2026-01-01T00:00:00.000Z</LastModified></Contents></ListBucketResult>"""
    client = Mock()
    response = HTTPResponse(body=xml.encode())
    client._execute.return_value = response
    page, token = list_objects_page(client, "bucket", "prefix/", "previous+token")
    assert page[0][0:2] == (key, 3)
    assert token == "opaque+token/=="
    client._execute.assert_called_once_with(
        "GET",
        "bucket",
        query_params={
            "list-type": "2",
            "max-keys": "1000",
            "prefix": "prefix/",
            "delimiter": "",
            "encoding-type": "url",
            "continuation-token": "previous+token",
        },
    )
    client.stat_object.assert_not_called()


@pytest.mark.parametrize(
    "remote_size,remote_mtime,skip",
    [
        (3, 10, True),
        (3, 11, True),
        (3, 9, False),
        (4, 11, False),
        (None, 11, False),
        (3, None, False),
    ],
)
def test_size_and_time_semantics(remote_size, remote_mtime, skip):
    remote_ns = remote_mtime * 1_000_000 if remote_mtime is not None else None
    assert UploadPolicy.SIZE_AND_MTIME.should_skip(3, 10_000_000, remote_size, remote_ns) is skip


@pytest.mark.parametrize(
    "body",
    [
        b"<ListBucketResult><IsTruncated>true</IsTruncated></ListBucketResult>",
        b"<ListBucketResult><IsTruncated>true</IsTruncated><NextContinuationToken>same</NextContinuationToken></ListBucketResult>",
        b'<!DOCTYPE ListBucketResult [<!ENTITY x "expanded">]><ListBucketResult>&x;</ListBucketResult>',
    ],
)
def test_invalid_listing_fails_and_releases_response(body):
    response = Mock(data=body)
    client = Mock()
    client._execute.return_value = response
    with pytest.raises(ValueError):
        list_objects_page(client, "bucket", "prefix/", "same")
    response.close.assert_called_once()
    response.release_conn.assert_called_once()


def test_listing_missing_metadata_does_not_skip_by_size(manager, tmp_path):
    write(tmp_path, "zero", 0)
    manager.objects["prefix/data/zero"] = (None, None)
    manager.upload_file_to_volume(str(tmp_path), "data")
    assert len(manager.put_calls) == 1


def test_changed_file_is_rejected_before_put(manager, tmp_path):
    source = write(tmp_path, "file")
    manager.list_hook = lambda *_: source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after upload planning"):
        manager.upload_file_to_volume(str(source), "data")
    assert not manager.put_calls


def test_failed_scan_cleans_sqlite_and_makes_no_cloud_requests(manager, tmp_path, monkeypatch):
    monkeypatch.setattr(_UploadManifest, "FILE_LIMIT", 0)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    source = tmp_path / "source"
    write(source, "file")
    (source / "cycle").symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="cycle"):
        manager.upload_file_to_volume(str(source), "data", temporary_directory=str(scratch))
    assert not manager.clients
    assert not list(scratch.iterdir())


def test_server_errors_and_callback_errors_have_distinct_retry_behavior():
    assert VolumeFileManager._retryable(ServerError("unavailable", 503))
    assert not VolumeFileManager._retryable(ServerError("forbidden", 403))
    assert not VolumeFileManager._retryable(_UploadProgressCallbackError("callback"))


@pytest.mark.parametrize("status,retry", [(503, True), (403, False)])
def test_legacy_minio_server_error_without_status_property(status, retry):
    error = ServerError.__new__(ServerError)
    Exception.__init__(error, f"server failed with HTTP status code {status}")
    assert VolumeFileManager._retryable(error) is retry


def test_list_adapter_through_real_minio_makes_one_bucket_get(monkeypatch):
    pool = PoolManager()
    request = Mock(
        return_value=HTTPResponse(
            body=b"<ListBucketResult><IsTruncated>false</IsTruncated></ListBucketResult>",
            status=200,
        )
    )
    monkeypatch.setattr(pool, "urlopen", request)
    client = Minio(
        "storage.example.com",
        access_key="test-access",
        secret_key="test-secret",
        region="us-east-1",
        http_client=pool,
    )
    try:
        assert list_objects_page(client, "bucket", "nested/中文+/", "opaque+token") == ([], None)
        request.assert_called_once()
        args, _kwargs = request.call_args
        assert args[0] == "GET"
        url = urlsplit(args[1])
        assert url.path == "/bucket"
        assert parse_qs(url.query, keep_blank_values=True) == {
            "list-type": ["2"],
            "max-keys": ["1000"],
            "prefix": ["nested/中文+/"],
            "delimiter": [""],
            "encoding-type": ["url"],
            "continuation-token": ["opaque+token"],
        }
    finally:
        pool.clear()


def test_retry_stops_listing_once_exact_key_with_different_size_is_found(manager, tmp_path):
    source = write(tmp_path, "file")
    manager.objects["prefix/data/file"] = (99, 0)
    for i in range(1001):
        manager.objects[f"prefix/data/file-suffix-{i}"] = (3, 0)

    def fail_first_put(client, _kwargs):
        if client.identity == 0:
            raise TimeoutError("transient")

    manager.put_hook = fail_first_put
    manager.upload_file_to_volume(str(source), "data", retry_interval=0)
    assert len(manager.list_calls) == 2  # One preflight page, one retry page.
    assert len(manager.put_calls) == 2


@pytest.mark.parametrize("code", ["Throttling", "ThrottlingException", "RequestLimitExceeded"])
def test_throttling_codes_refresh_and_retry_listing(manager, tmp_path, code):
    source = write(tmp_path, "file")

    def throttle(client, _prefix, _token):
        if client.identity == 0:
            raise s3_error(code)

    manager.list_hook = throttle
    manager.upload_file_to_volume(str(source), "data", max_retries=1, retry_interval=0)
    assert len(manager.list_calls) == 2
    assert len(manager.clients) == 2
    assert len(manager.put_calls) == 1


@pytest.mark.parametrize("status", [408, 429, 500, 501, 503, 599])
@pytest.mark.parametrize("error_type", ["s3", "server", "legacy"])
def test_retryable_http_statuses(status, error_type):
    if error_type == "s3":
        error = s3_error("UnknownVendorError", status)
    elif error_type == "server":
        error = ServerError("server unavailable", status)
    else:
        error = ServerError.__new__(ServerError)
        Exception.__init__(error, f"server failed with HTTP status code {status}")
    assert VolumeFileManager._retryable(error)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409])
def test_unrecognized_nontransient_http_errors_do_not_retry(status):
    assert not VolumeFileManager._retryable(s3_error("UnknownVendorError", status))
    assert not VolumeFileManager._retryable(ServerError("rejected", status))


def test_wrapped_errors_and_cycles_preserve_retry_classification():
    outer = RuntimeError("wrapped")
    outer.__cause__ = s3_error("Throttling")
    assert VolumeFileManager._retryable(outer)
    outer.__cause__ = s3_error("AccessDenied", 403)
    assert not VolumeFileManager._retryable(outer)
    outer.__cause__ = _UploadProgressCallbackError("callback")
    outer.__cause__.__cause__ = TimeoutError("callback's own timeout")
    assert not VolumeFileManager._retryable(outer)
    outer.__cause__ = ValueError("invalid argument")
    outer.__cause__.__cause__ = TimeoutError("wrapped network error")
    assert not VolumeFileManager._retryable(outer)
    outer.__cause__ = outer
    assert not VolumeFileManager._retryable(outer)
    outer.__cause__ = None
    outer.__context__ = TimeoutError("context")
    outer.__suppress_context__ = False
    assert VolumeFileManager._retryable(outer)
    outer.__suppress_context__ = True
    assert not VolumeFileManager._retryable(outer)


@pytest.mark.parametrize("retries", [0, 2])
def test_proactive_refresh_failure_obeys_retry_budget_and_cleans_up(
    manager, tmp_path, monkeypatch, retries
):
    source = write(tmp_path, "file")
    create = manager._create_volume_state
    calls = []

    def expire_then_fail(path):
        calls.append(path)
        if len(calls) > 1:
            raise TimeoutError("refresh timed out")
        info, client = create(path)
        info["credentials"]["expireTime"] = "2000-01-01T00:00:00Z"
        return info, client

    monkeypatch.setattr(manager, "_create_volume_state", expire_then_fail)
    with pytest.raises(RuntimeError, match=f"after {retries + 1} attempts"):
        manager.upload_file_to_volume(
            str(source),
            "data",
            max_retries=retries,
            retry_interval=0,
            upload_policy=UploadPolicy.OVERWRITE,
        )
    assert len(calls) == retries + 2  # Initial apply plus bounded refresh attempts.
    assert not manager.put_calls
    manager.clients[0]._http.clear.assert_called_once()


def test_proactive_refresh_recovers_before_first_put_without_extra_listing(
    manager, tmp_path, monkeypatch
):
    source = write(tmp_path, "file")
    create = manager._create_volume_state
    calls = []

    def refresh(path):
        calls.append(path)
        if len(calls) == 2:
            raise TimeoutError("refresh timed out")
        return create(path)

    def expire_after_listing(client, _prefix, _token):
        manager.volume_info["credentials"]["expireTime"] = "2000-01-01T00:00:00Z"

    manager.list_hook = expire_after_listing
    monkeypatch.setattr(manager, "_create_volume_state", refresh)
    manager.upload_file_to_volume(str(source), "data", max_retries=1, retry_interval=0)
    assert len(calls) == 3
    assert len(manager.put_calls) == 1
    assert len(manager.list_calls) == 1  # Refresh failed; no PUT was attempted yet.


def test_reactive_refresh_failure_stays_within_retry_budget(manager, tmp_path, monkeypatch):
    source = write(tmp_path, "file")
    create = manager._create_volume_state
    calls = []

    def fail_once_then_refresh(path):
        calls.append(path)
        if len(calls) == 2:
            raise TimeoutError("refresh timed out")
        return create(path)

    def fail_initial_request(client, _kwargs):
        if client.identity == 0:
            raise TimeoutError("storage request timed out")

    monkeypatch.setattr(manager, "_create_volume_state", fail_once_then_refresh)
    manager.put_hook = fail_initial_request
    manager.upload_file_to_volume(
        str(source),
        "data",
        max_retries=2,
        retry_interval=0,
        upload_policy=UploadPolicy.OVERWRITE,
    )
    assert len(calls) == 3  # Initial credentials, failed refresh, successful refresh.
    assert manager.put_calls == [(0, "prefix/data/file"), (1, "prefix/data/file")]


@pytest.mark.parametrize("status,retry", [(429, True), (503, True), (403, False)])
def test_apply_volume_http_status_preserves_retry_classification(monkeypatch, status, retry):
    response = Mock(status_code=status)
    monkeypatch.setattr(
        "pymilvus.bulk_writer.volume_restful.requests.post", Mock(return_value=response)
    )

    with pytest.raises(MilvusException) as caught:
        apply_volume("https://example.com", "key", "volume", "path")

    assert isinstance(caught.value.__cause__, requests.HTTPError)
    assert VolumeFileManager._retryable(caught.value) is retry


@pytest.mark.parametrize(
    "local_ns,remote_ns,skip",
    [
        (1_000_500_000, 1_000_000_000, True),
        (1_001_000_000, 1_000_999_999, False),
        (-1, -999_999, True),
        (0, -1, False),
    ],
)
def test_mtime_comparison_matches_java_millisecond_precision(local_ns, remote_ns, skip):
    assert UploadPolicy.SIZE_AND_MTIME.should_skip(3, local_ns, 3, remote_ns) is skip


def test_local_change_detection_retains_nanosecond_precision(tmp_path):
    source = write(tmp_path, "file")
    attrs = source.stat()
    with pytest.raises(ValueError, match="changed after upload planning"):
        VolumeFileManager._validate_source(source, attrs.st_size, attrs.st_mtime_ns + 1)


@pytest.mark.parametrize("target", ["..", "a/../b", "a/..", r"a\..\b", "/../a"])
def test_parent_path_segments_fail_before_cloud_requests(manager, tmp_path, target):
    source = write(tmp_path, "file")
    with pytest.raises(ValueError, match="must not escape"):
        manager.upload_file_to_volume(str(source), target)
    assert not manager.clients


@pytest.mark.parametrize(
    "target,expected", [("/a//./b/", "a/b/"), ("./", ""), (r"a\b", "a/b/"), ("a/..b", "a/..b/")]
)
def test_path_normalization_retains_supported_segments(manager, target, expected):
    assert manager._convert_dir_path(target) == expected


def test_preparation_logs_include_counts_timing_and_stop_reason(
    manager, tmp_path, caplog, monkeypatch
):
    monkeypatch.setattr(logging.getLogger("pymilvus.bulk_writer"), "handlers", [caplog.handler])
    source = tmp_path / "source"
    write(source, "nested/file")
    manager.objects["prefix/data/nested/file"] = (3, 0)
    caplog.set_level("INFO")
    manager.upload_file_to_volume(str(source), "data")
    assert "scannedDirectories:2" in caplog.text
    assert "elapsedMillis:" in caplog.text
    assert "unmatchedLocalFiles:0" in caplog.text
    assert "filesToUpload:0" in caplog.text
    assert "stopReason:all local keys checked" in caplog.text
    manager.objects.clear()
    caplog.clear()
    manager.upload_file_to_volume(str(source), "data")
    assert "unmatchedLocalFiles:1" in caplog.text
    assert "stopReason:end of listing" in caplog.text


@pytest.mark.parametrize(
    "limit,reason",
    [
        ("FILE_LIMIT", "file count"),
        ("DIRECTORY_LIMIT", "pending directory count"),
        ("MEMORY_LIMIT", "estimated index memory"),
    ],
)
def test_sqlite_switch_logs_reason_and_cleans_up(tmp_path, monkeypatch, caplog, limit, reason):
    monkeypatch.setattr(logging.getLogger("pymilvus.bulk_writer"), "handlers", [caplog.handler])
    caplog.set_level("INFO")
    monkeypatch.setattr(_UploadManifest, limit, 0)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    source = tmp_path / "source"
    write(source, "nested/file")
    with _UploadManifest(str(scratch)) as manifest:
        manifest.scan(source)
        assert manifest.storage == "sqlite"
    assert f"reason:{reason} exceeds 0" in caplog.text
    assert "estimatedMemoryBytes:" in caplog.text
    assert not list(scratch.iterdir())

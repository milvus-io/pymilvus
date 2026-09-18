from __future__ import annotations

import logging
import math
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import requests
import urllib3
from minio import Minio
from minio.error import S3Error, ServerError

from pymilvus.bulk_writer._upload_executor import run_bounded
from pymilvus.bulk_writer._upload_manifest import _UploadEntry, _UploadManifest
from pymilvus.bulk_writer._volume_listing import list_objects_page
from pymilvus.bulk_writer.constants import ConnectType
from pymilvus.bulk_writer.endpoint_resolver import EndpointResolver
from pymilvus.bulk_writer.upload_policy import UploadPolicy
from pymilvus.bulk_writer.volume_restful import apply_volume

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from minio.helpers import DictType


_MIB = 1024 * 1024
_MIN_MULTIPART_PART_SIZE = 5 * _MIB
_TARGET_MULTIPART_PART_COUNT = 1000
_MAX_MULTIPART_PART_COUNT = 10000
_HTTP_CONNECT_TIMEOUT_SECONDS = 10.0
_HTTP_READ_TIMEOUT_SECONDS = 300.0
_UPLOAD_PROGRESS_IDLE_TIMEOUT_SECONDS = 300.0


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    value = float(size)
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1
    return f"{value:.2f} {units[unit_index]}"


def _format_duration(seconds: float) -> str:
    if seconds < 0:
        return "unknown"
    total_seconds = math.ceil(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _format_part_size(part_size: int) -> str:
    if part_size <= 0:
        return "auto"
    return f"{part_size} bytes ({_format_bytes(part_size)})"


@dataclass(frozen=True)
class UploadProgress:
    uploaded_bytes: int
    total_bytes: int
    completed_files: int
    total_files: int
    current_file: str
    current_file_uploaded_bytes: int
    current_file_total_bytes: int
    percent: float


class _UploadProgressCallbackError(RuntimeError):
    pass


class _UploadProgressIdleTimeoutError(TimeoutError):
    pass


def _calculate_upload_part_size(file_size: int, requested_part_size: int = 0) -> int:
    if requested_part_size > 0:
        return requested_part_size
    if file_size <= 0:
        return _MIN_MULTIPART_PART_SIZE
    target_part_size = math.ceil(file_size / _TARGET_MULTIPART_PART_COUNT)
    max_part_count_size = math.ceil(file_size / _MAX_MULTIPART_PART_COUNT)
    part_size = max(_MIN_MULTIPART_PART_SIZE, target_part_size, max_part_count_size)
    return int(math.ceil(part_size / _MIB) * _MIB)


class _UploadProgressTracker:
    _LOG_INTERVAL_SECONDS = 5.0

    def __init__(
        self,
        total_bytes: int,
        total_files: int,
        progress_callback: Callable[[UploadProgress], None] | None = None,
    ):
        self._total_bytes = total_bytes
        self._total_files = total_files
        self._progress_callback = progress_callback
        self._file_progress: dict[str, int] = {}
        self._completed_files = 0
        self._uploaded_bytes = 0
        self._last_log_time = 0.0
        self._start_time = time.time()
        self._lock = threading.Lock()

    def reset_file(self, file_path: str) -> None:
        with self._lock:
            previous = self._file_progress.get(file_path, 0)
            self._uploaded_bytes -= previous
            self._file_progress[file_path] = 0

    def update_file(self, file_path: str, file_size: int, chunk_size: int) -> None:
        if chunk_size <= 0:
            return
        with self._lock:
            previous = self._file_progress.get(file_path, 0)
            current = min(file_size, previous + chunk_size)
            delta = current - previous
            if delta <= 0:
                return
            self._file_progress[file_path] = current
            self._uploaded_bytes += delta
            progress = self._progress_if_needed(file_path, current, file_size)
        if progress is not None:
            self._emit_progress(progress)

    def finish_file(self, file_path: str, file_size: int) -> tuple[int, int, float]:
        with self._lock:
            previous = self._file_progress.pop(file_path, 0)
            self._uploaded_bytes += file_size - previous
            self._completed_files += 1
            percent = self._percent()
            progress = self._progress_if_needed(file_path, file_size, file_size)
            uploaded_bytes = self._uploaded_bytes
            completed_files = self._completed_files
        if progress is not None:
            self._emit_progress(progress)
        return uploaded_bytes, completed_files, percent

    def finish_upload(self) -> None:
        with self._lock:
            progress = self._snapshot("", 0, 0, self._percent())
            self._mark_progress_emitted(progress.percent)
        self._emit_progress(progress)

    def _snapshot(
        self,
        current_file: str,
        current_file_uploaded_bytes: int,
        current_file_total_bytes: int,
        percent: float | None = None,
    ) -> UploadProgress:
        return UploadProgress(
            uploaded_bytes=self._uploaded_bytes,
            total_bytes=self._total_bytes,
            completed_files=self._completed_files,
            total_files=self._total_files,
            current_file=current_file,
            current_file_uploaded_bytes=current_file_uploaded_bytes,
            current_file_total_bytes=current_file_total_bytes,
            percent=self._percent() if percent is None else percent,
        )

    def _mark_progress_emitted(self, _percent: float) -> None:
        self._last_log_time = time.time()

    def speed_bps(self) -> int:
        with self._lock:
            return self._speed_bps_locked()

    def estimated_remaining_time(self) -> str:
        with self._lock:
            return self._estimated_remaining_time_locked()

    def _speed_bps_locked(self) -> int:
        elapsed = max(0.001, time.time() - self._start_time)
        return int(self._uploaded_bytes / elapsed)

    def _estimated_remaining_time_locked(self) -> str:
        remaining_bytes = max(0, self._total_bytes - self._uploaded_bytes)
        if remaining_bytes == 0:
            return "0s"
        speed_bps = self._speed_bps_locked()
        if speed_bps <= 0:
            return "unknown"
        return _format_duration(remaining_bytes / speed_bps)

    def _emit_progress(self, progress: UploadProgress) -> None:
        speed_bps = self.speed_bps()
        estimated_remaining_time = self.estimated_remaining_time()
        logger.info(
            "Upload progress: %s/%s bytes, progress: %.2f%%, files: %s/%s, "
            "speedBPS:%s, estimatedRemainingTime:%s",
            progress.uploaded_bytes,
            progress.total_bytes,
            progress.percent,
            progress.completed_files,
            progress.total_files,
            speed_bps,
            estimated_remaining_time,
        )
        if self._progress_callback is not None:
            try:
                self._progress_callback(progress)
            except Exception as exc:
                msg = "Upload progress callback failed"
                raise _UploadProgressCallbackError(msg) from exc

    def _percent(self) -> float:
        if self._total_bytes == 0:
            return (
                100.0
                if self._total_files == 0
                else self._completed_files * 100.0 / self._total_files
            )
        return min(100.0, self._uploaded_bytes / self._total_bytes * 100)

    def _progress_if_needed(
        self, current_file: str, current_file_uploaded_bytes: int, current_file_total_bytes: int
    ) -> UploadProgress | None:
        now = time.time()
        percent = self._percent()
        if now - self._last_log_time >= self._LOG_INTERVAL_SECONDS:
            self._last_log_time = now
            return self._snapshot(
                current_file, current_file_uploaded_bytes, current_file_total_bytes, percent
            )
        return None


class _FileUploadProgress:
    def __init__(
        self,
        tracker: _UploadProgressTracker,
        file_path: str,
        file_size: int,
        idle_timeout_seconds: float = _UPLOAD_PROGRESS_IDLE_TIMEOUT_SECONDS,
        stop: threading.Event | None = None,
    ):
        self._tracker = tracker
        self._stop = stop
        self._file_path = file_path
        self._file_size = file_size
        self._idle_timeout_seconds = idle_timeout_seconds
        self._last_progress_time = time.monotonic()
        self._uploaded_bytes = 0
        self._lock = threading.Lock()

    def set_meta(self, object_name: str, total_length: int) -> None:
        _ = object_name
        self._file_size = total_length

    def reset(self) -> None:
        with self._lock:
            self._last_progress_time = time.monotonic()
            self._uploaded_bytes = 0
        self._tracker.reset_file(self._file_path)

    def update(self, size: int) -> None:
        if self._stop is not None and self._stop.is_set():
            msg = "Volume upload stopped"
            raise RuntimeError(msg)
        self._raise_if_idle_too_long()
        self._tracker.update_file(self._file_path, self._file_size, size)
        if size <= 0:
            return
        with self._lock:
            self._uploaded_bytes = min(self._file_size, self._uploaded_bytes + size)
            self._last_progress_time = time.monotonic()

    def _raise_if_idle_too_long(self) -> None:
        if self._idle_timeout_seconds <= 0:
            return
        now = time.monotonic()
        with self._lock:
            idle_seconds = now - self._last_progress_time
            uploaded_bytes = self._uploaded_bytes
        if idle_seconds <= self._idle_timeout_seconds:
            return
        msg = (
            f"No upload progress for {_format_duration(idle_seconds)} while uploading "
            f"{self._file_path}; uploaded {uploaded_bytes}/{self._file_size} bytes"
        )
        raise _UploadProgressIdleTimeoutError(msg)


@dataclass
class _VolumeSession:
    info: dict
    client: Any
    users: int = 0
    retired: bool = False

    def close(self) -> None:
        http = getattr(self.client, "_http", None)
        if http is not None:
            http.clear()


class _VolumeUploadContext:
    def __init__(self, volume_info: dict, client: Any, refresh_margin: timedelta):
        self._session = _VolumeSession(volume_info, client)
        self.refresh_margin = refresh_margin
        self._state_lock = threading.RLock()
        self._refresh_lock = threading.Lock()

    def get_state(self) -> tuple[dict, Any]:
        with self._state_lock:
            return self._session.info, self._session.client

    @contextmanager
    def borrow(self):
        with self._state_lock:
            session = self._session
            session.users += 1
        try:
            yield session.info, session.client
        finally:
            with self._state_lock:
                session.users -= 1
                close = session.retired and session.users == 0
            if close:
                session.close()

    def set_state(self, volume_info: dict, client: Any) -> None:
        with self._state_lock:
            previous = self._session
            previous.retired = True
            close = previous.users == 0
            self._session = _VolumeSession(volume_info, client)
        if close:
            previous.close()

    def close(self) -> None:
        with self._state_lock:
            session = self._session
            close = not session.retired and session.users == 0
            session.retired = True
        if close:
            session.close()

    def credential_expiring_soon(self) -> bool:
        volume_info, _ = self.get_state()
        expire_time_str = volume_info["credentials"]["expireTime"]
        expire_time = datetime.fromisoformat(expire_time_str.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) + self.refresh_margin >= expire_time


class OAuthMinio(Minio):
    def __init__(self, *args, oauth_token: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.oauth_token = oauth_token

    def _url_open(
        self,
        method: str,
        region: str,
        bucket_name: str | None = None,
        object_name: str | None = None,
        body: bytes | None = None,
        headers: DictType | None = None,
        query_params: DictType | None = None,
        preload_content: bool = True,
        no_body_trace: bool = False,
    ):
        headers = headers or {}
        if self.oauth_token:
            headers["Authorization"] = f"Bearer {self.oauth_token}"
        return super()._url_open(
            method,
            region,
            bucket_name=bucket_name,
            object_name=object_name,
            headers=headers,
            query_params=query_params,
            body=body,
            preload_content=preload_content,
        )


def _create_minio_http_client() -> urllib3.PoolManager:
    return urllib3.PoolManager(
        maxsize=100,
        timeout=urllib3.Timeout(
            connect=_HTTP_CONNECT_TIMEOUT_SECONDS,
            read=_HTTP_READ_TIMEOUT_SECONDS,
        ),
    )


class VolumeFileManager:
    _DEFAULT_CREDENTIAL_REFRESH_MARGIN_SECONDS = 300

    def __init__(
        self,
        cloud_endpoint: str,
        api_key: str,
        volume_name: str,
        connect_type: ConnectType = ConnectType.AUTO,
    ):
        """
        Args:
            cloud_endpoint (str): The fixed cloud endpoint URL.
                - For international regions: https://api.cloud.zilliz.com
                - For regions in China: https://api.cloud.zilliz.com.cn
            api_key (str): The API key associated with your organization
            volume_name (str): The name of the Volume.
            connect_type: Current value is mainly for Aliyun OSS buckets, default is Auto.
             - Default case, if the OSS bucket is reachable via the internal endpoint,
               the internal endpoint will be used
             - otherwise, the public endpoint will be used.
             - You can also force the use of either the internal or public endpoint.
        """
        self.cloud_endpoint = cloud_endpoint
        self.api_key = api_key
        self.volume_name = volume_name
        self.connect_type = connect_type
        self.credential_refresh_margin = timedelta(
            seconds=self._DEFAULT_CREDENTIAL_REFRESH_MARGIN_SECONDS
        )
        self.local_file_paths = []
        self.total_bytes = 0
        self.volume_info = {}
        self._client = None
        self._state_lock = threading.RLock()
        self._refresh_lock = threading.Lock()

    def _convert_dir_path(self, input_path: str):
        if not input_path or input_path == "/":
            return ""
        parts = input_path.replace("\\", "/").split("/")
        if ".." in parts:
            msg = f"target volume path must not escape the volume root: {input_path}"
            raise ValueError(msg)
        normalized_path = "/".join(part for part in parts if part not in ("", "."))
        return normalized_path + "/" if normalized_path else ""

    def _create_volume_state(self, path: str):
        logger.info("refreshing volume info...")
        response = apply_volume(self.cloud_endpoint, self.api_key, self.volume_name, path)
        volume_info = response.json()["data"]
        logger.info("volume info refreshed.")

        creds = volume_info["credentials"]
        http_client = _create_minio_http_client()

        try:
            cloud = volume_info["cloud"]
            region = volume_info["region"]
            endpoint = EndpointResolver.resolve_endpoint(
                volume_info["endpoint"],
                cloud,
                region,
                self.connect_type,
            )

            session_token = creds["sessionToken"]
            if cloud == "gcp":
                client = OAuthMinio(
                    endpoint=endpoint,
                    region=region,
                    secure=True,
                    oauth_token=session_token,
                    http_client=http_client,
                )
            else:
                client = Minio(
                    endpoint=endpoint,
                    access_key=creds["tmpAK"],
                    secret_key=creds["tmpSK"],
                    session_token=session_token,
                    region=region,
                    secure=True,
                    http_client=http_client,
                )
        except BaseException:
            http_client.clear()
            raise
        return volume_info, client

    def _refresh_volume_and_client(self, path: str):
        volume_info, client = self._create_volume_state(path)
        with self._state_lock:
            self.volume_info = volume_info
            self._client = client
        logger.info("storage client refreshed")
        return _VolumeUploadContext(volume_info, client, self.credential_refresh_margin)

    def _get_volume_state(self):
        with self._state_lock:
            return self.volume_info, self._client

    def _credential_expiring_soon(self, context: _VolumeUploadContext | None = None) -> bool:
        if context is not None:
            return context.credential_expiring_soon()
        volume_info, client = self._get_volume_state()
        if not volume_info or client is None:
            return True
        return _VolumeUploadContext(
            volume_info, client, self.credential_refresh_margin
        ).credential_expiring_soon()

    def _refresh_volume_and_client_if_needed(
        self, volume_path: str, context: _VolumeUploadContext | None = None
    ):
        if not self._credential_expiring_soon(context):
            return
        if context is None:
            with self._refresh_lock:
                if self._credential_expiring_soon():
                    self._refresh_volume_and_client(volume_path)
            return

        with context._refresh_lock:
            if context.credential_expiring_soon():
                volume_info, client = self._create_volume_state(volume_path)
                context.set_state(volume_info, client)
                with self._state_lock:
                    self.volume_info = volume_info
                    self._client = client
                logger.info("storage client refreshed")

    def _validate_size(
        self,
        local_file_paths: list[str] | int | None = None,
        total_bytes: int | None = None,
        volume_info: dict | None = None,
    ):
        local_file_paths = self.local_file_paths if local_file_paths is None else local_file_paths
        file_size_total = self.total_bytes if total_bytes is None else total_bytes
        volume_info = self.volume_info if volume_info is None else volume_info
        file_count = (
            local_file_paths if isinstance(local_file_paths, int) else len(local_file_paths)
        )
        file_size_limit = volume_info["condition"].get("maxContentLength")
        if file_size_limit is not None and file_size_total > file_size_limit:
            error_message = (
                f"localFileTotalSize {file_size_total} exceeds "
                f"the maximum contentLength limit {file_size_limit} defined in the condition."
                f"If you are using the free tier, "
                f"you may switch to the pay-as-you-go volume plan to support uploading larger files."
            )
            raise ValueError(error_message)

        file_number_limit = volume_info["condition"].get("maxFileNumber")
        if file_number_limit is not None and file_count > file_number_limit:
            error_message = (
                f"localFileTotalNumber {file_count} exceeds "
                f"the maximum fileNumber limit {file_number_limit} defined in the condition."
                f"If you are using the free tier, "
                f"you may switch to the pay-as-you-go volume plan to support uploading larger files."
            )
            raise ValueError(error_message)

    def upload_file_to_volume(
        self,
        source_file_path: str,
        target_volume_path: str,
        upload_concurrency: int = 5,
        max_retries: int = 5,
        retry_interval: float = 5.0,
        progress_callback: Callable[[UploadProgress], None] | None = None,
        part_size: int = 0,
        upload_policy: UploadPolicy = UploadPolicy.SKIP_IF_SAME_SIZE,
        temporary_directory: str | None = None,
    ):
        """Upload a local file/directory using paginated LIST checks (no HEAD/GET).

        ``upload_policy`` defaults to skipping exact target keys with equal sizes.
        ``OVERWRITE`` avoids LIST; ``SIZE_AND_MTIME`` also requires the local mtime
        to be no newer than the remote server LastModified, comparing milliseconds.
        These are heuristics, not checksums. Target paths containing a ``..`` segment
        are rejected. VolumeBulkWriter always uses OVERWRITE for generated chunks.

        At most ``upload_concurrency`` files are in flight. ``max_retries`` counts
        additional attempts (zero disables retry). Preparation progress is logged;
        ``progress_callback`` receives throttled upload snapshots outside the tracker
        lock, plus a final snapshot. Callbacks may run on different worker threads.

        Small manifests stay in memory; large ones automatically use stdlib SQLite
        under ``temporary_directory`` (default: system temp). Temporary files are
        removed after workers stop, including on failure. Abrupt process termination
        may leave them behind. Symlinks retain their logical upload paths; cycles,
        broken links and special files fail planning. Keep sources unchanged.
        """
        policy = UploadPolicy(upload_policy)
        root = Path(source_file_path).absolute()
        volume_path = self._convert_dir_path(target_volume_path)
        concurrency = max(1, upload_concurrency)
        max_retries = max(0, max_retries)
        retry_interval = max(0.0, retry_interval)
        stop = threading.Event()
        context = None
        started_at = time.monotonic()
        logger.info(
            "Planning volume upload: sourcePath:%s, volumePath:%s, policy:%s",
            root,
            volume_path,
            policy.value,
        )
        try:
            with _UploadManifest(temporary_directory) as manifest:
                single = manifest.scan(root)
                context = self._refresh_volume_and_client(volume_path)
                info, _ = context.get_state()
                prefix = info["volumePrefix"] + volume_path
                if policy is not UploadPolicy.OVERWRITE:
                    self._filter_existing(
                        manifest,
                        prefix,
                        prefix + root.name if single else prefix,
                        policy,
                        volume_path,
                        max_retries,
                        retry_interval,
                        context,
                    )
                info, _ = context.get_state()
                self._validate_size(manifest.remaining_count, manifest.remaining_bytes, info)
                logger.info(
                    "Volume upload plan: filesToUpload:%s, skippedFiles:%s, bytesToUpload:%s",
                    manifest.remaining_count,
                    manifest.file_count - manifest.remaining_count,
                    manifest.remaining_bytes,
                )
                tracker = _UploadProgressTracker(
                    manifest.remaining_bytes, manifest.remaining_count, progress_callback
                )
                base = root.parent if single else root

                def upload(entry: _UploadEntry):
                    path = base / entry.path
                    self._validate_source(path, entry.size, entry.mtime_ns)
                    progress = _FileUploadProgress(tracker, str(path), entry.size, stop=stop)
                    self._put_object(
                        str(path),
                        prefix + entry.path,
                        volume_path,
                        max_retries,
                        retry_interval,
                        progress,
                        context,
                        entry.size,
                        max(0, part_size),
                        policy,
                        entry.mtime_ns,
                        stop,
                    )
                    self._validate_source(path, entry.size, entry.mtime_ns)
                    tracker.finish_file(str(path), entry.size)

                run_bounded(manifest.entries(), upload, concurrency, stop)
                tracker.finish_upload()
                logger.info(
                    "Volume upload completed: files:%s, skippedFiles:%s, elapsedSeconds:%.3f",
                    manifest.remaining_count,
                    manifest.file_count - manifest.remaining_count,
                    time.monotonic() - started_at,
                )
                info, _ = context.get_state()
                return {
                    "volumeName": info["volumeName"],
                    "volume_name": info["volumeName"],
                    "path": volume_path,
                }
        except Exception:
            logger.warning("Volume upload failed: sourcePath:%s, volumePath:%s", root, volume_path)
            raise
        finally:
            if context is not None:
                context.close()

    @staticmethod
    def _validate_source(path: Path, size: int, mtime_ns: int) -> None:
        attrs = path.stat()
        if not path.is_file() or attrs.st_size != size or attrs.st_mtime_ns != mtime_ns:
            msg = f"Local file changed after upload planning: {path}"
            raise ValueError(msg)

    def _filter_existing(
        self,
        manifest: _UploadManifest,
        target_prefix: str,
        listing_prefix: str,
        policy: UploadPolicy,
        volume_path: str,
        retries: int,
        interval: float,
        context: _VolumeUploadContext,
    ) -> None:
        if not manifest.remaining_count:
            logger.info(
                "Volume existence check skipped: prefix:%s, reason:no local files", listing_prefix
            )
            return
        token = None
        pages = objects = 0
        started_at = last_log = time.monotonic()
        logger.info(
            "Volume existence check started: prefix:%s, localFiles:%s",
            listing_prefix,
            manifest.file_count,
        )
        while True:
            page, next_token = self._with_retry(
                lambda info, client, attempt, page_token=token: list_objects_page(
                    client, info["bucketName"], listing_prefix, page_token
                ),
                volume_path,
                retries,
                interval,
                context,
            )
            pages += 1
            objects += len(page)
            for key, size, modified in page:
                if key.startswith(target_prefix):
                    manifest.match(key[len(target_prefix) :], size, modified, policy)
            done = next_token is None or manifest.unmatched_count == 0
            now = time.monotonic()
            if done or now - last_log >= 5:
                logger.info(
                    "Volume existence check %s: prefix:%s, pages:%s, objects:%s, skippedFiles:%s, "
                    "unmatchedLocalFiles:%s, filesToUpload:%s, elapsedMillis:%s, stopReason:%s",
                    "completed" if done else "progress",
                    listing_prefix,
                    pages,
                    objects,
                    manifest.file_count - manifest.remaining_count,
                    manifest.unmatched_count,
                    manifest.remaining_count,
                    int((now - started_at) * 1000),
                    (
                        (
                            "all local keys checked"
                            if manifest.unmatched_count == 0
                            else "end of listing"
                        )
                        if done
                        else "in progress"
                    ),
                )
                last_log = now
            if done:
                return
            token = next_token

    @staticmethod
    def _retryable(error: Exception) -> bool:
        chain = []
        seen = set()
        current = error
        while isinstance(current, BaseException) and id(current) not in seen:
            seen.add(id(current))
            chain.append(current)
            current = (
                current.__cause__
                or getattr(current, "reason", None)
                or (current.__context__ if not current.__suppress_context__ else None)
            )
        if any(
            isinstance(
                item,
                (
                    ValueError,
                    TypeError,
                    FileNotFoundError,
                    PermissionError,
                    _UploadProgressCallbackError,
                    KeyboardInterrupt,
                    SystemExit,
                ),
            )
            for item in chain
        ):
            return False
        for item in chain:
            if isinstance(item, S3Error):
                status = getattr(item.response, "status", 0)
                return (
                    status in {408, 429}
                    or status >= 500
                    or item.code
                    in {
                        "ExpiredToken",
                        "SecurityTokenExpired",
                        "RequestTimeout",
                        "SlowDown",
                        "InternalError",
                        "ServiceUnavailable",
                        "Throttling",
                        "ThrottlingException",
                        "RequestLimitExceeded",
                    }
                )
            if isinstance(item, ServerError):
                status = getattr(item, "status_code", None)
                if status is None:
                    # MinIO 7.0 reports the HTTP status only in this fixed message.
                    match = re.fullmatch(r"server failed with HTTP status code (\d{3})", str(item))
                    status = int(match[1]) if match else 0
                return status in {408, 429} or status >= 500
            if isinstance(item, requests.RequestException):
                response = getattr(item, "response", None)
                status = getattr(response, "status_code", 0) if response is not None else 0
                if status:
                    return status in {408, 429} or status >= 500
        return any(
            isinstance(
                item,
                (
                    TimeoutError,
                    ConnectionError,
                    OSError,
                    urllib3.exceptions.TimeoutError,
                    urllib3.exceptions.ProtocolError,
                    urllib3.exceptions.MaxRetryError,
                ),
            )
            for item in chain
        )

    def _refresh_failed_session(
        self, volume_path: str, context: _VolumeUploadContext | None, failed_client: Any
    ) -> None:
        if context is None:
            self._refresh_volume_and_client(volume_path)
            return
        with context._refresh_lock:
            if context.get_state()[1] is failed_client:
                info, client = self._create_volume_state(volume_path)
                context.set_state(info, client)
                with self._state_lock:
                    self.volume_info, self._client = info, client

    def _with_retry(
        self,
        action: Callable,
        volume_path: str,
        retries: int,
        interval: float,
        context: _VolumeUploadContext | None = None,
        stop: threading.Event | None = None,
    ):
        attempt = 0
        failed_client = None
        while True:
            if stop is not None and stop.is_set():
                msg = "Volume upload stopped"
                raise RuntimeError(msg)
            client = None
            phase = "proactive refresh"
            try:
                if failed_client is not None:
                    phase = "reactive refresh"
                    self._refresh_failed_session(volume_path, context, failed_client)
                    failed_client = None
                phase = "proactive refresh"
                self._refresh_volume_and_client_if_needed(volume_path, context)
                phase = "storage request"
                if context is None:
                    info, client = self._get_volume_state()
                    return action(info, client, attempt)
                with context.borrow() as (info, client):
                    return action(info, client, attempt)
            except Exception as exc:
                if not self._retryable(exc):
                    raise
                if attempt >= max(0, retries):
                    msg = f"Upload failed after {attempt + 1} attempts"
                    raise RuntimeError(msg) from exc
                logger.warning(
                    "Volume upload %s failed; retry %s/%s: %s",
                    phase,
                    attempt + 1,
                    retries,
                    type(exc).__name__,
                )
                attempt += 1
                if phase == "storage request":
                    failed_client = client
                if stop is None:
                    time.sleep(interval)
                elif stop.wait(interval):
                    msg = "Volume upload stopped"
                    raise RuntimeError(msg) from exc

    def _put_object(
        self,
        file_path: str,
        remote_file_path: str,
        volume_path: str,
        max_retries: int = 5,
        retry_interval: float = 5.0,
        progress: _FileUploadProgress | None = None,
        context: _VolumeUploadContext | None = None,
        file_size: int | None = None,
        part_size: int = 0,
        upload_policy: UploadPolicy = UploadPolicy.OVERWRITE,
        mtime_ns: int | None = None,
        stop: threading.Event | None = None,
    ):
        self._upload_with_retry(
            file_path,
            remote_file_path,
            volume_path,
            max_retries,
            retry_interval,
            progress,
            context,
            file_size,
            part_size,
            upload_policy,
            mtime_ns,
            stop,
        )

    def _upload_with_retry(
        self,
        file_path: str,
        object_name: str,
        volume_path: str,
        max_retries: int = 5,
        retry_interval: float = 5.0,
        progress: _FileUploadProgress | None = None,
        context: _VolumeUploadContext | None = None,
        file_size: int | None = None,
        part_size: int = 0,
        upload_policy: UploadPolicy = UploadPolicy.OVERWRITE,
        mtime_ns: int | None = None,
        stop: threading.Event | None = None,
    ):
        if file_size is None:
            file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
        part_size = _calculate_upload_part_size(file_size, part_size)

        put_attempted = False

        def put(info: dict, client: Any, _attempt: int):
            nonlocal put_attempted
            if client is None:
                msg = "Storage client is not initialized"
                raise RuntimeError(msg)
            if progress is not None:
                progress.reset()
            if mtime_ns is not None:
                self._validate_source(Path(file_path), file_size, mtime_ns)
            # A failed PUT may have committed. Retry checks are exact-key LIST requests,
            # separate from the single prefix scan used by normal preflight planning.
            if put_attempted and upload_policy is not UploadPolicy.OVERWRITE:
                token = None
                while True:
                    page, token = list_objects_page(client, info["bucketName"], object_name, token)
                    found = False
                    for key, size, modified in page:
                        if key == object_name:
                            if upload_policy.should_skip(file_size, mtime_ns or 0, size, modified):
                                return
                            found = True
                            break
                    if found or token is None:
                        break
            kwargs = {
                "bucket_name": info["bucketName"],
                "object_name": object_name,
                "file_path": file_path,
                "part_size": part_size,
            }
            if progress is not None:
                kwargs["progress"] = progress
            put_attempted = True
            client.fput_object(**kwargs)

        self._with_retry(put, volume_path, max_retries, retry_interval, context, stop)

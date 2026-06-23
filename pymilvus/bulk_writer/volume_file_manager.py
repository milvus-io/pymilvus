from __future__ import annotations

import logging
import math
import posixpath
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import urllib3
from minio import Minio
from minio.error import S3Error

from pymilvus.bulk_writer.constants import ConnectType
from pymilvus.bulk_writer.endpoint_resolver import EndpointResolver
from pymilvus.bulk_writer.file_utils import FileUtils
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


def _format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


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
    _LOG_PERCENT_STEP = 1.0

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
        self._completed_files: set[str] = set()
        self._uploaded_bytes = 0
        self._last_log_time = 0.0
        self._last_logged_percent = -1.0
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
            previous = self._file_progress.get(file_path, 0)
            current = max(previous, file_size)
            self._file_progress[file_path] = current
            self._uploaded_bytes += current - previous
            if file_path not in self._completed_files:
                self._completed_files.add(file_path)
            percent = self._percent()
            progress = self._snapshot(file_path, current, file_size, percent)
            uploaded_bytes = self._uploaded_bytes
            completed_files = len(self._completed_files)
            self._mark_progress_emitted(percent)
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
            completed_files=len(self._completed_files),
            total_files=self._total_files,
            current_file=current_file,
            current_file_uploaded_bytes=current_file_uploaded_bytes,
            current_file_total_bytes=current_file_total_bytes,
            percent=self._percent() if percent is None else percent,
        )

    def _mark_progress_emitted(self, percent: float) -> None:
        self._last_log_time = time.time()
        self._last_logged_percent = percent

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
            return 100.0
        return min(100.0, self._uploaded_bytes / self._total_bytes * 100)

    def _progress_if_needed(
        self, current_file: str, current_file_uploaded_bytes: int, current_file_total_bytes: int
    ) -> UploadProgress | None:
        now = time.time()
        percent = self._percent()
        if (
            percent - self._last_logged_percent >= self._LOG_PERCENT_STEP
            or now - self._last_log_time >= self._LOG_INTERVAL_SECONDS
        ):
            self._last_log_time = now
            self._last_logged_percent = percent
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
    ):
        self._tracker = tracker
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


class _VolumeUploadContext:
    def __init__(self, volume_info: dict, client: Any, refresh_margin: timedelta):
        self.volume_info = volume_info
        self.client = client
        self.refresh_margin = refresh_margin
        self._state_lock = threading.RLock()
        self._refresh_lock = threading.Lock()

    def get_state(self) -> tuple[dict, Any]:
        with self._state_lock:
            return self.volume_info, self.client

    def set_state(self, volume_info: dict, client: Any) -> None:
        with self._state_lock:
            self.volume_info = volume_info
            self.client = client

    def credential_expiring_soon(self) -> bool:
        volume_info, _ = self.get_state()
        expire_time_str = volume_info["credentials"]["expireTime"]
        expire_time = datetime.fromisoformat(expire_time_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return now + self.refresh_margin >= expire_time


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
        normalized_path = input_path.replace("\\", "/").lstrip("/")
        normalized_path = posixpath.normpath(normalized_path)
        if normalized_path in ("", "."):
            return ""
        if normalized_path == ".." or normalized_path.startswith("../"):
            msg = f"target volume path must not escape the volume root: {input_path}"
            raise ValueError(msg)
        return normalized_path.rstrip("/") + "/"

    def _create_volume_state(self, path: str):
        logger.info("refreshing volume info...")
        response = apply_volume(self.cloud_endpoint, self.api_key, self.volume_name, path)
        volume_info = response.json()["data"]
        logger.info("volume info refreshed.")

        creds = volume_info["credentials"]
        http_client = _create_minio_http_client()

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
        local_file_paths: list[str] | None = None,
        total_bytes: int | None = None,
        volume_info: dict | None = None,
    ):
        local_file_paths = self.local_file_paths if local_file_paths is None else local_file_paths
        file_size_total = self.total_bytes if total_bytes is None else total_bytes
        volume_info = self.volume_info if volume_info is None else volume_info
        file_size_limit = volume_info["condition"]["maxContentLength"]
        if file_size_total > file_size_limit:
            error_message = (
                f"localFileTotalSize {file_size_total} exceeds "
                f"the maximum contentLength limit {file_size_limit} defined in the condition."
                f"If you are using the free tier, "
                f"you may switch to the pay-as-you-go volume plan to support uploading larger files."
            )
            raise ValueError(error_message)

        file_number_limit = volume_info["condition"].get("maxFileNumber")
        if file_number_limit is not None and len(local_file_paths) > file_number_limit:
            error_message = (
                f"localFileTotalNumber {len(local_file_paths)} exceeds "
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
    ):
        """
        uploads a local file or directory to the specified path within the Volume.

        Args:
            source_file_path: the source local file or directory path
            target_volume_path: the target directory path in the Volume
            upload_concurrency: the maximum number of files to upload concurrently
            max_retries: the maximum retry count for each file
            retry_interval: retry interval in seconds
            progress_callback: callback invoked with upload progress snapshots
            part_size: multipart upload part size in bytes, 0 means automatic
        Raises:
            Exception: If an error occurs during the upload process.
        """

        upload_concurrency = max(1, upload_concurrency)
        max_retries = max(1, max_retries)
        retry_interval = max(0.0, retry_interval)
        part_size = max(0, part_size)

        local_file_paths, total_bytes = FileUtils.process_local_path(source_file_path)
        volume_path = self._convert_dir_path(target_volume_path)
        file_count = len(local_file_paths)
        start_time = time.time()
        logger.info(
            "Starting volume upload: sourcePath:%s, volumeName:%s, volumePath:%s, "
            "totalFileCount:%s, totalFileSize:%s bytes (%s), uploadConcurrency:%s, "
            "maxRetries:%s, retryInterval:%s, partSize:%s, startTime:%s",
            source_file_path,
            self.volume_name,
            volume_path,
            file_count,
            total_bytes,
            _format_bytes(total_bytes),
            upload_concurrency,
            max_retries,
            _format_duration(retry_interval),
            _format_part_size(part_size),
            _format_timestamp(start_time),
        )

        try:
            upload_context = self._refresh_volume_and_client(volume_path)
            volume_info, _ = upload_context.get_state()
            self._validate_size(local_file_paths, total_bytes, volume_info)
            with self._state_lock:
                self.local_file_paths = local_file_paths
                self.total_bytes = total_bytes

            root_path = Path(source_file_path).resolve()
            progress_tracker = _UploadProgressTracker(total_bytes, file_count, progress_callback)

            def _upload_task(
                file_path: str,
                root_path: Path,
                volume_path: str,
                context: _VolumeUploadContext,
            ):
                path_obj = Path(file_path).resolve()
                if root_path.is_file():
                    relative_path = path_obj.name
                else:
                    relative_path = path_obj.relative_to(root_path).as_posix()

                volume_info, _ = context.get_state()
                volume_prefix = f"{volume_info['volumePrefix']}"
                file_start_time = time.time()
                try:
                    size = Path(file_path).stat().st_size
                    logger.info(f"uploading file, fileName:{file_path}, size:{size} bytes")
                    remote_file_path = volume_prefix + volume_path + relative_path
                    progress = _FileUploadProgress(progress_tracker, file_path, size)
                    self._put_object(
                        file_path,
                        remote_file_path,
                        volume_path,
                        max_retries,
                        retry_interval,
                        progress=progress,
                        context=context,
                        file_size=size,
                        part_size=part_size,
                    )
                    uploaded_bytes, uploaded_count, percent = progress_tracker.finish_file(
                        file_path, size
                    )
                    elapsed = time.time() - file_start_time
                    logger.info(
                        "Uploaded file %s/%s: %s (%s bytes) elapsed:%s, "
                        "progress(total bytes): %s/%s bytes, progress(total percentage):%.2f%%, "
                        "speedBPS:%s, estimatedRemainingTime:%s",
                        uploaded_count,
                        file_count,
                        file_path,
                        size,
                        _format_duration(elapsed),
                        uploaded_bytes,
                        total_bytes,
                        percent,
                        progress_tracker.speed_bps(),
                        progress_tracker.estimated_remaining_time(),
                    )
                except S3Error as e:
                    logger.error(f"Failed to upload {file_path}: {e!s}")
                    raise

            with ThreadPoolExecutor(max_workers=upload_concurrency) as executor:
                futures = []
                for _, file_path in enumerate(local_file_paths):
                    futures.append(
                        executor.submit(
                            _upload_task, file_path, root_path, volume_path, upload_context
                        )
                    )
                for f in futures:
                    f.result()  # wait for all

            progress_tracker.finish_upload()
            end_time = time.time()
            volume_info, _ = upload_context.get_state()
            logger.info(
                "Volume upload completed: sourcePath:%s, volumeName:%s, volumePath:%s, "
                "totalFileCount:%s, totalFileSize:%s bytes (%s), endTime:%s, totalElapsed:%s",
                source_file_path,
                volume_info["volumeName"],
                volume_path,
                file_count,
                total_bytes,
                _format_bytes(total_bytes),
                _format_timestamp(end_time),
                _format_duration(end_time - start_time),
            )
            return {
                "volumeName": volume_info["volumeName"],
                "volume_name": volume_info["volumeName"],
                "path": volume_path,
            }
        except Exception:
            end_time = time.time()
            logger.warning(
                "Volume upload failed: sourcePath:%s, volumeName:%s, volumePath:%s, "
                "endTime:%s, totalElapsed:%s",
                source_file_path,
                self.volume_name,
                volume_path,
                _format_timestamp(end_time),
                _format_duration(end_time - start_time),
            )
            raise

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
    ):
        self._refresh_volume_and_client_if_needed(volume_path, context)

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
    ):
        if file_size is None:
            try:
                file_size = Path(file_path).stat().st_size
            except OSError:
                file_size = 0
        upload_part_size = _calculate_upload_part_size(file_size, part_size)
        attempt = 0
        while attempt < max_retries:
            try:
                if progress is not None:
                    progress.reset()
                volume_info, client = (
                    context.get_state() if context is not None else self._get_volume_state()
                )
                if client is None:
                    msg = "Storage client is not initialized"
                    raise RuntimeError(msg)
                kwargs = {
                    "bucket_name": volume_info["bucketName"],
                    "object_name": object_name,
                    "file_path": file_path,
                    "part_size": upload_part_size,
                }
                if progress is not None:
                    kwargs["progress"] = progress
                client.fput_object(**kwargs)
                break
            except _UploadProgressCallbackError:
                raise
            except Exception as e:
                attempt += 1
                logger.warning(f"Attempt {attempt} failed to upload {file_path}: {e}")

                if attempt == max_retries:
                    error_message = f"Upload failed after {max_retries} attempts"
                    raise RuntimeError(error_message) from e

                if context is None:
                    self._refresh_volume_and_client(volume_path)
                else:
                    volume_info, client = self._create_volume_state(volume_path)
                    context.set_state(volume_info, client)
                    with self._state_lock:
                        self.volume_info = volume_info
                        self._client = client
                    logger.info("storage client refreshed")
                time.sleep(retry_interval)

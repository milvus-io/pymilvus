from typing import Any, Optional

from pymilvus.client.utils import current_time_ms


def is_valid_client_request_id(value: Any) -> bool:
    """Return whether value is a non-zero lowercase OpenTelemetry TraceID."""

    if not isinstance(value, str) or len(value) != 32 or value == "0" * 32:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _api_level_md(context: Optional["CallContext"]) -> Optional[list]:
    if context is None:
        return None
    return context.to_grpc_metadata()


class CallContext:
    def __init__(self, db_name: str = "", client_request_id: str = ""):
        self._db_name = db_name
        self._client_request_id = client_request_id

    def to_grpc_metadata(self):
        metadata = [
            ("dbname", self._db_name),
            ("client-request-unixmsec", current_time_ms()),
        ]
        # Preserve the legacy access-log contract: callers may attach any nonempty ID to
        # the wire. Telemetry correlation validates the stricter OTel trace-ID shape at
        # the recording boundary instead of silently dropping an existing header here.
        if self._client_request_id:
            metadata.append(("client-request-id", self._client_request_id))
        return metadata

    def get_db_name(self):
        return self._db_name

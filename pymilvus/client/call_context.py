from pymilvus.client.utils import current_time_ms


class CallContext:
    def __init__(self, db_name: str = "", client_request_id: str = ""):
        self._db_name = db_name
        self._client_request_id = client_request_id

    def to_grpc_metadata(self):
        return [
            ("dbname", self._db_name),
            ("client-request-id", self._client_request_id),
            ("client-request-unixmsec", current_time_ms()),
        ]

    def get_db_name(self):
        return self._db_name

"""Request configuration for per-request gRPC metadata headers."""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class RequestConfig:
    """Configuration for per-request gRPC metadata headers.

    This class encapsulates dynamic headers that need to be sent with each gRPC request,
    such as database name. It provides a centralized way to manage request-scoped settings.

    Attributes:
        db_name: The database name to use for this request.

    Example:
        >>> config = RequestConfig(db_name="my_database")
        >>> metadata = config.to_metadata()
        >>> # [("dbname", "my_database")]
    """

    db_name: str = ""

    def to_metadata(self) -> List[Tuple[str, str]]:
        """Convert config to gRPC metadata tuples.

        Returns:
            List of (key, value) tuples for gRPC metadata headers.
        """
        metadata = []
        if self.db_name:
            metadata.append(("dbname", self.db_name))
        return metadata

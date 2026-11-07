"""Single-page LIST adapter. MinIO's public iterator hides continuation tokens."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import unquote_plus
from xml.etree import ElementTree as ET


class _NoDoctypeTreeBuilder(ET.TreeBuilder):
    def doctype(self, name: str, pubid: str | None, system: str | None) -> None:
        msg = "DTD declarations are not allowed in object listings"
        raise ValueError(msg)


def _modified_ns(value: str | None) -> int | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        msg = "Object LastModified must include a timezone"
        raise ValueError(msg)
    delta = parsed - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1000


def list_objects_page(client: Any, bucket: str, prefix: str, token: str | None):
    query = {
        "list-type": "2",
        "max-keys": "1000",
        "prefix": prefix,
        "delimiter": "",
        "encoding-type": "url",
    }
    if token is not None:
        query["continuation-token"] = token
    response = client._execute("GET", bucket, query_params=query)
    try:
        parser = ET.XMLParser(target=_NoDoctypeTreeBuilder())  # noqa: S314 - rejects DTDs
        root = ET.fromstring(response.data, parser=parser)  # noqa: S314 - rejects DTDs
        ns = root.tag.partition("}")[0] + "}" if root.tag.startswith("{") else ""
        encoded = root.findtext(ns + "EncodingType") == "url"
        objects = []
        for item in root.findall(ns + "Contents"):
            key = item.findtext(ns + "Key")
            if key is None:
                msg = "Object listing is missing an object key"
                raise ValueError(msg)
            size = item.findtext(ns + "Size")
            modified = item.findtext(ns + "LastModified")
            objects.append(
                (
                    unquote_plus(key) if encoded else key,
                    int(size) if size is not None else None,
                    _modified_ns(modified),
                )
            )
        truncated = root.findtext(ns + "IsTruncated", "false").lower() == "true"
        next_token = root.findtext(ns + "NextContinuationToken") if truncated else None
        if truncated and (not next_token or next_token == token):
            msg = "Truncated object listing has no advancing continuation token"
            raise ValueError(msg)
        return objects, next_token
    finally:
        response.close()
        response.release_conn()

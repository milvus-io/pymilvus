"""Grep an existing collection with a VarChar/Text content field.

Requires a Milvus server supporting QueryScan and grep. Older servers may ignore
function_chains. The current implementation does not support count, aggregation,
GROUP BY, ORDER BY or query iterators with QueryScan.
"""

from pymilvus import MilvusClient

client = MilvusClient("http://localhost:19530")
rows = client.grep(
    collection_name="source_files",
    text_field="content",
    args="-i -e 'timeout'",
    filter="",
    output_fields=["id", "path"],
    limit=20,
)
for row in rows:
    print(row["id"], row["path"], row["matches"])

# Each matches value is a list of matching-line objects:
# [{"content": "timeout", "starts": [0], "ends": [7]}]
# Spans are half-open UTF-8 byte offsets within the returned line.
# Use match_field="text_hits" to customize the matching-details column name.
# Async usage: rows = await async_client.grep(...) with the same arguments.

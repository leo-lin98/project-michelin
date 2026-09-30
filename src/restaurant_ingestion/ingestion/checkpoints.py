"""Checkpoint identifiers and statuses."""

import hashlib
import json


PENDING = "pending"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"


def search_checkpoint(query: str, tile_index: int, page_index: int, rectangle: dict[str, dict[str, float]]) -> tuple[str, str]:
    payload = {"query": query, "tile_index": tile_index, "page_index": page_index, "rectangle": rectangle}
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:16]
    return f"search-{digest}", payload_json


def detail_checkpoint(place_id: str) -> str:
    digest = hashlib.sha256(place_id.encode("utf-8")).hexdigest()[:16]
    return f"detail-{digest}"

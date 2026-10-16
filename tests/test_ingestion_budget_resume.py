import json
import logging
from pathlib import Path

import httpx
import pytest

from restaurant_ingestion.clients.google_places import GooglePlacesClient
from restaurant_ingestion.storage.duckdb_store import DuckDbStore
from scripts import ingest_taipei


@pytest.mark.parametrize("discovery_budget", [3, 4])
def test_paginated_discovery_enriches_saved_pages_with_independent_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    discovery_budget: int,
) -> None:
    database_path = tmp_path / "pagination.duckdb"
    settings = {
        "DATABASE_PATH": database_path,
        "TAIPEI_GRID_SOUTH": 25.0,
        "TAIPEI_GRID_WEST": 121.0,
        "TAIPEI_GRID_NORTH": 25.1,
        "TAIPEI_GRID_EAST": 121.1,
        "TAIPEI_GRID_TILE_SIZE_DEGREES": 0.2,
        "DISCOVERY_QUERIES": ("restaurant", "cafe"),
        "ORDINARY_RESTAURANT_TARGET": 10,
        "TARGET_ENRICHED_RESTAURANTS": 10,
        "DISCOVERY_REQUEST_BUDGET": discovery_budget,
        "DETAILS_REQUEST_BUDGET": 2,
        "DETAILS_REFRESH_TTL_DAYS": 30,
        "PROGRESS_LOG_INTERVAL": 100,
    }
    for name, value in settings.items():
        monkeypatch.setattr(ingest_taipei, name, value)
    store = DuckDbStore(database_path)
    try:
        store.create_schema()
        store.log_api_call("older-run", "places:searchText", "text_search_ids_only", 200, 0, None, None)
    finally:
        store.close()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            payload = json.loads(request.content)
            if payload["textQuery"] == "cafe":
                return httpx.Response(200, json={"places": []})
            if payload.get("pageToken") == "second-page":
                return httpx.Response(200, json={"places": [{"id": "second"}]})
            return httpx.Response(200, json={"places": [{"id": "first"}], "nextPageToken": "second-page"})
        place_id = request.url.path.rsplit("/", 1)[1]
        return httpx.Response(200, json={
            "id": place_id, "displayName": {"text": place_id},
            "location": {"latitude": 25.0, "longitude": 121.0},
        })

    def build_client() -> GooglePlacesClient:
        client = GooglePlacesClient("test-only", "TEST_ONLY_KEY", "en", "TW", 1.0, 0, 0.0, 0.0)
        client._client.close()
        client._client = httpx.Client(transport=httpx.MockTransport(handler))
        return client

    monkeypatch.setattr(ingest_taipei, "build_client", build_client)
    with caplog.at_level(logging.WARNING):
        first = ingest_taipei.run_ingestion()
        resumed = ingest_taipei.run_ingestion()
    assert first["enrichment"] == {"enriched_count": 2, "failed_count": 0}
    assert resumed["enrichment"] == {"enriched_count": 0, "failed_count": 0}
    if discovery_budget == 3:
        assert first["discovery"] == {"budget_exhausted": True, "requests_made": 2}
        assert resumed["discovery"] == {"budget_exhausted": True, "requests_made": 0}
        assert all(record.discovery_request_budget == 3 for record in caplog.records)
        assert all(record.discovery_requests_used == 3 for record in caplog.records)
        assert len(caplog.records) == 2
    else:
        assert first["discovery"] == {"requests_made": 3, "candidates_seen": 2, "stopped_at_target": False}
        assert resumed["discovery"] == {"requests_made": 0, "candidates_seen": 0, "stopped_at_target": False}
        assert not caplog.records
    store = DuckDbStore(database_path)
    try:
        assert store.enriched_restaurant_count() == 2
        assert store.api_call_count("text_search_ids_only") == discovery_budget
        assert store.api_call_count("place_details_enterprise_no_atmosphere") == 2
    finally:
        store.close()

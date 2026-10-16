from pathlib import Path
import json
import sys

import duckdb
import httpx
import pytest

from restaurant_ingestion.clients.google_places import GooglePlacesClient
from restaurant_ingestion.models import PlaceDetails
from restaurant_ingestion.storage.duckdb_store import DuckDbStore
from scripts import hydrate_michelin_prices as script


def seed_restaurants(database_path: Path) -> None:
    store = DuckDbStore(database_path)
    try:
        store.create_schema()
        for place_id, is_michelin in (("missing-guide", True), ("missing-ordinary", False), ("priced-guide", True)):
            payload = {
                "id": place_id,
                "displayName": {"text": place_id},
                "location": {"latitude": 25.0, "longitude": 121.5},
            }
            store.upsert_restaurant(PlaceDetails.model_validate(payload), json.dumps(payload), place_id, is_michelin)
        store.upsert_michelin_metadata("missing-guide", "Guide restaurant", "Bib Gourmand", 0, True, None)
    finally:
        store.close()
    with duckdb.connect(str(database_path)) as connection:
        connection.execute("UPDATE restaurants SET price_range = '2000+' WHERE place_id = 'priced-guide'")


def test_refreshes_recent_missing_guide_prices_and_preserves_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database_path = tmp_path / "restaurants.duckdb"
    seed_restaurants(database_path)

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith('/missing-guide')
        assert 'priceRange' in request.headers['X-Goog-FieldMask'].split(',')
        return httpx.Response(200, json={
            "id": "missing-guide", "displayName": {"text": "Guide restaurant"},
            "location": {"latitude": 25.0, "longitude": 121.5},
            "priceRange": {"startPrice": {"currencyCode": "TWD", "units": "2000"}},
            "priceLevel": "PRICE_LEVEL_VERY_EXPENSIVE",
        })

    client = GooglePlacesClient("test-only", "TEST_ONLY", "en", "TW", 1.0, 0, 0.0, 0.0)
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(script, "DATABASE_PATH", database_path)
    monkeypatch.setattr(script, "build_client", lambda: client)
    monkeypatch.setattr(sys, "argv", ["hydrate_michelin_prices"])
    script.main()

    with duckdb.connect(str(database_path), read_only=True) as connection:
        assert connection.execute("SELECT place_id, price_range, is_michelin FROM restaurants ORDER BY place_id").fetchall() == [
            ("missing-guide", "2000+", True),
            ("missing-ordinary", None, False),
            ("priced-guide", "2000+", True),
        ]
        assert connection.execute("SELECT michelin_category FROM restaurant_michelin WHERE place_id = 'missing-guide'").fetchone() == ("Bib Gourmand",)
        assert connection.execute("SELECT COUNT(*) FROM api_call_log").fetchone() == (1,)


def test_dry_run_selects_only_missing_michelin_ranges_without_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = tmp_path / "restaurants.duckdb"
    seed_restaurants(database_path)

    def reject_authentication() -> GooglePlacesClient:
        raise AssertionError("Dry run must not initialize authentication")

    monkeypatch.setattr(script, "DATABASE_PATH", database_path)
    monkeypatch.setattr(script, "build_client", reject_authentication)
    monkeypatch.setattr(sys, "argv", ["hydrate_michelin_prices", "--dry-run"])
    script.main()

    assert json.loads(capsys.readouterr().out)["selected"] == 1
    with duckdb.connect(str(database_path), read_only=True) as connection:
        assert connection.execute("SELECT price_range FROM restaurants WHERE place_id = 'missing-guide'").fetchone() == (None,)
        assert connection.execute("SELECT COUNT(*) FROM api_call_log").fetchone() == (0,)

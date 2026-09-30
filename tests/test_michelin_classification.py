from pathlib import Path

import duckdb
import pytest

from restaurant_ingestion.config import PLACE_DETAILS_FIELD_MASK
from restaurant_ingestion.ingestion import michelin
from restaurant_ingestion.ingestion.michelin_place_ids import QUERY_MODE_SIMPLE, taipei_source_rows, text_query_for_row
from restaurant_ingestion.models import PlaceDetails
from restaurant_ingestion.storage.duckdb_store import DuckDbStore


def test_new_taipei_guide_rows_are_included_and_queries_keep_their_city() -> None:
    rows = (
        {"Name": "Garden", "Address": "22 Zhiguang Street", "Location": "New Taipei, Taiwan"},
        {"Name": "Taipei Place", "Address": "1 Road", "Location": "Taipei, Taiwan"},
        {"Name": "Outside", "Address": "2 Road", "Location": "Taoyuan, Taiwan"},
    )
    selected = taipei_source_rows(rows)
    assert [row["Name"] for row in selected] == ["Garden", "Taipei Place"]
    assert text_query_for_row(rows[0], QUERY_MODE_SIMPLE) == "Garden, New Taipei, Taiwan"


def seed_cached_ordinary(path: Path) -> None:
    store = DuckDbStore(path)
    try:
        store.create_schema()
        place = PlaceDetails.model_validate({
            "id": "existing", "displayName": {"text": "Google name"},
            "formattedAddress": "22 Zhiguang Street, New Taipei City, Taiwan",
            "location": {"latitude": 25.0, "longitude": 121.5},
            "rating": 4.5, "userRatingCount": 150,
        })
        store.upsert_restaurant(place, '{"id":"existing"}', "hash", False)
    finally:
        store.close()


def test_cached_google_record_receives_classification_without_api_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "restaurants.duckdb"
    seed_cached_ordinary(path)
    guide = tmp_path / "guide.csv"
    guide.write_text("place_id,name,michelin_category,year\nexisting,Guide name,Bib Gourmand,2025\n")
    with duckdb.connect(str(path), read_only=True) as connection:
        before = connection.execute("SELECT name,rating,review_count,last_updated FROM restaurants WHERE place_id='existing'").fetchone()

    api_calls: list[bool] = []

    def refuse_api(*args: object, **kwargs: object) -> None:
        api_calls.append(True)
        raise AssertionError("Classification must not fetch cached Google details")

    monkeypatch.setattr(michelin.GooglePlacesClient, "place_details", refuse_api)
    store = DuckDbStore(path)
    try:
        client = object.__new__(michelin.GooglePlacesClient)
        result = michelin.run_michelin_csv_enrichment(guide, client, store, 90, 0, PLACE_DETAILS_FIELD_MASK, 100)
        assert result["enriched_count"] == 0
        assert result["failed_count"] == 0
        assert result["cached_place_ids"] == 1
        assert api_calls == []
    finally:
        store.close()
    with duckdb.connect(str(path), read_only=True) as connection:
        assert connection.execute("SELECT is_michelin FROM restaurants WHERE place_id='existing'").fetchone() == (True,)
        assert connection.execute("SELECT michelin_category,michelin_stars,michelin_bib_gourmand,guide_year FROM restaurant_michelin WHERE place_id='existing'").fetchone() == ("Bib Gourmand", 0, True, 2025)
        assert connection.execute("SELECT name,rating,review_count,last_updated FROM restaurants WHERE place_id='existing'").fetchone() == before


def test_classification_only_sync_is_idempotent_and_does_not_create_missing_restaurants(tmp_path: Path) -> None:
    path = tmp_path / "restaurants.duckdb"
    seed_cached_ordinary(path)
    guide = tmp_path / "guide.csv"
    guide.write_text("place_id,name,michelin_category\nexisting,Guide name,Selected Restaurants\nabsent,Missing,1 Star\n")
    store = DuckDbStore(path)
    try:
        assert michelin.classify_existing_michelin_csv(guide, store) == 1
        with duckdb.connect(str(path)) as connection:
            first = connection.execute("SELECT place_id,michelin_category,last_updated FROM restaurant_michelin ORDER BY place_id").fetchall()
        assert michelin.classify_existing_michelin_csv(guide, store) == 1
        with duckdb.connect(str(path)) as connection:
            assert connection.execute("SELECT place_id,michelin_category,last_updated FROM restaurant_michelin ORDER BY place_id").fetchall() == first
            assert connection.execute("SELECT count(*) FROM restaurants").fetchone() == (1,)
    finally:
        store.close()

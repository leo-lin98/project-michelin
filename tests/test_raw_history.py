from datetime import datetime
from pathlib import Path

import duckdb

from restaurant_ingestion.storage.duckdb_store import DuckDbStore


def test_refresh_retains_prior_raw_response_and_replay_does_not_duplicate(tmp_path: Path) -> None:
    path = tmp_path / "restaurants.duckdb"
    store = DuckDbStore(path)
    try:
        store.create_schema()
        store.upsert_raw_response("place-1", '{"rating":4.0}', "hash-a", datetime(2026, 6, 25))
        store.upsert_raw_response("place-1", '{"rating":4.5}', "hash-b", datetime(2026, 9, 11))
        store.upsert_raw_response("place-1", '{"rating":4.5}', "hash-b", datetime(2026, 9, 11))
    finally:
        store.close()
    with duckdb.connect(str(path), read_only=True) as connection:
        assert connection.execute("SELECT response_hash, fetched_at FROM place_raw_response_history ORDER BY fetched_at").fetchall() == [
            ("hash-a", datetime(2026, 6, 25)), ("hash-b", datetime(2026, 9, 11)),
        ]
        assert connection.execute("SELECT response_hash FROM place_raw_responses").fetchone() == ("hash-b",)


def test_history_migration_seeds_surviving_response_once(tmp_path: Path) -> None:
    path = tmp_path / "restaurants.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE TABLE place_raw_responses (place_id VARCHAR PRIMARY KEY, response_json JSON NOT NULL, response_hash VARCHAR NOT NULL, fetched_at TIMESTAMP NOT NULL)")
        connection.execute("INSERT INTO place_raw_responses VALUES ('old','{}','old-hash',?)", [datetime(2026, 6, 25)])
    store = DuckDbStore(path)
    try:
        store.create_schema()
        store.create_schema()
        store.upsert_raw_response("old", '{"rating":4}', "new-hash", datetime(2026, 9, 11))
    finally:
        store.close()
    with duckdb.connect(str(path), read_only=True) as connection:
        assert connection.execute("SELECT response_hash FROM place_raw_response_history ORDER BY fetched_at").fetchall() == [("old-hash",), ("new-hash",)]

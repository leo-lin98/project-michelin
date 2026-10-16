import argparse
import json
from pathlib import Path

import duckdb

from restaurant_ingestion.config import (
    DATABASE_PATH,
    GOOGLE_MAX_RETRIES,
    PRICE_BACKFILL_PROGRESS_INTERVAL,
    PRICE_BACKFILL_SAMPLE_SIZE,
)
from restaurant_ingestion.storage.duckdb_store import DuckDbStore
from scripts.backfill_price_fields import backfill_prices, build_client


def missing_michelin_price_state(database_path: Path) -> list[tuple[str, str | None]]:
    """Select Guide rows missing price_range regardless of their last refresh date."""
    with duckdb.connect(str(database_path), read_only=True) as connection:
        rows = connection.execute(
            """SELECT place_id, price_level
               FROM restaurants
               WHERE is_michelin = TRUE AND price_range IS NULL
               ORDER BY last_updated, place_id
               LIMIT ?""",
            [PRICE_BACKFILL_SAMPLE_SIZE],
        ).fetchall()
    return [(str(place_id), None if price_level is None else str(price_level)) for place_id, price_level in rows]


def main() -> None:
    """Refresh missing Michelin price ranges once, bypassing the retry window."""
    parser = argparse.ArgumentParser(description="Refresh Michelin restaurants missing price_range, ignoring the cache window.")
    parser.add_argument("--dry-run", action="store_true", help="Show the selection and request budget without API calls or writes.")
    args = parser.parse_args()
    price_state = missing_michelin_price_state(DATABASE_PATH)
    request_budget = len(price_state) * (GOOGLE_MAX_RETRIES + 1)
    print(json.dumps({"selected": len(price_state), "request_budget": request_budget}, indent=2))
    if args.dry_run or not price_state:
        return

    store = DuckDbStore(DATABASE_PATH)
    try:
        store.create_schema()
        client = build_client()
        try:
            counts = backfill_prices(client, store, price_state, request_budget, PRICE_BACKFILL_PROGRESS_INTERVAL)
        finally:
            client.close()
    finally:
        store.close()
    print(json.dumps(counts, indent=2))
    if counts["failed"] or counts["budget_exhausted"]:
        raise RuntimeError(f"Michelin price hydration was incomplete: {counts['failed']} failed; budget_exhausted={counts['budget_exhausted']}")


if __name__ == "__main__":
    main()

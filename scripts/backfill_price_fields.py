"""Refresh price fields for a seeded random sample of already-ingested restaurants.

Re-fetches Place Details for PRICE_BACKFILL_SAMPLE_SIZE restaurants and writes the
result through DuckDbStore.upsert_restaurant, which replaces the stale
`place_raw_responses` row, the category/hours child rows, and the `restaurants`
columns -- including `price_level` and the new `price_range`.

By default the batch is a seeded random sample across all restaurants, which spans
existing price tiers and can show whether `priceRange` varies within a tier or merely
restates it. `--missing` instead works through the rows that still have no price,
stalest first, for filling coverage gaps.

Every field comes back in the one Place Details call the pipeline already pays for
at the Enterprise tier, so recovering `price_level` costs nothing beyond the
`price_range` fetch itself.

Run with Doppler so GOOGLE_MAPS_API_KEY is injected into the environment:

    doppler run -- uv run python scripts/backfill_price_fields.py --dry-run
    doppler run -- uv run python scripts/backfill_price_fields.py
    doppler run -- uv run python scripts/backfill_price_fields.py --missing
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json

from restaurant_ingestion.config import (
    DATABASE_PATH,
    PRICE_BACKFILL_PROGRESS_INTERVAL,
    PRICE_BACKFILL_REQUEST_BUDGET,
    PRICE_BACKFILL_RETRY_AFTER_DAYS,
    PRICE_BACKFILL_SAMPLE_SEED,
    PRICE_BACKFILL_SAMPLE_SIZE,
)
from restaurant_ingestion.ingestion.price_backfill import backfill_prices, build_client, price_outcome
from restaurant_ingestion.storage.duckdb_store import DuckDbStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh price_level and price_range for a batch of restaurants.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--missing",
        action="store_true",
        help=(
            "Target restaurants still missing price_level or price_range, stalest first, "
            f"skipping any refreshed in the last {PRICE_BACKFILL_RETRY_AFTER_DAYS} days. "
            "Without this flag a seeded random sample of all restaurants is used."
        ),
    )
    args = parser.parse_args()

    store = DuckDbStore(DATABASE_PATH)
    try:
        store.create_schema()
        price_state = (
            store.missing_price_state(PRICE_BACKFILL_SAMPLE_SIZE, PRICE_BACKFILL_RETRY_AFTER_DAYS)
            if args.missing
            else store.sample_restaurant_price_state(PRICE_BACKFILL_SAMPLE_SIZE, PRICE_BACKFILL_SAMPLE_SEED)
        )
        if not price_state:
            raise RuntimeError(
                "No restaurants selected: every row missing a price field was refreshed within the last "
                f"{PRICE_BACKFILL_RETRY_AFTER_DAYS} days"
            )
        if args.dry_run:
            missing_price_level = sum(1 for _place_id, price_level in price_state if price_level is None)
            print(
                json.dumps(
                    {
                        "mode": "missing" if args.missing else "random-sample",
                        "selected": len(price_state),
                        "missing_price_level": missing_price_level,
                        "request_budget": PRICE_BACKFILL_REQUEST_BUDGET,
                    },
                    indent=2,
                )
            )
            return

        client = build_client()
        try:
            counts = backfill_prices(
                client,
                store,
                price_state,
                PRICE_BACKFILL_REQUEST_BUDGET,
                PRICE_BACKFILL_PROGRESS_INTERVAL,
            )
        finally:
            client.close()
        print(json.dumps(counts, indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()

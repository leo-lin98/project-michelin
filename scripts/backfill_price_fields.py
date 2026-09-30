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
import hashlib
import json
import time

from restaurant_ingestion.clients.google_places import (
    BudgetExceededError,
    GooglePlacesClient,
    PlacesClientError,
    ensure_request_budget,
)
from restaurant_ingestion.config import (
    DATABASE_PATH,
    GOOGLE_API_KEY_ENV_VAR,
    GOOGLE_LANGUAGE_CODE,
    GOOGLE_MAX_RETRIES,
    GOOGLE_REGION_CODE,
    GOOGLE_REQUEST_TIMEOUT_SECONDS,
    GOOGLE_RETRY_INITIAL_BACKOFF_SECONDS,
    GOOGLE_RETRY_MAX_BACKOFF_SECONDS,
    PLACE_DETAILS_FIELD_MASK,
    PRICE_BACKFILL_PROGRESS_INTERVAL,
    PRICE_BACKFILL_REQUEST_BUDGET,
    PRICE_BACKFILL_RETRY_AFTER_DAYS,
    PRICE_BACKFILL_SAMPLE_SEED,
    PRICE_BACKFILL_SAMPLE_SIZE,
)
from restaurant_ingestion.ingestion.checkpoints import FAILED, RUNNING, SUCCEEDED, detail_checkpoint
from restaurant_ingestion.ingestion.enrichment import log_detail_attempts
from restaurant_ingestion.ingestion.progress import print_request_progress
from restaurant_ingestion.models import PriceRange
from restaurant_ingestion.storage.duckdb_store import DuckDbStore, format_price_range

# Sampled ids already exist in `restaurants`, and upsert_restaurant ORs this flag
# into the stored value, so passing False cannot clear an existing Michelin row.
NOT_MICHELIN = False


def backfill_prices(
    client: GooglePlacesClient,
    store: DuckDbStore,
    price_state: list[tuple[str, str | None]],
    request_budget: int,
    progress_interval: int,
) -> dict[str, int]:
    """Re-fetch each sampled place and report what the refresh changed."""
    used_requests = 0
    processed = 0
    started_at = time.monotonic()
    print_request_progress("price-backfill", processed, len(price_state), 0.0)
    counts = {
        "sampled": len(price_state),
        "refreshed": 0,
        "failed": 0,
        "price_level_recovered": 0,
        "price_level_lost": 0,
        "price_range_filled": 0,
        "price_range_partial": 0,
        "price_range_absent": 0,
        "budget_exhausted": 0,
    }

    for place_id, previous_price_level in price_state:
        try:
            ensure_request_budget(used_requests, request_budget, 1)
        except BudgetExceededError:
            counts["budget_exhausted"] = 1
            break

        checkpoint_id = detail_checkpoint(place_id)
        payload_json = json.dumps({"place_id": place_id}, sort_keys=True, separators=(",", ":"))
        store.upsert_checkpoint(checkpoint_id, "place_details", RUNNING, payload_json, None, None)
        try:
            place, raw_response, attempts = client.place_details(
                place_id,
                PLACE_DETAILS_FIELD_MASK,
                request_budget - used_requests,
            )
            log_detail_attempts(store, place_id, checkpoint_id, attempts, used_requests)
            used_requests += len(attempts)
            raw_json = json.dumps(raw_response, sort_keys=True, separators=(",", ":"))
            response_hash = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
            store.upsert_restaurant(place, raw_json, response_hash, NOT_MICHELIN)
            store.upsert_checkpoint(checkpoint_id, "place_details", SUCCEEDED, payload_json, None, None)
            counts["refreshed"] += 1
            for outcome in price_outcome(previous_price_level, place.priceLevel, place.priceRange):
                counts[outcome] += 1
        except PlacesClientError as exc:
            log_detail_attempts(store, place_id, checkpoint_id, exc.attempts, used_requests)
            used_requests += len(exc.attempts)
            store.upsert_checkpoint(checkpoint_id, "place_details", FAILED, payload_json, None, str(exc))
            counts["failed"] += 1
        except Exception as exc:
            store.upsert_checkpoint(checkpoint_id, "place_details", FAILED, payload_json, None, str(exc))
            counts["failed"] += 1

        processed += 1
        if processed % progress_interval == 0:
            print_request_progress("price-backfill", processed, len(price_state), time.monotonic() - started_at)

    counts["requests_used"] = used_requests
    counts["elapsed_seconds"] = round(time.monotonic() - started_at)
    return counts


def price_outcome(
    previous_price_level: str | None,
    current_price_level: str | None,
    price_range: PriceRange | None,
) -> tuple[str, ...]:
    """Classify how one refreshed row changed, as the counter keys it should increment."""
    outcomes: list[str] = []
    if previous_price_level is None and current_price_level is not None:
        outcomes.append("price_level_recovered")
    if previous_price_level is not None and current_price_level is None:
        outcomes.append("price_level_lost")
    if format_price_range(price_range) is not None:
        outcomes.append("price_range_filled")
    elif price_range is not None:
        outcomes.append("price_range_partial")
    else:
        outcomes.append("price_range_absent")
    return tuple(outcomes)


def build_client() -> GooglePlacesClient:
    return GooglePlacesClient.from_environment(
        GOOGLE_API_KEY_ENV_VAR,
        GOOGLE_LANGUAGE_CODE,
        GOOGLE_REGION_CODE,
        GOOGLE_REQUEST_TIMEOUT_SECONDS,
        GOOGLE_MAX_RETRIES,
        GOOGLE_RETRY_INITIAL_BACKOFF_SECONDS,
        GOOGLE_RETRY_MAX_BACKOFF_SECONDS,
    )


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

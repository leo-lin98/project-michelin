from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

from restaurant_ingestion.clients.google_places import GooglePlacesClient
from restaurant_ingestion.config import (
    DATABASE_PATH,
    DETAILS_REFRESH_TTL_DAYS,
    DETAILS_REQUEST_BUDGET,
    GOOGLE_API_KEY_ENV_VAR,
    GOOGLE_LANGUAGE_CODE,
    GOOGLE_MAX_RETRIES,
    GOOGLE_REGION_CODE,
    GOOGLE_REQUEST_TIMEOUT_SECONDS,
    GOOGLE_RETRY_INITIAL_BACKOFF_SECONDS,
    GOOGLE_RETRY_MAX_BACKOFF_SECONDS,
    PLACE_DETAILS_FIELD_MASK,
    PROGRESS_LOG_INTERVAL,
)
from restaurant_ingestion.ingestion.enrichment import plan_restaurant_refresh, run_restaurant_refresh
from restaurant_ingestion.storage.duckdb_store import DuckDbStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh stale Google Places restaurant details.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(plan_refresh())
        return

    print(run_refresh())


def plan_refresh() -> dict[str, int]:
    store = DuckDbStore(DATABASE_PATH)
    try:
        store.create_schema()
        return plan_restaurant_refresh(store, DETAILS_REFRESH_TTL_DAYS, DETAILS_REQUEST_BUDGET)
    finally:
        store.close()


def run_refresh() -> dict[str, int]:
    store = DuckDbStore(DATABASE_PATH)
    client = build_client()
    try:
        store.create_schema()
        return run_restaurant_refresh(
            client,
            store,
            DETAILS_REFRESH_TTL_DAYS,
            DETAILS_REQUEST_BUDGET,
            PLACE_DETAILS_FIELD_MASK,
            PROGRESS_LOG_INTERVAL,
        )
    finally:
        client.close()
        store.close()


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


if __name__ == "__main__":
    main()

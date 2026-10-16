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
from restaurant_ingestion.ingestion.michelin import classify_existing_michelin_csv, plan_michelin_csv_enrichment, run_michelin_csv_enrichment
from restaurant_ingestion.storage.duckdb_store import DuckDbStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich Michelin guide rows through the Google Place Details pipeline.")
    parser.add_argument("--guide-csv", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--classifications-only", action="store_true", help="Apply CSV labels to existing database rows without Google API calls.")
    args = parser.parse_args()
    guide_csv_path = Path(args.guide_csv)

    if args.dry_run:
        print(plan_michelin_enrichment(guide_csv_path))
        return

    if args.classifications_only:
        store = DuckDbStore(DATABASE_PATH)
        try:
            print({"classified_count": classify_existing_michelin_csv(guide_csv_path, store)})
        finally:
            store.close()
        return

    print(run_michelin_enrichment(guide_csv_path))


def run_michelin_enrichment(guide_csv_path: Path) -> dict[str, int]:
    store = DuckDbStore(DATABASE_PATH)
    client = build_client()
    try:
        store.create_schema()
        return run_michelin_csv_enrichment(
            guide_csv_path,
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


def plan_michelin_enrichment(guide_csv_path: Path) -> dict[str, int]:
    store = DuckDbStore(DATABASE_PATH)
    try:
        store.create_schema()
        return plan_michelin_csv_enrichment(guide_csv_path, store, DETAILS_REFRESH_TTL_DAYS)
    finally:
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

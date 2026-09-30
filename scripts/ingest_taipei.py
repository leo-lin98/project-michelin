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
    DISCOVERY_QUERIES,
    DISCOVERY_REQUEST_BUDGET,
    GOOGLE_API_KEY_ENV_VAR,
    GOOGLE_LANGUAGE_CODE,
    GOOGLE_MAX_RETRIES,
    GOOGLE_REGION_CODE,
    GOOGLE_REQUEST_TIMEOUT_SECONDS,
    GOOGLE_RETRY_INITIAL_BACKOFF_SECONDS,
    GOOGLE_RETRY_MAX_BACKOFF_SECONDS,
    ORDINARY_RESTAURANT_TARGET,
    PLACE_DETAILS_FIELD_MASK,
    PROGRESS_LOG_INTERVAL,
    TAIPEI_GRID_EAST,
    TAIPEI_GRID_NORTH,
    TAIPEI_GRID_SOUTH,
    TAIPEI_GRID_TILE_SIZE_DEGREES,
    TAIPEI_GRID_WEST,
    TARGET_ENRICHED_RESTAURANTS,
)
from restaurant_ingestion.ingestion.enrichment import plan_candidate_enrichment, run_candidate_enrichment
from restaurant_ingestion.ingestion.grid_search import plan_taipei_discovery, run_taipei_discovery
from restaurant_ingestion.storage.duckdb_store import DuckDbStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and enrich Taipei restaurants with Google Places API New.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(plan_ingestion())
        return

    print(run_ingestion())


def plan_ingestion() -> dict[str, object]:
    store = DuckDbStore(DATABASE_PATH)
    try:
        store.create_schema()
        return {
            "discovery": plan_taipei_discovery(
                TAIPEI_GRID_SOUTH,
                TAIPEI_GRID_WEST,
                TAIPEI_GRID_NORTH,
                TAIPEI_GRID_EAST,
                TAIPEI_GRID_TILE_SIZE_DEGREES,
                DISCOVERY_QUERIES,
            ),
            "enrichment": plan_candidate_enrichment(store, TARGET_ENRICHED_RESTAURANTS, DETAILS_REFRESH_TTL_DAYS),
            "details_request_budget": DETAILS_REQUEST_BUDGET,
            "discovery_request_budget": DISCOVERY_REQUEST_BUDGET,
        }
    finally:
        store.close()


def run_ingestion() -> dict[str, object]:
    store = DuckDbStore(DATABASE_PATH)
    client = build_client()
    try:
        store.create_schema()
        discovery = run_taipei_discovery(
            client,
            store,
            TAIPEI_GRID_SOUTH,
            TAIPEI_GRID_WEST,
            TAIPEI_GRID_NORTH,
            TAIPEI_GRID_EAST,
            TAIPEI_GRID_TILE_SIZE_DEGREES,
            DISCOVERY_QUERIES,
            ORDINARY_RESTAURANT_TARGET,
            DISCOVERY_REQUEST_BUDGET,
            PROGRESS_LOG_INTERVAL,
        )
        enrichment = run_candidate_enrichment(
            client,
            store,
            TARGET_ENRICHED_RESTAURANTS,
            DETAILS_REFRESH_TTL_DAYS,
            DETAILS_REQUEST_BUDGET,
            PLACE_DETAILS_FIELD_MASK,
            PROGRESS_LOG_INTERVAL,
        )
        return {"discovery": discovery, "enrichment": enrichment}
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

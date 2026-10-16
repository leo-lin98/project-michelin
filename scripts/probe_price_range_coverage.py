"""One-off probe: measure `priceRange` coverage on restaurants that lack `priceLevel`.

Samples PROBE_SAMPLE_SIZE restaurants whose `price_level` is NULL and makes exactly
one Place Details (New) call each, to test the hypothesis in
`scripts/inspect_place_details.py` that a currency range exists where a price level
does not. `priceRange` is an Enterprise SKU field, the same tier the pipeline already
pays for via `priceLevel`/`rating`/`userRatingCount`, so this adds no incremental
per-call cost -- and the sample sits inside the 1,000/month free Enterprise quota.

Read-only: nothing is written to the database.

Run with Doppler so GOOGLE_MAPS_API_KEY is injected into the environment:

    doppler run -- uv run python scripts/probe_price_range_coverage.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import json

import duckdb

from restaurant_ingestion.clients.google_places import GooglePlacesClient
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
)

PROBE_SAMPLE_SIZE = 5
PROBE_SAMPLE_SEED = 42
# Match scripts/inspect_place_details.py: one call, no retry budget, so a probe can
# never quietly fan out into extra billable requests.
SINGLE_REQUEST_BUDGET = 1


def sample_place_ids_without_price_level(database_path: Path, sample_size: int, seed: int) -> tuple[tuple[str, str], ...]:
    """Deterministically sample (place_id, name) for restaurants missing `price_level`."""
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        # The filter must be materialised in a subquery: applied as a sibling of WHERE,
        # DuckDB samples the base table first and then filters, yielding fewer rows.
        rows = connection.execute(
            f"""
            SELECT place_id, name
            FROM (
                SELECT place_id, name
                FROM restaurants
                WHERE price_level IS NULL
            )
            USING SAMPLE reservoir({sample_size} ROWS) REPEATABLE ({seed})
            """
        ).fetchall()
    finally:
        connection.close()
    return tuple((str(row[0]), str(row[1])) for row in rows)


def price_range_summary(raw_response: dict[str, object]) -> dict[str, object]:
    """Extract the priceLevel/priceRange pair that the probe is measuring."""
    return {
        "price_level": raw_response.get("priceLevel"),
        "price_range": raw_response.get("priceRange"),
    }


def coverage_counts(results: tuple[dict[str, object], ...]) -> dict[str, int]:
    """Count how many probed places returned a usable priceRange."""
    return {
        "probed": len(results),
        "with_price_range": sum(1 for result in results if result["price_range"] is not None),
        "with_price_level": sum(1 for result in results if result["price_level"] is not None),
    }


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
    sampled = sample_place_ids_without_price_level(DATABASE_PATH, PROBE_SAMPLE_SIZE, PROBE_SAMPLE_SEED)
    if not sampled:
        raise RuntimeError(f"No restaurants with NULL price_level found in {DATABASE_PATH}")

    client = build_client()
    results: list[dict[str, object]] = []
    try:
        for place_id, name in sampled:
            _place, raw_response, attempts = client.place_details(
                place_id,
                PLACE_DETAILS_FIELD_MASK,
                SINGLE_REQUEST_BUDGET,
            )
            results.append(
                {
                    "place_id": place_id,
                    "db_name": name,
                    "http_attempts": list(attempts),
                    **price_range_summary(raw_response),
                }
            )
    finally:
        client.close()

    output = {
        "field_mask": list(PLACE_DETAILS_FIELD_MASK),
        "sample_seed": PROBE_SAMPLE_SEED,
        "coverage": coverage_counts(tuple(results)),
        "results": results,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

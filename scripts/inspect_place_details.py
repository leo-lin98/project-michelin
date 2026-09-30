"""One-off probe: fetch a single Google Place and show the DuckDB-pertinent columns.

Makes exactly ONE Place Details (New) call (request budget = 1) and prints the
fields that would be persisted to the DuckDB `restaurants` table plus the
`restaurant_categories` / `restaurant_hours` child tables. Nothing is written to
the database.

Run with Doppler so GOOGLE_MAPS_API_KEY is injected into the environment:

    doppler run -- uv run python scripts/inspect_place_details.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import json

from restaurant_ingestion.clients.google_places import GooglePlacesClient
from restaurant_ingestion.config import (
    GOOGLE_API_KEY_ENV_VAR,
    GOOGLE_LANGUAGE_CODE,
    GOOGLE_MAX_RETRIES,
    GOOGLE_REGION_CODE,
    GOOGLE_REQUEST_TIMEOUT_SECONDS,
    GOOGLE_RETRY_INITIAL_BACKOFF_SECONDS,
    GOOGLE_RETRY_MAX_BACKOFF_SECONDS,
    PLACE_DETAILS_FIELD_MASK,
)
from restaurant_ingestion.models import PlaceDetails, RegularOpeningPeriod
from restaurant_ingestion.storage.duckdb_store import point_time

PLACE_ID = "ChIJ1ec_XJEDaDQRQyQjaWjSdQg"
SINGLE_REQUEST_BUDGET = 1

# Probe-only mask: the shared pipeline mask plus priceRange, so we can see whether
# a currency range exists where priceLevel does not. priceRange is an Enterprise
# SKU field (same tier as priceLevel), so this adds no incremental cost.
PROBE_FIELD_MASK = PLACE_DETAILS_FIELD_MASK + ("priceRange",)


def restaurant_columns(place: PlaceDetails) -> dict[str, object]:
    """Mirror DuckDbStore.upsert_restaurant's mapping onto the `restaurants` row."""
    return {
        "place_id": place.id,
        "name": place.displayName.text,
        "address": place.formattedAddress,
        "latitude": place.location.latitude,
        "longitude": place.location.longitude,
        "rating": place.rating,
        "review_count": place.userRatingCount,
        "price_level": place.priceLevel,
        "website": None,  # not in the details field mask; stored NULL
        "phone": None,  # not in the details field mask; stored NULL
        "google_maps_url": str(place.googleMapsUri) if place.googleMapsUri is not None else None,
        "business_status": place.businessStatus,
        "primary_category": place.primaryType,
    }


def hours_rows(periods: tuple[RegularOpeningPeriod, ...]) -> list[dict[str, object]]:
    """Mirror DuckDbStore.replace_hours' mapping onto `restaurant_hours` rows."""
    return [
        {
            "day_of_week": period.open.day,
            "open_time": point_time(period.open),
            "close_time": point_time(period.close) if period.close is not None else "unknown",
        }
        for period in periods
    ]


def main() -> None:
    client = GooglePlacesClient.from_environment(
        GOOGLE_API_KEY_ENV_VAR,
        GOOGLE_LANGUAGE_CODE,
        GOOGLE_REGION_CODE,
        GOOGLE_REQUEST_TIMEOUT_SECONDS,
        GOOGLE_MAX_RETRIES,
        GOOGLE_RETRY_INITIAL_BACKOFF_SECONDS,
        GOOGLE_RETRY_MAX_BACKOFF_SECONDS,
    )
    try:
        place, raw_response, attempts = client.place_details(
            PLACE_ID,
            PROBE_FIELD_MASK,
            SINGLE_REQUEST_BUDGET,
        )
    finally:
        client.close()

    periods = place.regularOpeningHours.periods if place.regularOpeningHours is not None else ()

    output = {
        "field_mask": list(PROBE_FIELD_MASK),
        "http_attempts": list(attempts),
        "restaurants_row": restaurant_columns(place),
        "price_range": raw_response.get("priceRange"),
        "restaurant_categories": list(place.types),
        "restaurant_hours": hours_rows(periods),
        "raw_response": raw_response,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

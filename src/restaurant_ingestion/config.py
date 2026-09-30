"""Runtime constants for restaurant ingestion."""

from pathlib import Path

DATABASE_PATH = Path("data/restaurants.duckdb")
MICHELIN_PLACE_ID_WORK_DIR = Path("data/interim/michelin_place_ids")
MICHELIN_PLACE_ID_RESOLVED_DIR = MICHELIN_PLACE_ID_WORK_DIR / "resolved"
MICHELIN_PLACE_ID_UNRESOLVED_DIR = MICHELIN_PLACE_ID_WORK_DIR / "unresolved"
MICHELIN_PLACE_ID_DATASET_CSV = MICHELIN_PLACE_ID_WORK_DIR / "michelin_taipei_place_ids.csv"

GOOGLE_API_KEY_ENV_VAR = "GOOGLE_MAPS_API_KEY"
GOOGLE_LANGUAGE_CODE = "zh-TW"
MICHELIN_PLACE_ID_LANGUAGE_CODE = "en"
GOOGLE_REGION_CODE = "TW"
GOOGLE_REQUEST_TIMEOUT_SECONDS = 30.0
GOOGLE_MAX_RETRIES = 3
GOOGLE_RETRY_INITIAL_BACKOFF_SECONDS = 1.0
GOOGLE_RETRY_MAX_BACKOFF_SECONDS = 16.0
PLACE_ID_LLM_MODEL = "gemini-2.5-flash"
PLACE_ID_LLM_PROJECT_ENV_VAR = "GOOGLE_CLOUD_PROJECT"
PLACE_ID_LLM_REGION_ENV_VAR = "GOOGLE_CLOUD_REGION"
PLACE_ID_LLM_USE_VERTEX_ENV_VAR = "GOOGLE_GENAI_USE_VERTEXAI"
PLACE_ID_LLM_MAX_RETRIES = 2
PLACE_ID_LLM_RETRY_BACKOFF_SECONDS = 1.0

DISCOVERY_REQUEST_BUDGET = 4_000
DETAILS_REQUEST_BUDGET = 6_564
MICHELIN_PLACE_ID_REQUEST_BUDGET = 250
DETAILS_REFRESH_TTL_DAYS = 90
PROGRESS_LOG_INTERVAL = 500

# The price backfill carries its own budget because DETAILS_REQUEST_BUDGET is a
# lifetime odometer against api_call_log, already nearly spent by the initial load.
PRICE_BACKFILL_SAMPLE_SIZE = 1_000
PRICE_BACKFILL_SAMPLE_SEED = 42
PRICE_BACKFILL_REQUEST_BUDGET = 1_000
PRICE_BACKFILL_PROGRESS_INTERVAL = 100
# A --missing run skips rows refreshed inside this window so it cannot re-spend on
# rows a recent run already confirmed Google has no price for. They become eligible
# again afterwards: Google has been observed adding price data within ~34 days.
PRICE_BACKFILL_RETRY_AFTER_DAYS = 30

TARGET_ENRICHED_RESTAURANTS = 3_000
ORDINARY_RESTAURANT_TARGET = 3_000

TAIPEI_GRID_SOUTH = 24.9600
TAIPEI_GRID_WEST = 121.4450
TAIPEI_GRID_NORTH = 25.2100
TAIPEI_GRID_EAST = 121.6650
TAIPEI_GRID_TILE_SIZE_DEGREES = 0.01

DISCOVERY_QUERIES = (
    "restaurant",
    "taiwanese restaurant",
    "japanese restaurant",
    "hot pot restaurant",
    "noodle restaurant",
    "brunch restaurant",
)

PLACE_DETAILS_FIELD_MASK = (
    "id",
    "displayName",
    "formattedAddress",
    "location",
    "rating",
    "userRatingCount",
    "priceLevel",
    "priceRange",
    "businessStatus",
    "primaryType",
    "types",
    "googleMapsUri",
    "regularOpeningHours",
)

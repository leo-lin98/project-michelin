"""Look up the current Google name for each stored place_id and write it to a new column.

Reads the merged Michelin place_id CSV, calls Google Place Details (place_id -> displayName)
for every row, and writes the returned name into a `google_place_name` column so mismatches
against the Michelin name are eyeballable. Run under Doppler so the API key is in the env.
"""

from pathlib import Path
import csv
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from restaurant_ingestion.clients.google_places import GooglePlacesClient, PlacesClientError
from restaurant_ingestion.config import (
    GOOGLE_API_KEY_ENV_VAR,
    GOOGLE_LANGUAGE_CODE,
    GOOGLE_MAX_RETRIES,
    GOOGLE_REGION_CODE,
    GOOGLE_REQUEST_TIMEOUT_SECONDS,
    GOOGLE_RETRY_INITIAL_BACKOFF_SECONDS,
    GOOGLE_RETRY_MAX_BACKOFF_SECONDS,
)

CSV_PATH = ROOT / "data" / "interim" / "michelin_place_ids" / "michelin_taipei_place_ids.csv"
NAME_COLUMN = "google_place_name"
NAME_FIELD_MASK = ("id", "displayName", "location")


def main() -> None:
    fieldnames, rows = read_rows(CSV_PATH)
    if NAME_COLUMN not in fieldnames:
        fieldnames = [*fieldnames, NAME_COLUMN]

    client = build_client()
    failures = 0
    try:
        for index, row in enumerate(rows, start=1):
            place_id = row["place_id"]
            try:
                details, _, _ = client.place_details(
                    place_id,
                    NAME_FIELD_MASK,
                    request_budget_remaining=GOOGLE_MAX_RETRIES + 1,
                )
                row[NAME_COLUMN] = details.displayName.text
            except PlacesClientError as error:
                row[NAME_COLUMN] = ""
                failures += 1
                print(f"[{index}/{len(rows)}] LOOKUP FAILED place_id={place_id}: {error}", file=sys.stderr)
                continue
            print(f"[{index}/{len(rows)}] {row['name']}  ->  {row[NAME_COLUMN]}")
    finally:
        client.close()

    write_rows(CSV_PATH, fieldnames, rows)
    print(f"Wrote {NAME_COLUMN} for {len(rows) - failures}/{len(rows)} rows ({failures} failures) -> {CSV_PATH}")


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


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


if __name__ == "__main__":
    main()

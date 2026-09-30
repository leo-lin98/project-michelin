"""Michelin guide enrichment through the same Google Place Details path."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from restaurant_ingestion.clients.google_places import GooglePlacesClient, PlacesClientError, ensure_request_budget
from restaurant_ingestion.ingestion.progress import print_crossed_request_progress
from restaurant_ingestion.storage.duckdb_store import DuckDbStore


def plan_michelin_csv_enrichment(guide_csv_path: Path, store: DuckDbStore, details_refresh_ttl_days: int) -> dict[str, int]:
    rows = read_michelin_rows(guide_csv_path)
    missing_or_stale_place_ids = store.stale_or_missing_place_ids(place_ids_from_rows(rows), details_refresh_ttl_days)
    return {
        "michelin_rows": len(rows),
        "unique_michelin_place_ids": len(place_ids_from_rows(rows)),
        "michelin_details_requests": len(missing_or_stale_place_ids),
        "cached_place_ids": len(place_ids_from_rows(rows)) - len(missing_or_stale_place_ids),
    }


def run_michelin_csv_enrichment(
    guide_csv_path: Path,
    client: GooglePlacesClient,
    store: DuckDbStore,
    details_refresh_ttl_days: int,
    details_request_budget: int,
    place_details_field_mask: tuple[str, ...],
    progress_log_interval: int,
) -> dict[str, int]:
    rows = read_michelin_rows(guide_csv_path)
    rows_by_place_id = rows_by_unique_place_id(rows)
    classified_count = sync_existing_classifications(rows, store)
    place_ids_to_fetch = store.stale_or_missing_place_ids(tuple(rows_by_place_id.keys()), details_refresh_ttl_days)
    used_requests = store.api_call_count("place_details_enterprise_no_atmosphere")
    enriched_count = 0
    failed_count = 0

    for place_id in place_ids_to_fetch:
        row = rows_by_place_id[place_id]
        ensure_request_budget(used_requests, details_request_budget, 1)
        try:
            place, raw_response, attempts = client.place_details(
                place_id,
                place_details_field_mask,
                details_request_budget - used_requests,
            )
            previous_used_requests = used_requests
            log_michelin_attempts(store, place_id, attempts, previous_used_requests)
            used_requests += len(attempts)
            print_crossed_request_progress(
                "michelin_enrichment",
                previous_used_requests,
                used_requests,
                details_request_budget,
                progress_log_interval,
            )
            raw_json = json.dumps(raw_response, sort_keys=True, separators=(",", ":"))
            response_hash = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
            store.upsert_restaurant(place, raw_json, response_hash, True)
            store.upsert_michelin_metadata(
                place_id,
                str(row["name"]),
                str(row["category"]),
                int(row["stars"]),
                bool(row["bib_gourmand"]),
                row["year"] if isinstance(row["year"], int) else None,
            )
            enriched_count += 1
        except PlacesClientError as exc:
            previous_used_requests = used_requests
            log_michelin_attempts(store, place_id, exc.attempts, previous_used_requests)
            used_requests += len(exc.attempts)
            print_crossed_request_progress(
                "michelin_enrichment",
                previous_used_requests,
                used_requests,
                details_request_budget,
                progress_log_interval,
            )
            failed_count += 1
        except Exception:
            failed_count += 1
    return {
        "classified_count": classified_count,
        "enriched_count": enriched_count,
        "failed_count": failed_count,
        "cached_place_ids": len(rows_by_place_id) - len(place_ids_to_fetch),
    }


def classify_existing_michelin_csv(guide_csv_path: Path, store: DuckDbStore) -> int:
    """Apply verified CSV classifications to existing restaurants without fetching Google data."""
    return sync_existing_classifications(read_michelin_rows(guide_csv_path), store)


def sync_existing_classifications(rows: tuple[dict[str, object], ...], store: DuckDbStore) -> int:
    metadata = tuple(
        (place_id, str(row["name"]), str(row["category"]), int(row["stars"]), bool(row["bib_gourmand"]),
         row["year"] if isinstance(row["year"], int) else None)
        for place_id, row in rows_by_unique_place_id(rows).items()
    )
    return store.sync_existing_michelin_metadata(metadata)


def read_michelin_rows(path: Path) -> tuple[dict[str, object], ...]:
    rows = pd.read_csv(path)
    required_columns = {"place_id", "name", "michelin_category"}
    missing_columns = required_columns.difference(rows.columns)
    if missing_columns:
        raise ValueError(f"Michelin CSV missing required columns: {sorted(missing_columns)}")

    parsed_rows: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        place_id = str(row["place_id"]).strip()
        if pd.isna(row["place_id"]) or not place_id:
            raise ValueError("Michelin CSV contains a blank place_id")
        category = str(row["michelin_category"]).strip()
        parsed_rows.append(
            {
                "place_id": place_id,
                "name": str(row["name"]).strip(),
                "category": category,
                "stars": stars_from_category(category),
                "bib_gourmand": category == "Bib Gourmand",
                "year": int(row["year"]) if "year" in rows.columns and pd.notna(row["year"]) else None,
            }
        )
    return tuple(parsed_rows)


def stars_from_category(category: str) -> int:
    if category == "1 Star":
        return 1
    if category == "2 Stars":
        return 2
    if category == "3 Stars":
        return 3
    if category in {"Bib Gourmand", "Selected Restaurants"}:
        return 0
    raise ValueError(f"Unsupported Michelin category: {category}")


def place_ids_from_rows(rows: tuple[dict[str, object], ...]) -> tuple[str, ...]:
    return tuple(rows_by_unique_place_id(rows).keys())


def rows_by_unique_place_id(rows: tuple[dict[str, object], ...]) -> dict[str, dict[str, object]]:
    unique_rows: dict[str, dict[str, object]] = {}
    for row in rows:
        place_id = str(row["place_id"])
        if place_id not in unique_rows:
            unique_rows[place_id] = row
    return unique_rows


def log_michelin_attempts(
    store: DuckDbStore,
    place_id: str,
    attempts: tuple[dict[str, int], ...],
    used_requests_before_call: int,
) -> None:
    for index, attempt in enumerate(attempts, start=1):
        store.log_api_call(
            f"michelin-{place_id}-{used_requests_before_call + index}",
            "places/{place_id}",
            "place_details_enterprise_no_atmosphere",
            int(attempt["status_code"]),
            int(attempt["retry_count"]),
            place_id,
            None,
        )

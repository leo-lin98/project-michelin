"""Place Details enrichment workflows."""

import hashlib
import json

from restaurant_ingestion.clients.google_places import GooglePlacesClient, PlacesClientError, ensure_request_budget
from restaurant_ingestion.ingestion.checkpoints import FAILED, RUNNING, SUCCEEDED, detail_checkpoint
from restaurant_ingestion.ingestion.progress import print_crossed_request_progress
from restaurant_ingestion.storage.duckdb_store import DuckDbStore


def plan_candidate_enrichment(store: DuckDbStore, target_enriched_restaurants: int, details_refresh_ttl_days: int) -> dict[str, int]:
    remaining = max(target_enriched_restaurants - store.enriched_restaurant_count(), 0)
    place_ids = store.unenriched_candidate_ids(remaining, details_refresh_ttl_days)
    return {"candidate_details_requests": len(place_ids)}


def run_candidate_enrichment(
    client: GooglePlacesClient,
    store: DuckDbStore,
    target_enriched_restaurants: int,
    details_refresh_ttl_days: int,
    details_request_budget: int,
    place_details_field_mask: tuple[str, ...],
    progress_log_interval: int,
) -> dict[str, int]:
    remaining = max(target_enriched_restaurants - store.enriched_restaurant_count(), 0)
    place_ids = store.unenriched_candidate_ids(remaining, details_refresh_ttl_days)
    used_requests = store.api_call_count("place_details_enterprise_no_atmosphere")
    enriched_count = 0
    failed_count = 0

    for place_id in place_ids:
        ensure_request_budget(used_requests, details_request_budget, 1)
        checkpoint_id = detail_checkpoint(place_id)
        payload_json = json.dumps({"place_id": place_id}, sort_keys=True, separators=(",", ":"))
        store.upsert_checkpoint(checkpoint_id, "place_details", RUNNING, payload_json, None, None)
        try:
            place, raw_response, attempts = client.place_details(
                place_id,
                place_details_field_mask,
                details_request_budget - used_requests,
            )
            previous_used_requests = used_requests
            log_detail_attempts(store, place_id, checkpoint_id, attempts, previous_used_requests)
            used_requests += len(attempts)
            print_crossed_request_progress(
                "enrichment",
                previous_used_requests,
                used_requests,
                details_request_budget,
                progress_log_interval,
            )
            raw_json = json.dumps(raw_response, sort_keys=True, separators=(",", ":"))
            response_hash = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
            store.upsert_restaurant(place, raw_json, response_hash, False)
            store.upsert_checkpoint(checkpoint_id, "place_details", SUCCEEDED, payload_json, None, None)
            enriched_count += 1
        except PlacesClientError as exc:
            previous_used_requests = used_requests
            log_detail_attempts(store, place_id, checkpoint_id, exc.attempts, previous_used_requests)
            used_requests += len(exc.attempts)
            print_crossed_request_progress(
                "enrichment",
                previous_used_requests,
                used_requests,
                details_request_budget,
                progress_log_interval,
            )
            store.upsert_checkpoint(checkpoint_id, "place_details", FAILED, payload_json, None, str(exc))
            failed_count += 1
        except Exception as exc:
            store.upsert_checkpoint(checkpoint_id, "place_details", FAILED, payload_json, None, str(exc))
            failed_count += 1
    return {"enriched_count": enriched_count, "failed_count": failed_count}


def log_detail_attempts(
    store: DuckDbStore,
    place_id: str,
    checkpoint_id: str,
    attempts: tuple[dict[str, int], ...],
    used_requests_before_call: int,
) -> None:
    for index, attempt in enumerate(attempts, start=1):
        store.log_api_call(
            f"{checkpoint_id}-{used_requests_before_call + index}",
            "places/{place_id}",
            "place_details_enterprise_no_atmosphere",
            int(attempt["status_code"]),
            int(attempt["retry_count"]),
            place_id,
            checkpoint_id,
        )

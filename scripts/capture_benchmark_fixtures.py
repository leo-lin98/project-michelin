"""Fetch and freeze real Google candidate payloads for the place_id benchmark.

Run ONCE. Issues both the simple and strict queries for every benchmark row and writes the
raw responses to tests/benchmark/fixtures/<row_id>.json so the evaluation harness is
deterministic and offline. This is the only paid step; it is gated behind Doppler + explicit
cost confirmation, mirroring scripts/add_michelin_place_ids.py.
"""

from pathlib import Path
import argparse
import csv
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from restaurant_ingestion.clients.google_places import GooglePlacesClient
from restaurant_ingestion.config import (
    GOOGLE_MAX_RETRIES,
    GOOGLE_REGION_CODE,
    GOOGLE_REQUEST_TIMEOUT_SECONDS,
    GOOGLE_RETRY_INITIAL_BACKOFF_SECONDS,
    GOOGLE_RETRY_MAX_BACKOFF_SECONDS,
    GOOGLE_API_KEY_ENV_VAR,
    MICHELIN_PLACE_ID_LANGUAGE_CODE,
)
from restaurant_ingestion.ingestion.michelin_place_ids import (
    QUERY_MODE_SIMPLE,
    QUERY_MODE_STRICT,
    text_query_for_row,
)

BENCHMARK_DIR = ROOT / "tests" / "benchmark"
MANIFEST_CSV = BENCHMARK_DIR / "place_id_benchmark.csv"
FIXTURES_DIR = BENCHMARK_DIR / "fixtures"
QUERY_MODES = (QUERY_MODE_SIMPLE, QUERY_MODE_STRICT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Google candidate payloads for the place_id benchmark.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--confirm-live-api-cost", action="store_true")
    args = parser.parse_args()

    manifest_rows = load_manifest()
    planned_requests = len(manifest_rows) * len(QUERY_MODES)
    if args.dry_run:
        print({"rows": len(manifest_rows), "queries_per_row": len(QUERY_MODES), "planned_requests": planned_requests, "fixtures_dir": str(FIXTURES_DIR)})
        return

    request_budget = validate_live_run_args(args.max_requests, args.confirm_live_api_cost, planned_requests)
    require_doppler_runtime()
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    client = build_client()
    used_requests = 0
    try:
        for row in manifest_rows:
            queries = []
            for query_mode in QUERY_MODES:
                text_query = text_query_for_row(source_lookup_row(row), query_mode)
                response_json, attempts = client.search_text_place_id_match_candidates(text_query, request_budget - used_requests)
                used_requests += len(attempts)
                queries.append({"query_mode": query_mode, "text_query": text_query, "response": response_json})
            fixture_path = FIXTURES_DIR / f"{row['row_id']}.json"
            fixture_path.write_text(
                json.dumps({"row_id": row["row_id"], "name": row["name"], "queries": queries}, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
    finally:
        client.close()
    print({"rows": len(manifest_rows), "requests_used": used_requests, "fixtures_dir": str(FIXTURES_DIR)})


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def source_lookup_row(manifest_row: dict[str, str]) -> dict[str, object]:
    return {"Name": manifest_row["name"], "Address": manifest_row["address"]}


def validate_live_run_args(max_requests: int | None, confirm_live_api_cost: bool, planned_requests: int) -> int:
    if not confirm_live_api_cost:
        raise RuntimeError("Live Google Places calls are disabled without --confirm-live-api-cost")
    if max_requests is None:
        raise RuntimeError("Set --max-requests for this paid Google Places run")
    if max_requests < planned_requests:
        raise RuntimeError(f"--max-requests must cover all planned requests: planned={planned_requests}, max={max_requests}")
    return max_requests


def require_doppler_runtime() -> None:
    if not os.environ.get("DOPPLER_PROJECT") or not os.environ.get("DOPPLER_CONFIG"):
        raise RuntimeError("Run this command through Doppler: doppler run -- python3 scripts/capture_benchmark_fixtures.py ...")


def build_client() -> GooglePlacesClient:
    return GooglePlacesClient.from_environment(
        GOOGLE_API_KEY_ENV_VAR,
        MICHELIN_PLACE_ID_LANGUAGE_CODE,
        GOOGLE_REGION_CODE,
        GOOGLE_REQUEST_TIMEOUT_SECONDS,
        GOOGLE_MAX_RETRIES,
        GOOGLE_RETRY_INITIAL_BACKOFF_SECONDS,
        GOOGLE_RETRY_MAX_BACKOFF_SECONDS,
    )


if __name__ == "__main__":
    main()

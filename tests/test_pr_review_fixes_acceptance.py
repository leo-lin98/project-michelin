"""Independent acceptance coverage for the three PR #1 review findings."""

from collections.abc import Callable
import csv
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

import httpx
import pytest

from restaurant_ingestion.clients.google_places import BudgetExceededError, GooglePlacesClient, PlacesClientError
from restaurant_ingestion.ingestion.grid_search import run_taipei_discovery
from restaurant_ingestion.ingestion.michelin_place_ids import (
    OUTPUT_COLUMNS,
    QUERY_MODE_SIMPLE,
    UNRESOLVED_COLUMNS,
    VALIDATION_REPORT_COLUMNS,
    build_processed_place_id_dataset,
    latest_unresolved_csv_path,
    latest_unresolved_csv_path_or_none,
    resolve_michelin_place_ids_to_paths,
    resolve_michelin_place_ids_with_matcher_to_paths,
    stack_resolved_place_id_datasets,
    stack_resolved_place_id_directory,
    validation_report_csv_path,
)
from restaurant_ingestion.storage.duckdb_store import DuckDbStore
from scripts import ingest_taipei


def acceptance_client(handler: Callable[[httpx.Request], httpx.Response], retries: int) -> GooglePlacesClient:
    client = GooglePlacesClient("test-only", "TEST_ONLY_KEY", "en", "TW", 1.0, retries, 0.0, 0.0)
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def configure_ingestion(monkeypatch: pytest.MonkeyPatch, database_path: Path, budget: int, details_budget: int) -> None:
    settings = {
        "DATABASE_PATH": database_path,
        "TAIPEI_GRID_SOUTH": 25.0,
        "TAIPEI_GRID_WEST": 121.0,
        "TAIPEI_GRID_NORTH": 25.1,
        "TAIPEI_GRID_EAST": 121.1,
        "TAIPEI_GRID_TILE_SIZE_DEGREES": 0.2,
        "DISCOVERY_QUERIES": ("restaurant", "cafe"),
        "ORDINARY_RESTAURANT_TARGET": 10,
        "TARGET_ENRICHED_RESTAURANTS": 10,
        "DISCOVERY_REQUEST_BUDGET": budget,
        "DETAILS_REQUEST_BUDGET": details_budget,
        "DETAILS_REFRESH_TTL_DAYS": 30,
        "PROGRESS_LOG_INTERVAL": 100,
    }
    for name, value in settings.items():
        monkeypatch.setattr(ingest_taipei, name, value)


@pytest.mark.parametrize("boundary", ["entry", "mid_run", "retry"])
def test_discovery_budget_exhaustion_still_enriches_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, boundary: str,
) -> None:
    path = tmp_path / "ingestion.duckdb"
    configure_ingestion(monkeypatch, path, 1, 1)
    store = DuckDbStore(path)
    try:
        store.create_schema()
        if boundary != "mid_run":
            store.upsert_candidate("saved-place", "prior-search")
        if boundary == "entry":
            store.log_api_call("prior-search", "places:searchText", "text_search_ids_only", 200, 0, None, None)
    finally:
        store.close()
    searches: list[str] = []
    details: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            searches.append(str(json.loads(request.content)["textQuery"]))
            if boundary == "retry":
                return httpx.Response(429, json={"error": {"message": "test rate limit"}})
            return httpx.Response(200, json={"places": [{"id": "saved-place"}]})
        details.append(request.url.path.rsplit("/", 1)[1])
        return httpx.Response(200, json={
            "id": "saved-place", "displayName": {"text": "Saved Place"},
            "location": {"latitude": 25.0, "longitude": 121.0},
        })

    monkeypatch.setattr(ingest_taipei, "build_client", lambda: acceptance_client(handler, 3))
    with caplog.at_level(logging.WARNING):
        first = ingest_taipei.run_ingestion()
        second = ingest_taipei.run_ingestion()
    assert first["enrichment"]["enriched_count"] == 1
    assert second["enrichment"]["enriched_count"] == 0
    assert details == ["saved-place"]
    assert len(searches) == (0 if boundary == "entry" else 1)
    assert "budget" in caplog.text.lower()
    assert "discover" in caplog.text.lower()
    assert "exhaust" in json.dumps(first["discovery"]).lower()
    store = DuckDbStore(path)
    try:
        assert store.api_call_count("text_search_ids_only") == 1
        assert store.api_call_count("place_details_enterprise_no_atmosphere") == 1
        assert store.enriched_restaurant_count() == 1
        client = acceptance_client(handler, 0)
        try:
            with pytest.raises(BudgetExceededError):
                run_taipei_discovery(client, store, 25.0, 121.0, 25.1, 121.1, 0.2, ("restaurant", "cafe"), 10, 1, 100)
        finally:
            client.close()
    finally:
        store.close()
    assert len(searches) == (0 if boundary == "entry" else 1)


def test_discovery_unrelated_or_details_failure_is_not_suppressed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_ingestion(monkeypatch, tmp_path / "failure.duckdb", 1, 1)

    def forbidden_network(request: httpx.Request) -> httpx.Response:
        raise AssertionError("No external requests expected")

    monkeypatch.setattr(ingest_taipei, "build_client", lambda: acceptance_client(forbidden_network, 0))

    def broken_discovery(*args: object) -> dict[str, int]:
        raise ValueError("unrelated discovery failure")

    monkeypatch.setattr(ingest_taipei, "run_taipei_discovery", broken_discovery)
    with pytest.raises(ValueError, match="unrelated discovery failure"):
        ingest_taipei.run_ingestion()

    def exhausted_discovery(*args: object) -> dict[str, int]:
        raise BudgetExceededError("discovery exhausted", ())

    def exhausted_details(*args: object) -> dict[str, int]:
        raise BudgetExceededError("details exhausted", ())

    monkeypatch.setattr(ingest_taipei, "run_taipei_discovery", exhausted_discovery)
    monkeypatch.setattr(ingest_taipei, "run_candidate_enrichment", exhausted_details)
    with pytest.raises(BudgetExceededError, match="details exhausted"):
        ingest_taipei.run_ingestion()


def lookup_source(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Name", "Address", "Location", "Award", "Latitude", "Longitude", "Url"))
        for name in ("Alpha Dining", "Beta Dining"):
            writer.writerow((name, "28 Jilin Road, Taipei, Taiwan", "Taipei, Taiwan", "1 Star", 25.0, 121.0, f"https://guide.example/{name.split()[0]}"))


def lookup_candidate(name: str, place_id: str) -> dict[str, object]:
    return {
        "id": place_id, "displayName": {"text": name},
        "formattedAddress": "28 Jilin Road, Taipei, Taiwan",
        "location": {"latitude": 25.0, "longitude": 121.0}, "primaryType": "restaurant",
    }


def csv_rows(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    assert path.exists(), f"Completed decisions must be checkpointed: {path.name}"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == columns
        return list(reader)


@pytest.mark.parametrize("resolver", ["legacy", "matcher"])
@pytest.mark.parametrize("first_outcome", ["resolved", "unresolved"])
@pytest.mark.parametrize("interruption", ["budget", "api", "retry"])
def test_lookup_completed_decisions_survive_interruption_and_resume(
    tmp_path: Path, resolver: str, first_outcome: str, interruption: str,
) -> None:
    source = tmp_path / "guide.csv"
    output = tmp_path / "resolved" / "base.csv"
    unresolved = tmp_path / "unresolved" / "base.unresolved.csv"
    report = validation_report_csv_path(unresolved)
    lookup_source(source)
    queries: list[str] = []
    first_cost = 2 if resolver == "matcher" and first_outcome == "unresolved" else 1
    budget = first_cost if interruption == "budget" else first_cost + 2

    def interrupted(request: httpx.Request) -> httpx.Response:
        query = str(json.loads(request.content)["textQuery"])
        queries.append(query)
        if query.startswith("Beta Dining"):
            assert [row["name"] for row in csv_rows(output, OUTPUT_COLUMNS)] == (["Alpha Dining"] if first_outcome == "resolved" else [])
            assert [row["name"] for row in csv_rows(unresolved, UNRESOLVED_COLUMNS)] == (["Alpha Dining"] if first_outcome == "unresolved" else [])
            expected = ["Alpha Dining"] if resolver == "matcher" or first_outcome == "unresolved" else []
            assert [row["name"] for row in csv_rows(report, VALIDATION_REPORT_COLUMNS)] == expected
            return httpx.Response(429 if interruption == "retry" else 403, json={"error": {"message": "later request interrupted"}})
        places = [lookup_candidate("Alpha Dining", "alpha-id")] if first_outcome == "resolved" else []
        return httpx.Response(200, json={"places": places})

    client = acceptance_client(interrupted, 3)
    failure_type = BudgetExceededError if interruption in {"budget", "retry"} else PlacesClientError
    try:
        with pytest.raises(failure_type):
            if resolver == "legacy":
                resolve_michelin_place_ids_to_paths(source, output, unresolved, client, budget, QUERY_MODE_SIMPLE)
            else:
                resolve_michelin_place_ids_with_matcher_to_paths(source, output, unresolved, client, budget, QUERY_MODE_SIMPLE, None)
    finally:
        client.close()
    assert len(queries) <= budget
    resolved_before = csv_rows(output, OUTPUT_COLUMNS)
    unresolved_before = csv_rows(unresolved, UNRESOLVED_COLUMNS)
    reports_before = csv_rows(report, VALIDATION_REPORT_COLUMNS)
    assert [row["name"] for row in resolved_before] == (["Alpha Dining"] if first_outcome == "resolved" else [])
    assert [row["name"] for row in unresolved_before] == (["Alpha Dining"] if first_outcome == "unresolved" else [])
    expected_reports = ["Alpha Dining"] if resolver == "matcher" or first_outcome == "unresolved" else []
    assert [row["name"] for row in reports_before] == expected_reports
    if reports_before:
        assert reports_before[0]["reason"]
        assert reports_before[0]["candidate_count"] == ("1" if first_outcome == "resolved" else "0")
    resumed_queries: list[str] = []

    def resumed(request: httpx.Request) -> httpx.Response:
        query = str(json.loads(request.content)["textQuery"])
        resumed_queries.append(query)
        assert query.startswith("Beta Dining"), "Completed Alpha decision must not be queried again"
        return httpx.Response(200, json={"places": [lookup_candidate("Beta Dining", "beta-id")]})

    client = acceptance_client(resumed, 0)
    try:
        if resolver == "legacy":
            result = resolve_michelin_place_ids_to_paths(source, output, unresolved, client, 1, QUERY_MODE_SIMPLE)
            again = resolve_michelin_place_ids_to_paths(source, output, unresolved, client, 1, QUERY_MODE_SIMPLE)
        else:
            result = resolve_michelin_place_ids_with_matcher_to_paths(source, output, unresolved, client, 1, QUERY_MODE_SIMPLE, None)
            again = resolve_michelin_place_ids_with_matcher_to_paths(source, output, unresolved, client, 1, QUERY_MODE_SIMPLE, None)
    finally:
        client.close()
    assert len(resumed_queries) == 1
    assert result["resolved_rows"] == 1
    assert result["skipped_existing_rows"] == (1 if first_outcome == "resolved" else 0)
    assert result["skipped_existing_unresolved_rows"] == (1 if first_outcome == "unresolved" else 0)
    assert again["text_search_requests"] == 0
    assert csv_rows(output, OUTPUT_COLUMNS) == [*resolved_before, {
        "place_id": "beta-id", "name": "Beta Dining", "michelin_category": "1 Star",
        "address": "28 Jilin Road, Taipei, Taiwan", "location": "Taipei, Taiwan",
        "latitude": "25.0", "longitude": "121.0", "michelin_url": "https://guide.example/Beta",
    }]
    assert csv_rows(unresolved, UNRESOLVED_COLUMNS) == unresolved_before
    final_reports = csv_rows(report, VALIDATION_REPORT_COLUMNS)
    assert final_reports[:len(reports_before)] == reports_before
    assert [row["name"] for row in final_reports] == expected_reports + (["Beta Dining"] if resolver == "matcher" else [])


@pytest.mark.parametrize("first_outcome", ["resolved", "unresolved"])
def test_lookup_matcher_prior_decision_survives_llm_failure(tmp_path: Path, first_outcome: str) -> None:
    source = tmp_path / "guide.csv"
    output = tmp_path / "base.csv"
    unresolved = tmp_path / "base.unresolved.csv"
    lookup_source(source)
    requests: list[str] = []
    judgments: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        query = str(json.loads(request.content)["textQuery"])
        requests.append(query)
        if query.startswith("Alpha Dining"):
            places = [lookup_candidate("Alpha Dining", "alpha-id")] if first_outcome == "resolved" else []
        else:
            assert [row["name"] for row in csv_rows(validation_report_csv_path(unresolved), VALIDATION_REPORT_COLUMNS)] == ["Alpha Dining"]
            places = [lookup_candidate("Beta Dining Annex", "beta-a"), lookup_candidate("Beta Dining House", "beta-b")]
        return httpx.Response(200, json={"places": places})

    def judge(row: dict[str, object], candidates: tuple[dict[str, object], ...]) -> dict[str, str | tuple[str, ...]]:
        judgments.append(str(row["name"]))
        if row["name"] == "Beta Dining":
            raise RuntimeError("later LLM failure")
        return {"recommended_outcome": "no_safe_match", "recommended_place_id": "", "confidence": "high", "evidence": (), "conflicts": ()}

    client = acceptance_client(handler, 0)
    try:
        with pytest.raises(RuntimeError, match="later LLM failure"):
            resolve_michelin_place_ids_with_matcher_to_paths(source, output, unresolved, client, 8, QUERY_MODE_SIMPLE, judge)
    finally:
        client.close()
    assert judgments[-1] == "Beta Dining"
    assert [row["name"] for row in csv_rows(output, OUTPUT_COLUMNS)] == (["Alpha Dining"] if first_outcome == "resolved" else [])
    assert [row["name"] for row in csv_rows(unresolved, UNRESOLVED_COLUMNS)] == (["Alpha Dining"] if first_outcome == "unresolved" else [])
    assert [row["name"] for row in csv_rows(validation_report_csv_path(unresolved), VALIDATION_REPORT_COLUMNS)] == ["Alpha Dining"]


def test_hydration_file_cli_help_without_pythonpath() -> None:
    root = Path(__file__).resolve().parents[1]
    process_environment = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "scripts/hydrate_michelin_prices.py", "--help"],
        cwd=root, env=process_environment, capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--dry-run" in result.stdout
    assert "price_range" in result.stdout


def write_acceptance_csv(path: Path, columns: tuple[str, ...], rows: tuple[dict[str, object], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def historical_retry_inputs(resolved_dir: Path, unresolved_dir: Path) -> dict[Path, bytes]:
    guide_fields = {
        "michelin_category": "1 Star", "address": "28 Jilin Road, Taipei, Taiwan",
        "location": "Taipei, Taiwan", "latitude": 25.0, "longitude": 121.0,
    }
    base = resolved_dir / "base.csv"
    unresolved = unresolved_dir / "base.unresolved.csv"
    write_acceptance_csv(base, OUTPUT_COLUMNS, ({
        **guide_fields, "place_id": "gamma-id", "name": "Gamma Dining", "michelin_url": "https://guide.example/Gamma",
    },))
    write_acceptance_csv(unresolved, UNRESOLVED_COLUMNS, tuple({
        **guide_fields, "name": name, "michelin_url": f"https://guide.example/{name.split()[0]}",
        "phone_number": "", "website_url": "", "text_query": f"{name}, Taipei, Taiwan",
    } for name in ("Alpha Dining", "Beta Dining")))
    return {path: path.read_bytes() for path in (base, unresolved)}


def invoke_acceptance_lookup(
    resolver: str, source: Path, output: Path, unresolved: Path, client: GooglePlacesClient, budget: int,
) -> dict[str, int]:
    if resolver == "legacy":
        return resolve_michelin_place_ids_to_paths(source, output, unresolved, client, budget, QUERY_MODE_SIMPLE)
    return resolve_michelin_place_ids_with_matcher_to_paths(source, output, unresolved, client, budget, QUERY_MODE_SIMPLE, None)


def assert_incomplete_workflow_rejected(operations: tuple[tuple[str, Callable[[], object]], ...]) -> None:
    failures: list[str] = []
    for label, operation in operations:
        try:
            operation()
        except Exception as exc:
            message = str(exc).lower()
            if "incomplete" not in message or "resume" not in message:
                failures.append(f"{label}: error must identify incomplete work and direct original-run resume: {exc}")
        else:
            failures.append(f"{label}: accepted incomplete lookup progress")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("resolver", ["legacy", "matcher"])
@pytest.mark.parametrize("run_stem", ["base", "retry-001"])
@pytest.mark.parametrize("interruption", ["budget", "api"])
def test_lookup_incomplete_workflow_rejects_promotion_until_same_run_resume(
    tmp_path: Path, resolver: str, run_stem: str, interruption: str,
) -> None:
    source = tmp_path / "guide.csv"
    lookup_source(source)
    original_source = source.read_bytes()
    resolved_dir = tmp_path / "resolved"
    unresolved_dir = tmp_path / "unresolved"
    historical = historical_retry_inputs(resolved_dir, unresolved_dir) if run_stem != "base" else {}
    output = resolved_dir / f"{run_stem}.csv"
    unresolved = unresolved_dir / f"{run_stem}.unresolved.csv"
    processed = tmp_path / "processed.csv"
    processed.write_text("previously completed dataset\n", encoding="utf-8")
    previous_processed = processed.read_bytes()
    queries: list[str] = []

    def interrupted(request: httpx.Request) -> httpx.Response:
        query = str(json.loads(request.content)["textQuery"])
        queries.append(query)
        if query.startswith("Beta Dining"):
            return httpx.Response(403, json={"error": {"message": "test later API failure"}})
        return httpx.Response(200, json={"places": [lookup_candidate("Alpha Dining", "alpha-id")]})

    client = acceptance_client(interrupted, 0)
    try:
        with pytest.raises(BudgetExceededError if interruption == "budget" else PlacesClientError):
            invoke_acceptance_lookup(resolver, source, output, unresolved, client, 1 if interruption == "budget" else 2)
    finally:
        client.close()
    assert [row["name"] for row in csv_rows(output, OUTPUT_COLUMNS)] == ["Alpha Dining"]
    assert csv_rows(unresolved, UNRESOLVED_COLUMNS) == []
    input_paths = (resolved_dir / "base.csv", output) if run_stem != "base" else (output,)
    assert_incomplete_workflow_rejected((
        ("processed directory assembly", lambda: build_processed_place_id_dataset(resolved_dir, unresolved_dir, processed)),
        ("resolved directory stacking", lambda: stack_resolved_place_id_directory(resolved_dir, processed)),
        ("explicit path stacking", lambda: stack_resolved_place_id_datasets(input_paths, processed)),
        ("required latest unresolved selector", lambda: latest_unresolved_csv_path(unresolved_dir)),
        ("optional latest unresolved selector", lambda: latest_unresolved_csv_path_or_none(unresolved_dir)),
    ))
    assert processed.read_bytes() == previous_processed
    resumed_queries: list[str] = []

    def resumed(request: httpx.Request) -> httpx.Response:
        query = str(json.loads(request.content)["textQuery"])
        resumed_queries.append(query)
        assert query.startswith("Beta Dining"), "Same-run resume must skip completed Alpha decision"
        return httpx.Response(200, json={"places": [lookup_candidate("Beta Dining", "beta-id")]})

    client = acceptance_client(resumed, 0)
    try:
        resumed_result = invoke_acceptance_lookup(resolver, source, output, unresolved, client, 1)
        repeated_result = invoke_acceptance_lookup(resolver, source, output, unresolved, client, 1)
    finally:
        client.close()
    assert len(resumed_queries) == 1
    assert resumed_result["skipped_existing_rows"] == 1
    assert resumed_result["resolved_rows"] == 1
    assert repeated_result["text_search_requests"] == 0
    expected_ids = {"alpha-id", "beta-id"} | ({"gamma-id"} if historical else set())
    result = build_processed_place_id_dataset(resolved_dir, unresolved_dir, processed)
    assert result["output_rows"] == len(expected_ids)
    assert {row["place_id"] for row in csv_rows(processed, OUTPUT_COLUMNS)} == expected_ids
    assert stack_resolved_place_id_directory(resolved_dir, processed)["output_rows"] == len(expected_ids)
    assert stack_resolved_place_id_datasets(input_paths, processed)["output_rows"] == len(expected_ids)
    assert source.read_bytes() == original_source
    assert {path: path.read_bytes() for path in historical} == historical
    assert latest_unresolved_csv_path_or_none(unresolved_dir) == (unresolved_dir / "base.unresolved.csv" if historical else None)


@pytest.mark.parametrize("resolver", ["legacy", "matcher"])
def test_lookup_incomplete_workflow_partial_unresolved_cannot_advance_retry(tmp_path: Path, resolver: str) -> None:
    source = tmp_path / "guide.csv"
    lookup_source(source)
    resolved_dir = tmp_path / "resolved"
    unresolved_dir = tmp_path / "unresolved"
    historical = historical_retry_inputs(resolved_dir, unresolved_dir)
    output = resolved_dir / "retry-001.csv"
    unresolved = unresolved_dir / "retry-001.unresolved.csv"

    def unmatched(request: httpx.Request) -> httpx.Response:
        assert str(json.loads(request.content)["textQuery"]).startswith("Alpha Dining")
        return httpx.Response(200, json={"places": []})

    client = acceptance_client(unmatched, 0)
    try:
        with pytest.raises(BudgetExceededError):
            invoke_acceptance_lookup(resolver, source, output, unresolved, client, 2 if resolver == "matcher" else 1)
    finally:
        client.close()
    assert [row["name"] for row in csv_rows(unresolved, UNRESOLVED_COLUMNS)] == ["Alpha Dining"]
    assert_incomplete_workflow_rejected((
        ("required selector with partial unresolved decision", lambda: latest_unresolved_csv_path(unresolved_dir)),
        ("optional selector with partial unresolved decision", lambda: latest_unresolved_csv_path_or_none(unresolved_dir)),
    ))

    def remaining(request: httpx.Request) -> httpx.Response:
        assert str(json.loads(request.content)["textQuery"]).startswith("Beta Dining")
        return httpx.Response(200, json={"places": [lookup_candidate("Beta Dining", "beta-id")]})

    client = acceptance_client(remaining, 0)
    try:
        result = invoke_acceptance_lookup(resolver, source, output, unresolved, client, 1)
    finally:
        client.close()
    assert result["skipped_existing_unresolved_rows"] == 1
    assert result["resolved_rows"] == 1
    assert latest_unresolved_csv_path(unresolved_dir) == unresolved
    assert latest_unresolved_csv_path_or_none(unresolved_dir) == unresolved
    assert {path: path.read_bytes() for path in historical} == historical


@pytest.mark.parametrize("resolver", ["legacy", "matcher"])
def test_lookup_incomplete_workflow_empty_success_is_complete(tmp_path: Path, resolver: str) -> None:
    source = tmp_path / "empty-guide.csv"
    write_acceptance_csv(source, ("Name", "Address", "Location", "Award", "Latitude", "Longitude", "Url"), ())
    output = tmp_path / "resolved" / "base.csv"
    unresolved = tmp_path / "unresolved" / "base.unresolved.csv"

    def no_requests(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Empty source must not call Places")

    client = acceptance_client(no_requests, 0)
    try:
        result = invoke_acceptance_lookup(resolver, source, output, unresolved, client, 1)
    finally:
        client.close()
    assert result["text_search_requests"] == 0
    assert build_processed_place_id_dataset(output.parent, unresolved.parent, tmp_path / "processed.csv")["output_rows"] == 0
    assert latest_unresolved_csv_path_or_none(unresolved.parent) is None

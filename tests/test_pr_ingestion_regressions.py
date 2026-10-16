from collections.abc import Callable
from datetime import UTC, datetime, timedelta
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import duckdb
import httpx
import pytest

from restaurant_ingestion.clients.google_places import BudgetExceededError, GooglePlacesClient
from restaurant_ingestion.ingestion.enrichment import run_candidate_enrichment
from restaurant_ingestion.ingestion.grid_search import run_taipei_discovery
from restaurant_ingestion.ingestion.michelin import run_michelin_csv_enrichment
from restaurant_ingestion.models import PlaceDetails
from restaurant_ingestion.storage.duckdb_store import DuckDbStore


def script_module(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mock_client(handler: Callable[[httpx.Request], httpx.Response], retries: int) -> GooglePlacesClient:
    client = GooglePlacesClient("test-only", "TEST_ONLY_KEY", "en", "TW", 1.0, retries, 0.0, 0.0)
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def details(place_id: str) -> PlaceDetails:
    return PlaceDetails.model_validate({
        "id": place_id, "displayName": {"text": place_id},
        "location": {"latitude": 25.0, "longitude": 121.5}, "rating": 4.7,
    })


@pytest.mark.parametrize("workflow", ["ordinary", "michelin", "discovery", "backfill"])
@pytest.mark.parametrize("failure", ["rate_limit", "transport"])
@pytest.mark.parametrize("budget", [1, 2])
def test_workflow_retry_ceiling_matches_persisted_attempts(
    tmp_path: Path, workflow: str, failure: str, budget: int,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if failure == "transport":
            raise httpx.ConnectError("offline test failure", request=request)
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    store = DuckDbStore(tmp_path / "budget.duckdb")
    client = mock_client(handler, 3)
    try:
        store.create_schema()
        for place_id in ("one", "two", "three"):
            store.upsert_candidate(place_id, "seed")
        guide = tmp_path / "guide.csv"
        guide.write_text("place_id,name,michelin_category\none,One,1 Star\ntwo,Two,Bib Gourmand\nthree,Three,Selected Restaurants\n")
        try:
            if workflow == "ordinary":
                run_candidate_enrichment(client, store, 3, 30, budget, ("id", "displayName", "location"), 100)
            elif workflow == "michelin":
                run_michelin_csv_enrichment(guide, client, store, 30, budget, ("id", "displayName", "location"), 100)
            elif workflow == "discovery":
                run_taipei_discovery(client, store, 25.0, 121.0, 25.1, 121.1, 0.2, ("restaurant", "noodles", "cafe"), 10, budget, 100)
            else:
                script_module("backfill_price_fields").backfill_prices(
                    client, store, [("one", None), ("two", None), ("three", None)], budget, 100,
                )
        except BudgetExceededError:
            pass
        sku = "text_search_ids_only" if workflow == "discovery" else "place_details_enterprise_no_atmosphere"
        assert len(requests) == budget
        assert store.api_call_count(sku) == len(requests)
    finally:
        client.close()
        store.close()


@pytest.mark.parametrize("workflow", ["ordinary", "michelin", "discovery", "backfill"])
@pytest.mark.parametrize("invalid_response", ["invalid_json", "invalid_schema"])
def test_successful_http_with_invalid_body_still_consumes_budget(
    tmp_path: Path, workflow: str, invalid_response: str,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if invalid_response == "invalid_json":
            return httpx.Response(200, content="{broken", headers={"content-type": "application/json"})
        payload = {"places": [{"id": ""}]} if workflow == "discovery" else {"id": "one"}
        return httpx.Response(200, json=payload)

    store = DuckDbStore(tmp_path / "invalid.duckdb")
    client = mock_client(handler, 0)
    try:
        store.create_schema()
        for place_id in ("one", "two"):
            store.upsert_candidate(place_id, "seed")
        guide = tmp_path / "guide.csv"
        guide.write_text("place_id,name,michelin_category\none,One,1 Star\ntwo,Two,Bib Gourmand\n")
        try:
            if workflow == "ordinary":
                run_candidate_enrichment(client, store, 2, 30, 1, ("id", "displayName", "location"), 100)
            elif workflow == "michelin":
                run_michelin_csv_enrichment(guide, client, store, 30, 1, ("id", "displayName", "location"), 100)
            elif workflow == "discovery":
                run_taipei_discovery(client, store, 25.0, 121.0, 25.1, 121.1, 0.2, ("restaurant", "cafe"), 10, 1, 100)
            else:
                script_module("backfill_price_fields").backfill_prices(client, store, [("one", None), ("two", None)], 1, 100)
        except BudgetExceededError:
            pass
        sku = "text_search_ids_only" if workflow == "discovery" else "place_details_enterprise_no_atmosphere"
        assert len(requests) == 1
        assert store.api_call_count(sku) == 1
        assert store.enriched_restaurant_count() == 0
    finally:
        client.close()
        store.close()


@pytest.mark.parametrize("completed_pages", [1, 2])
def test_discovery_resume_requests_only_unfinished_pages(tmp_path: Path, completed_pages: int) -> None:
    pages: list[str | None] = []

    def interrupted(request: httpx.Request) -> httpx.Response:
        token = json.loads(request.content).get("pageToken")
        pages.append(token)
        index = 0 if token is None else int(token)
        if index == completed_pages:
            return httpx.Response(403, json={"error": {"message": "interrupted"}})
        return httpx.Response(200, json={"places": [{"id": f"place-{index}"}], "nextPageToken": str(index + 1)})

    def resumed(request: httpx.Request) -> httpx.Response:
        token = json.loads(request.content).get("pageToken")
        pages.append(token)
        return httpx.Response(200, json={"places": [{"id": f"place-{token}"}]})

    path = tmp_path / "resume.duckdb"
    store = DuckDbStore(path)
    client = mock_client(interrupted, 0)
    try:
        store.create_schema()
        run_taipei_discovery(client, store, 25.0, 121.0, 25.1, 121.1, 0.2, ("restaurant",), 10, 20, 100)
    finally:
        client.close()
        store.close()
    first_pages = tuple(pages)
    store = DuckDbStore(path)
    client = mock_client(resumed, 0)
    try:
        run_taipei_discovery(client, store, 25.0, 121.0, 25.1, 121.1, 0.2, ("restaurant",), 10, 20, 100)
        assert tuple(pages) == first_pages + (str(completed_pages),)
        assert set(store.unenriched_candidate_ids(10, 30)) == {f"place-{index}" for index in range(completed_pages + 1)}
        run_taipei_discovery(client, store, 25.0, 121.0, 25.1, 121.1, 0.2, ("restaurant",), 10, 20, 100)
        assert tuple(pages) == first_pages + (str(completed_pages),)
    finally:
        client.close()
        store.close()


@pytest.mark.parametrize("target", [2, 3])
@pytest.mark.parametrize("prior_requests", [0, 1])
def test_refresh_script_at_population_target_uses_remaining_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: int, prior_requests: int,
) -> None:
    path = tmp_path / "refresh.duckdb"
    store = DuckDbStore(path)
    try:
        store.create_schema()
        for place_id in ("stale-one", "stale-two", "fresh"):
            place = details(place_id)
            store.upsert_candidate(place_id, "seed")
            store.upsert_restaurant(place, place.model_dump_json(), f"hash-{place_id}", place_id == "stale-one")
        store.upsert_michelin_metadata("stale-one", "Guide One", "1 Star", 1, False, 2026)
        for index in range(prior_requests):
            store.log_api_call(f"prior-{index}", "places/{place_id}", "place_details_enterprise_no_atmosphere", 200, 0, "fresh", None)
    finally:
        store.close()
    with duckdb.connect(str(path)) as connection:
        connection.execute("UPDATE restaurants SET last_updated = ? WHERE place_id LIKE 'stale-%'", [datetime.now(UTC).replace(tzinfo=None) - timedelta(days=100)])

    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        place_id = request.url.path.rsplit("/", 1)[1]
        requested.append(place_id)
        return httpx.Response(200, json=details(place_id).model_dump(mode="json"))

    module = script_module("refresh_restaurants")
    client = mock_client(handler, 0)
    monkeypatch.setattr(module, "DATABASE_PATH", path)
    monkeypatch.setattr(module, "TARGET_ENRICHED_RESTAURANTS", target, raising=False)
    monkeypatch.setattr(module, "DETAILS_REFRESH_TTL_DAYS", 30)
    monkeypatch.setattr(module, "DETAILS_REQUEST_BUDGET", 2)
    monkeypatch.setattr(module, "build_client", lambda: client)
    assert sum(module.plan_refresh().values()) == 2 - prior_requests
    module.run_refresh()
    assert len(requested) == 2 - prior_requests
    assert set(requested).issubset({"stale-one", "stale-two"})
    with duckdb.connect(str(path), read_only=True) as connection:
        assert connection.execute("SELECT is_michelin FROM restaurants WHERE place_id='stale-one'").fetchone() == (True,)
        assert connection.execute("SELECT michelin_stars FROM restaurant_michelin WHERE place_id='stale-one'").fetchone() == (1,)
    assert sum(module.plan_refresh().values()) == 0


@pytest.mark.parametrize("outcome", ["success", "retry_success", "failure", "exhausted"])
def test_repeated_backfill_retains_each_runs_attempts(tmp_path: Path, outcome: str) -> None:
    requests: list[httpx.Request] = []
    budget = 1 if outcome in {"success", "failure"} else 2

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if outcome == "failure":
            return httpx.Response(403, json={"error": {"message": "forbidden"}})
        if outcome == "exhausted" or (outcome == "retry_success" and len(requests) % 2 == 1):
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(200, json=details("same-place").model_dump(mode="json"))

    path = tmp_path / "backfill.duckdb"
    module = script_module("backfill_price_fields")
    for run in range(2):
        store = DuckDbStore(path)
        client = mock_client(handler, 3)
        try:
            store.create_schema()
            result = module.backfill_prices(client, store, [("same-place", None), ("unreached", None)], budget, 100)
            assert result["requests_used"] == budget
            assert len(requests) == (run + 1) * budget
            assert store.api_call_count("place_details_enterprise_no_atmosphere") == len(requests)
        finally:
            client.close()
            store.close()

"""Lookup checkpoints retain existing artifacts and complete decisions."""

import csv
import json
from pathlib import Path

import httpx
import pytest

from restaurant_ingestion.clients.google_places import GooglePlacesClient, PlacesClientError
from restaurant_ingestion.ingestion.michelin_place_ids import (
    QUERY_MODE_SIMPLE,
    latest_unresolved_csv_path_or_none,
    resolve_michelin_place_ids_to_paths,
    resolve_michelin_place_ids_with_matcher_to_paths,
    stack_resolved_place_id_datasets,
)


def read_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("resolver", ["legacy", "matcher"])
def test_lookup_checkpoints_keep_older_artifacts_and_resume_only_incomplete_rows(
    tmp_path: Path, resolver: str,
) -> None:
    source = tmp_path / "guide.csv"
    resolved = tmp_path / "base.csv"
    unresolved = tmp_path / "base.unresolved.csv"
    report = tmp_path / "base.validation-report.csv"
    source.write_text(
        'Name,Address,Location,Award,Latitude,Longitude,Url\n'
        'Cedar,"28 Jilin Road, Taipei, Taiwan","Taipei, Taiwan",1 Star,25.0,121.0,https://guide.example/cedar\n'
        'Juniper,"28 Jilin Road, Taipei, Taiwan","Taipei, Taiwan",1 Star,25.0,121.0,https://guide.example/juniper\n',
        encoding="utf-8",
    )
    # Previously supported CSVs can omit optional columns and a terminal newline.
    resolved.write_text(
        'name,place_id,michelin_category,address,location,latitude,longitude,michelin_url\n'
        'Saved,saved-id,Selected,old address,"Taipei, Taiwan",25.0,121.0,https://guide.example/saved',
        encoding="utf-8",
    )
    unresolved.write_text(
        'name,michelin_category,address,location,latitude,longitude,michelin_url,text_query\n'
        'Deferred,Selected,old address,"Taipei, Taiwan",25.0,121.0,https://guide.example/deferred,old query',
        encoding="utf-8",
    )
    report.write_text(
        'name,michelin_category,address,location,latitude,longitude,michelin_url,text_query,reason,'
        'candidate_count,valid_candidate_count,candidate_place_ids,valid_candidate_place_ids,'
        'candidate_addresses,candidate_distances_meters,candidate_failed_checks\n'
        'Deferred,Selected,old address,"Taipei, Taiwan",25.0,121.0,https://guide.example/deferred,'
        'old query,no_candidates,0,0,,,,,',
        encoding="utf-8",
    )
    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        query = str(json.loads(request.content)["textQuery"])
        queries.append(query)
        if len(queries) == 1:
            with pytest.raises(ValueError, match="(?i)incomplete.*resume") as error:
                stack_resolved_place_id_datasets((resolved,), tmp_path / "processed.csv")
            for original_path in (source, resolved, unresolved):
                assert str(original_path) in str(error.value)
            with pytest.raises(ValueError, match="(?i)incomplete.*resume"):
                latest_unresolved_csv_path_or_none(tmp_path)
        if query.startswith("Juniper") and len(queries) == 2:
            assert [row["name"] for row in read_records(resolved)] == ["Saved", "Cedar"]
            deferred = read_records(unresolved)
            assert [row["name"] for row in deferred] == ["Deferred"]
            assert deferred[0]["phone_number"] == ""
            assert deferred[0]["website_url"] == ""
            diagnostics = read_records(report)
            assert [row["name"] for row in diagnostics] == (
                ["Deferred", "Cedar"] if resolver == "matcher" else ["Deferred"]
            )
            assert diagnostics[0]["llm_evidence"] == ""
            return httpx.Response(403, json={"error": {"message": "later request failed"}})
        name = "Cedar" if query.startswith("Cedar") else "Juniper"
        return httpx.Response(200, json={"places": [{
            "id": f"{name.lower()}-id", "displayName": {"text": name},
            "formattedAddress": "28 Jilin Road, Taipei, Taiwan",
            "location": {"latitude": 25.0, "longitude": 121.0}, "primaryType": "restaurant",
        }]})

    client = GooglePlacesClient("test-only", "TEST_KEY", "en", "TW", 1.0, 0, 0.0, 0.0)
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))

    def run_lookup() -> dict[str, int]:
        if resolver == "legacy":
            return resolve_michelin_place_ids_to_paths(source, resolved, unresolved, client, 2, QUERY_MODE_SIMPLE)
        return resolve_michelin_place_ids_with_matcher_to_paths(
            source, resolved, unresolved, client, 2, QUERY_MODE_SIMPLE, None,
        )

    try:
        with pytest.raises(PlacesClientError, match="403"):
            run_lookup()
        assert [row["name"] for row in read_records(resolved)] == ["Saved", "Cedar"]
        resumed = run_lookup()
        repeated = run_lookup()
    finally:
        client.close()

    assert queries == ["Cedar, Taipei, Taiwan", "Juniper, Taipei, Taiwan", "Juniper, Taipei, Taiwan"]
    assert resumed["resolved_rows"] == 1
    assert resumed["skipped_existing_rows"] == 1
    assert repeated["text_search_requests"] == 0
    assert [row["place_id"] for row in read_records(resolved)] == ["saved-id", "cedar-id", "juniper-id"]
    assert [row["name"] for row in read_records(unresolved)] == ["Deferred"]
    assert [row["name"] for row in read_records(report)] == (
        ["Deferred", "Cedar", "Juniper"] if resolver == "matcher" else ["Deferred"]
    )
    assert stack_resolved_place_id_datasets((resolved,), tmp_path / "processed.csv")["output_rows"] == 3
    assert latest_unresolved_csv_path_or_none(tmp_path) == unresolved

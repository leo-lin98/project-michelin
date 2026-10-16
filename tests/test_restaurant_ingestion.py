import csv
import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import duckdb
import httpx

from michelin.config import load_config
from restaurant_ingestion.clients.google_places import (
    BudgetExceededError,
    GooglePlacesClient,
    PlacesClientError,
    ensure_request_budget,
    validate_no_atmosphere_fields,
)
from restaurant_ingestion.config import (
    DETAILS_REQUEST_BUDGET,
    DISCOVERY_QUERIES,
    GOOGLE_LANGUAGE_CODE,
    MICHELIN_PLACE_ID_LANGUAGE_CODE,
    MICHELIN_PLACE_ID_DATASET_CSV,
    PLACE_DETAILS_FIELD_MASK,
    TAIPEI_GRID_EAST,
    TAIPEI_GRID_NORTH,
    TAIPEI_GRID_SOUTH,
    TAIPEI_GRID_TILE_SIZE_DEGREES,
    TAIPEI_GRID_WEST,
    TARGET_ENRICHED_RESTAURANTS,
)
from restaurant_ingestion.ingestion.grid_search import generate_tiles, plan_taipei_discovery
from restaurant_ingestion.ingestion.michelin import plan_michelin_csv_enrichment
from restaurant_ingestion.ingestion.place_id_llm_judge import (
    apply_llm_judgment,
    decide_place_id_match_with_llm,
    parse_place_id_llm_judgment,
)
from restaurant_ingestion.ingestion.place_id_scoring import MatchDecision, ScoredCandidate, generic_source_name
from restaurant_ingestion.ingestion.michelin_place_ids import (
    QUERY_MODE_SIMPLE,
    QUERY_MODE_STRICT,
    build_processed_place_id_dataset,
    latest_unresolved_csv_path,
    next_retry_output_path,
    parse_structured_address,
    plan_michelin_place_id_lookup,
    processed_place_id_input_paths,
    read_source_rows,
    resolve_michelin_place_ids,
    resolve_michelin_place_ids_to_paths,
    resolve_michelin_place_ids_with_matcher_to_paths,
    run_output_path,
    stack_resolved_place_id_directory,
    stack_resolved_place_id_datasets,
    structured_addresses_match,
    taipei_source_rows,
    text_query_for_row,
    unresolved_output_path_for_resolved_output,
)
from restaurant_ingestion.ingestion.progress import crossed_progress_marks, request_progress_report
from restaurant_ingestion.models import PlaceDetails, PlaceSearchResponse, PriceRange
from restaurant_ingestion.storage.duckdb_store import DuckDbStore, format_price_range

MICHELIN_GUIDE_SOURCE_CSV = Path("data/raw/guide/michelin_my_maps.csv")


def test_schema_upserts_restaurant_categories_hours_and_raw_response(tmp_path: Path) -> None:
    store = DuckDbStore(tmp_path / "restaurants.duckdb")
    store.create_schema()
    place = PlaceDetails.model_validate(
        {
            "id": "place-1",
            "displayName": {"text": "Test Restaurant"},
            "formattedAddress": "1 Taipei Road",
            "location": {"latitude": 25.033, "longitude": 121.5654},
            "rating": 4.7,
            "userRatingCount": 123,
            "priceLevel": "PRICE_LEVEL_MODERATE",
            "businessStatus": "OPERATIONAL",
            "primaryType": "restaurant",
            "types": ["restaurant", "food"],
            "googleMapsUri": "https://maps.google.com/?cid=1",
            "regularOpeningHours": {
                "periods": [
                    {"open": {"day": 1, "hour": 9, "minute": 30}, "close": {"day": 1, "hour": 21, "minute": 0}}
                ]
            },
        }
    )

    store.upsert_candidate("place-1", "checkpoint-1")
    store.upsert_restaurant(place, '{"id":"place-1"}', "hash-1", False)
    store.upsert_restaurant(place, '{"id":"place-1"}', "hash-1", False)
    store.log_api_call("call-1", "places/{place_id}", "place_details_enterprise_no_atmosphere", 200, 0, "place-1", None)

    assert store.enriched_restaurant_count() == 1
    assert store.unenriched_candidate_ids(10, 90) == []
    assert store.api_call_count("place_details_enterprise_no_atmosphere") == 1
    store.close()


def test_budget_and_atmosphere_guards() -> None:
    ensure_request_budget(0, 1, 1)

    with pytest.raises(BudgetExceededError):
        ensure_request_budget(1, 1, 1)

    with pytest.raises(ValueError, match="Atmosphere fields"):
        validate_no_atmosphere_fields(("id", "reviews"))


def test_progress_marks_emit_on_crossed_intervals() -> None:
    assert crossed_progress_marks(499, 500, 500) == (500,)
    assert crossed_progress_marks(499, 1001, 500) == (500, 1000)
    assert crossed_progress_marks(500, 999, 500) == ()


def test_google_client_counts_retries_against_attempt_budget() -> None:
    attempts_seen = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts_seen
        attempts_seen += 1
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    client = GooglePlacesClient(
        "test-api-key",
        "GOOGLE_MAPS_API_KEY",
        "zh-TW",
        "TW",
        30.0,
        3,
        0.0,
        0.0,
    )
    client._client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(BudgetExceededError):
        client.place_details("place-1", ("id", "displayName", "location"), 2)

    assert attempts_seen == 2
    client.close()


def test_google_client_exposes_failed_attempts_for_logging() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"message": "forbidden"}})

    client = GooglePlacesClient(
        "test-api-key",
        "GOOGLE_MAPS_API_KEY",
        "zh-TW",
        "TW",
        30.0,
        3,
        0.0,
        0.0,
    )
    client._client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(PlacesClientError) as exc_info:
        client.place_details("place-1", ("id", "displayName", "location"), 3)

    assert exc_info.value.attempts == ({"status_code": 403, "retry_count": 0},)
    client.close()


def test_google_client_searches_place_id_candidates_with_validation_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.headers["X-Goog-FieldMask"] == "places.id,places.formattedAddress,places.location"
        assert payload["textQuery"] == "Sushi Kajin, 28 Jilin Road, Taipei, Taiwan"
        assert payload["pageSize"] == 5
        assert "includedType" not in payload
        assert "strictTypeFiltering" not in payload
        assert "locationRestriction" not in payload
        return httpx.Response(
            200,
            json={
                "places": [
                    {
                        "id": "place-1",
                        "formattedAddress": "28 Jilin Road, Zhongshan District, Taipei",
                        "location": {"latitude": 25.0, "longitude": 121.0},
                    }
                ]
            },
        )

    client = GooglePlacesClient(
        "test-api-key",
        "GOOGLE_MAPS_API_KEY",
        "zh-TW",
        "TW",
        30.0,
        3,
        0.0,
        0.0,
    )
    client._client = httpx.Client(transport=httpx.MockTransport(handler))

    response, attempts = client.search_text_place_candidates("Sushi Kajin, 28 Jilin Road, Taipei, Taiwan", 1)

    assert response.places[0].id == "place-1"
    assert response.places[0].formattedAddress == "28 Jilin Road, Zhongshan District, Taipei"
    assert attempts == ({"status_code": 200, "retry_count": 0},)
    client.close()


def test_google_client_searches_ids_only_without_type_filtering() -> None:
    rectangle = {
        "low": {"latitude": 25.0, "longitude": 121.0},
        "high": {"latitude": 25.1, "longitude": 121.1},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.headers["X-Goog-FieldMask"] == "places.id,nextPageToken"
        assert payload["textQuery"] == "restaurants in Taipei"
        assert "includedType" not in payload
        assert "strictTypeFiltering" not in payload
        assert payload["locationRestriction"] == {"rectangle": rectangle}
        assert payload["pageSize"] == 20
        return httpx.Response(
            200,
            json={"places": [{"id": "place-1"}], "nextPageToken": "next-page"},
        )

    client = GooglePlacesClient(
        "test-api-key",
        "GOOGLE_MAPS_API_KEY",
        "zh-TW",
        "TW",
        30.0,
        3,
        0.0,
        0.0,
    )
    client._client = httpx.Client(transport=httpx.MockTransport(handler))

    response, attempts = client.search_text_ids_only("restaurants in Taipei", rectangle, None, 1)

    assert response.places[0].id == "place-1"
    assert response.nextPageToken == "next-page"
    assert attempts == ({"status_code": 200, "retry_count": 0},)
    client.close()


def test_google_client_searches_place_id_match_candidates_with_scoring_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.headers["X-Goog-FieldMask"] == (
            "places.id,places.displayName,places.formattedAddress,places.location,"
            "places.nationalPhoneNumber,places.internationalPhoneNumber,places.websiteUri,places.primaryType,places.types"
        )
        assert payload["textQuery"] == "Pang, Taipei, Taiwan"
        assert "includedType" not in payload
        assert "strictTypeFiltering" not in payload
        assert payload["pageSize"] == 5
        return httpx.Response(
            200,
            json={
                "places": [
                    {
                        "id": "place-pang",
                        "displayName": {"text": "Pang Taqueria"},
                        "formattedAddress": "181 Wenchang Street, Taipei",
                        "location": {"latitude": 25.0, "longitude": 121.0},
                        "primaryType": "restaurant",
                    }
                ]
            },
        )

    client = GooglePlacesClient(
        "test-api-key",
        "GOOGLE_MAPS_API_KEY",
        "zh-TW",
        "TW",
        30.0,
        3,
        0.0,
        0.0,
    )
    client._client = httpx.Client(transport=httpx.MockTransport(handler))

    response_json, attempts = client.search_text_place_id_match_candidates("Pang, Taipei, Taiwan", 1)

    assert response_json["places"][0]["id"] == "place-pang"
    assert attempts == ({"status_code": 200, "retry_count": 0},)
    client.close()


def test_default_grid_and_cost_limits_are_bounded() -> None:
    discovery_plan = plan_taipei_discovery(
        TAIPEI_GRID_SOUTH,
        TAIPEI_GRID_WEST,
        TAIPEI_GRID_NORTH,
        TAIPEI_GRID_EAST,
        TAIPEI_GRID_TILE_SIZE_DEGREES,
        DISCOVERY_QUERIES,
    )

    assert TARGET_ENRICHED_RESTAURANTS == 3_000
    assert DETAILS_REQUEST_BUDGET == 6_564
    assert "reviews" not in PLACE_DETAILS_FIELD_MASK
    assert "editorialSummary" not in PLACE_DETAILS_FIELD_MASK
    assert len(generate_tiles(TAIPEI_GRID_SOUTH, TAIPEI_GRID_WEST, TAIPEI_GRID_NORTH, TAIPEI_GRID_EAST, TAIPEI_GRID_TILE_SIZE_DEGREES)) > 0
    assert discovery_plan["first_page_requests"] > 0


def test_michelin_plan_dedupes_and_skips_cached_place_details(tmp_path: Path) -> None:
    guide_csv = tmp_path / "michelin.csv"
    guide_csv.write_text(
        "\n".join(
            [
                "place_id,name,michelin_category,year",
                "place-1,Test Restaurant,1 Star,2025",
                "place-1,Test Restaurant Duplicate,1 Star,2025",
                "place-2,Other Restaurant,Bib Gourmand,2025",
            ]
        ),
        encoding="utf-8",
    )
    store = DuckDbStore(tmp_path / "restaurants.duckdb")
    store.create_schema()
    store.upsert_restaurant(cached_place_details("place-1"), '{"id":"place-1"}', "hash-1", True)

    plan = plan_michelin_csv_enrichment(guide_csv, store, 90)

    assert plan == {
        "michelin_rows": 3,
        "unique_michelin_place_ids": 2,
        "michelin_details_requests": 1,
        "cached_place_ids": 1,
    }
    store.close()


def test_michelin_place_id_lookup_filters_both_cities_and_writes_existing_enrichment_shape(tmp_path: Path) -> None:
    source_csv = tmp_path / "michelin_my_maps.csv"
    output_csv = tmp_path / "michelin_taipei_place_ids.csv"
    source_csv.write_text(
        "\n".join(
            [
                "Name,Address,Location,Award,Latitude,Longitude,Url",
                "Sushi Kajin,\"28 Jilin Road, Zhongshan District, Taipei, 104, Taiwan\",\"Taipei, Taiwan\",1 Star,25.0,121.0,https://guide.michelin.com/tw/sushi-kajin",
                "Chia I,\"2 Renhe Road, Gongliao District, New Taipei, Taiwan\",\"New Taipei, Taiwan\",Bib Gourmand,25.0,121.0,https://guide.michelin.com/tw/chia-i",
                "ES:SENZ,\"Mietenkamer Strasse 65, Grassau, Germany\",\"Grassau, Germany\",3 Stars,47.0,12.0,https://guide.michelin.com/de/es-senz",
            ]
        ),
        encoding="utf-8",
    )
    client = StubPlaceIdClient(("place-sushi-kajin", "place-chia-i"))

    plan = plan_michelin_place_id_lookup(source_csv)
    result = resolve_michelin_place_ids(source_csv, output_csv, client, 2, QUERY_MODE_STRICT)

    assert plan == {
        "source_rows": 3,
        "taipei_rows": 2,
        "non_taipei_rows": 1,
        "text_search_requests": 2,
    }
    assert result == {
        "taipei_rows": 2,
        "resolved_rows": 2,
        "unresolved_rows": 0,
        "skipped_existing_rows": 0,
        "skipped_existing_unresolved_rows": 0,
        "text_search_requests": 2,
    }
    assert client.queries == (
        "Sushi Kajin, 28 Jilin Road, Zhongshan District, Taipei, 104, Taiwan, Taipei, Taiwan",
        "Chia I, 2 Renhe Road, Gongliao District, New Taipei, Taiwan, New Taipei, Taiwan",
    )

    rows = output_csv.read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("place_id,name,michelin_category")
    assert "place-sushi-kajin,Sushi Kajin,1 Star" in rows[1]

    assert "place-chia-i,Chia I,Bib Gourmand" in rows[2]


def test_michelin_place_id_lookup_skips_existing_output_rows(tmp_path: Path) -> None:
    source_csv = tmp_path / "michelin_my_maps.csv"
    output_csv = tmp_path / "michelin_taipei_place_ids.csv"
    source_csv.write_text(
        "\n".join(
            [
                "Name,Address,Location,Award,Latitude,Longitude,Url",
                "Sushi Kajin,\"28 Jilin Road, Zhongshan District, Taipei, 104, Taiwan\",\"Taipei, Taiwan\",1 Star,25.0,121.0,https://guide.michelin.com/tw/sushi-kajin",
                "Longtail,\"174 Section 2 Dunhua South Road, Taipei, Taiwan\",\"Taipei, Taiwan\",1 Star,25.0,121.0,https://guide.michelin.com/tw/longtail",
            ]
        ),
        encoding="utf-8",
    )
    output_csv.write_text(
        "\n".join(
            [
                "place_id,name,michelin_category,address,location,latitude,longitude,michelin_url",
                "place-sushi-kajin,Sushi Kajin,1 Star,\"28 Jilin Road, Zhongshan District, Taipei, 104, Taiwan\",\"Taipei, Taiwan\",25.0,121.0,https://guide.michelin.com/tw/sushi-kajin",
            ]
        ),
        encoding="utf-8",
    )
    client = StubPlaceIdClient(("place-longtail",))

    result = resolve_michelin_place_ids(source_csv, output_csv, client, 1, QUERY_MODE_STRICT)

    assert result == {
        "taipei_rows": 2,
        "resolved_rows": 1,
        "unresolved_rows": 0,
        "skipped_existing_rows": 1,
        "skipped_existing_unresolved_rows": 0,
        "text_search_requests": 1,
    }
    assert client.queries == ("Longtail, 174 Section 2 Dunhua South Road, Taipei, Taiwan, Taipei, Taiwan",)


def test_michelin_place_id_lookup_writes_unresolved_rows_without_failing(tmp_path: Path) -> None:
    source_csv = tmp_path / "michelin_my_maps.csv"
    output_csv = tmp_path / "michelin_taipei_place_ids.csv"
    source_csv.write_text(
        "\n".join(
            [
                "Name,Address,Location,Award,Latitude,Longitude,Url",
                "Longtail,\"174, Section 2, Dunhua South Road, Da'an District, Taipei, 106, Taiwan\",\"Taipei, Taiwan\",1 Star,25.0,121.0,https://guide.michelin.com/tw/longtail",
            ]
        ),
        encoding="utf-8",
    )
    client = EmptyPlaceIdClient()

    result = resolve_michelin_place_ids(source_csv, output_csv, client, 1, QUERY_MODE_STRICT)

    assert result == {
        "taipei_rows": 1,
        "resolved_rows": 0,
        "unresolved_rows": 1,
        "skipped_existing_rows": 0,
        "skipped_existing_unresolved_rows": 0,
        "text_search_requests": 1,
    }
    assert output_csv.read_text(encoding="utf-8").splitlines() == [
        "place_id,name,michelin_category,address,location,latitude,longitude,michelin_url"
    ]
    unresolved_rows = (tmp_path / "michelin_taipei_place_ids.unresolved.csv").read_text(encoding="utf-8").splitlines()
    assert unresolved_rows[0].startswith("name,michelin_category,address")
    assert "Longtail" in unresolved_rows[1]
    report_rows = read_csv_dicts(tmp_path / "michelin_taipei_place_ids.validation-report.csv")
    assert report_rows[0]["reason"] == "no_candidates"
    assert report_rows[0]["candidate_count"] == "0"

    retry_result = resolve_michelin_place_ids(source_csv, output_csv, ExplodingPlaceIdClient(), 1, QUERY_MODE_STRICT)

    assert retry_result == {
        "taipei_rows": 1,
        "resolved_rows": 0,
        "unresolved_rows": 0,
        "skipped_existing_rows": 0,
        "skipped_existing_unresolved_rows": 1,
        "text_search_requests": 0,
    }


def test_michelin_place_id_lookup_rejects_nearby_wrong_address_candidate(tmp_path: Path) -> None:
    source_csv = tmp_path / "michelin_my_maps.csv"
    output_csv = tmp_path / "michelin_taipei_place_ids.csv"
    source_csv.write_text(
        "\n".join(
            [
                "Name,Address,Location,Award,Latitude,Longitude,Url",
                "Motoichi,\"11, Alley 27, Lane 216, Section 4, Zhongxiao East Road, Da’an District, Taipei, Taiwan\",\"Taipei, Taiwan\",1 Star,25.039966,121.553788,https://guide.michelin.com/tw/motoichi",
            ]
        ),
        encoding="utf-8",
    )
    client = FixedPlaceCandidatesClient(
        (
            {
                "id": "place-sushi-touryuumon",
                "formattedAddress": "16, Alley 27, Lane 216, Zhongxiao East Road, Da'an District, Taipei, Taiwan",
                "location": {"latitude": 25.039792, "longitude": 121.5540329},
            },
        )
    )

    result = resolve_michelin_place_ids(source_csv, output_csv, client, 1, QUERY_MODE_STRICT)

    assert result == {
        "taipei_rows": 1,
        "resolved_rows": 0,
        "unresolved_rows": 1,
        "skipped_existing_rows": 0,
        "skipped_existing_unresolved_rows": 0,
        "text_search_requests": 1,
    }
    assert output_csv.read_text(encoding="utf-8").splitlines() == [
        "place_id,name,michelin_category,address,location,latitude,longitude,michelin_url"
    ]
    report_rows = read_csv_dicts(tmp_path / "michelin_taipei_place_ids.validation-report.csv")
    assert report_rows[0]["reason"] == "address_mismatch"
    assert report_rows[0]["candidate_place_ids"] == "place-sushi-touryuumon"
    assert report_rows[0]["candidate_addresses"] == "16, Alley 27, Lane 216, Zhongxiao East Road, Da'an District, Taipei, Taiwan"
    assert report_rows[0]["candidate_failed_checks"] == "address_mismatch"


def test_michelin_place_id_lookup_rejects_candidate_without_location(tmp_path: Path) -> None:
    source_csv = tmp_path / "michelin_my_maps.csv"
    output_csv = tmp_path / "michelin_taipei_place_ids.csv"
    source_csv.write_text(
        "\n".join(
            [
                "Name,Address,Location,Award,Latitude,Longitude,Url",
                "Sushi Kajin,\"28 Jilin Road, Zhongshan District, Taipei, 104, Taiwan\",\"Taipei, Taiwan\",1 Star,25.0,121.0,https://guide.michelin.com/tw/sushi-kajin",
            ]
        ),
        encoding="utf-8",
    )
    client = FixedPlaceCandidatesClient(({"id": "place-sushi-kajin", "formattedAddress": "28 Jilin Road, Taipei"},))

    result = resolve_michelin_place_ids(source_csv, output_csv, client, 1, QUERY_MODE_STRICT)

    assert result["resolved_rows"] == 0
    assert result["unresolved_rows"] == 1
    report_rows = read_csv_dicts(tmp_path / "michelin_taipei_place_ids.validation-report.csv")
    assert report_rows[0]["reason"] == "missing_location"
    assert report_rows[0]["candidate_failed_checks"] == "missing_location"


def test_structured_addresses_match_taipei_street_components() -> None:
    assert structured_addresses_match(
        "21, Alley 30, Lane 135, Section 2, Minquan East Road, Zhongshan District, Taipei, Taiwan",
        "No. 21號, Alley 30, Lane 135, Section 2, Minquan E Rd, Zhongshan District, Taipei City",
    )
    assert structured_addresses_match(
        "B3, Regent Hotel, 3, Lane 39, Section 2, Zhongshan North Road, Zhongshan District, Taipei, Taiwan",
        "10491, Taipei City, Zhongshan District, Lane 39, Section 2, Zhongshan N Rd, 3號B3",
    )
    assert structured_addresses_match(
        "85F, Taipei 101, 7, Section 5, Xinyi Road, Xinyi District, Taipei, Taiwan",
        "No. 7, Section 5, Xinyi Rd, Xinyi District, Taipei City, 110",
    )
    assert not structured_addresses_match(
        "11, Alley 27, Lane 216, Section 4, Zhongxiao East Road, Da’an District, Taipei, Taiwan",
        "16, Alley 27, Lane 216, Zhongxiao East Road, Da'an District, Taipei, Taiwan",
    )


def test_parse_structured_address_extracts_house_number_not_floor_or_building() -> None:
    assert parse_structured_address("85F, Taipei 101, 7, Section 5, Xinyi Road, Xinyi District") == {
        "road": "xinyi road",
        "section": "5",
        "lane": None,
        "alley": None,
        "house_numbers": ("7",),
    }


def test_michelin_place_id_lookup_reports_distance_too_far(tmp_path: Path) -> None:
    source_csv = tmp_path / "michelin_my_maps.csv"
    output_csv = tmp_path / "michelin_taipei_place_ids.csv"
    source_csv.write_text(
        "\n".join(
            [
                "Name,Address,Location,Award,Latitude,Longitude,Url",
                "Sushi Kajin,\"28 Jilin Road, Zhongshan District, Taipei, 104, Taiwan\",\"Taipei, Taiwan\",1 Star,25.0,121.0,https://guide.michelin.com/tw/sushi-kajin",
            ]
        ),
        encoding="utf-8",
    )
    client = FixedPlaceCandidatesClient(
        (
            {
                "id": "place-sushi-kajin",
                "formattedAddress": "28 Jilin Road, Taipei",
                "location": {"latitude": 25.1, "longitude": 121.1},
            },
        )
    )

    result = resolve_michelin_place_ids(source_csv, output_csv, client, 1, QUERY_MODE_STRICT)

    assert result["resolved_rows"] == 0
    assert result["unresolved_rows"] == 1
    report_rows = read_csv_dicts(tmp_path / "michelin_taipei_place_ids.validation-report.csv")
    assert report_rows[0]["reason"] == "distance_too_far"
    assert report_rows[0]["candidate_failed_checks"] == "distance_too_far"


def test_michelin_place_id_lookup_accepts_strong_address_match_beyond_strict_distance(tmp_path: Path) -> None:
    source_csv = tmp_path / "michelin_my_maps.csv"
    output_csv = tmp_path / "michelin_taipei_place_ids.csv"
    source_csv.write_text(
        "\n".join(
            [
                "Name,Address,Location,Award,Latitude,Longitude,Url",
                "Soft Power,\"21, Alley 30, Lane 135, Section 2, Minquan East Road, Zhongshan District, Taipei, Taiwan\",\"Taipei, Taiwan\",Bib Gourmand,25.06425,121.53611,https://guide.michelin.com/tw/soft-power",
            ]
        ),
        encoding="utf-8",
    )
    client = FixedPlaceCandidatesClient(
        (
            {
                "id": "place-soft-power",
                "formattedAddress": "No. 21號, Alley 30, Lane 135, Section 2, Minquan E Rd, Zhongshan District, Taipei City",
                "location": {"latitude": 25.066, "longitude": 121.53611},
            },
        )
    )

    result = resolve_michelin_place_ids(source_csv, output_csv, client, 1, QUERY_MODE_SIMPLE)

    assert result["resolved_rows"] == 1
    assert result["unresolved_rows"] == 0
    assert "place-soft-power,Soft Power,Bib Gourmand" in output_csv.read_text(encoding="utf-8")


def test_michelin_place_id_matcher_resolves_initial_lookup_with_deterministic_v4(tmp_path: Path) -> None:
    source_csv = tmp_path / "michelin_my_maps.csv"
    output_csv = tmp_path / "resolved" / "base.csv"
    unresolved_csv = tmp_path / "unresolved" / "base.unresolved.csv"
    source_csv.write_text(
        "\n".join(
            [
                "Name,Address,Location,Award,Latitude,Longitude,PhoneNumber,WebsiteUrl,Url",
                "Sushi Kajin,\"28 Jilin Road, Zhongshan District, Taipei, Taiwan\",\"Taipei, Taiwan\",1 Star,25.0,121.0,02 1234 5678,https://sushi-kajin.example/taipei,https://guide.michelin.com/tw/sushi-kajin",
            ]
        ),
        encoding="utf-8",
    )
    client = FixedPlaceIdMatchCandidatesClient(
        (
            {
                "id": "place-sushi-kajin",
                "displayName": {"text": "Sushi Kajin"},
                "formattedAddress": "28 Jilin Road, Zhongshan District, Taipei, Taiwan",
                "location": {"latitude": 25.0, "longitude": 121.0},
                "nationalPhoneNumber": "02 1234 5678",
                "websiteUri": "https://sushi-kajin.example/taipei",
                "primaryType": "restaurant",
            },
        )
    )

    result = resolve_michelin_place_ids_with_matcher_to_paths(
        source_csv,
        output_csv,
        unresolved_csv,
        client,
        1,
        QUERY_MODE_SIMPLE,
        None,
    )

    assert result == {
        "taipei_rows": 1,
        "resolved_rows": 1,
        "unresolved_rows": 0,
        "skipped_existing_rows": 0,
        "skipped_existing_unresolved_rows": 0,
        "text_search_requests": 1,
        "llm_judged_rows": 0,
    }
    assert "place-sushi-kajin,Sushi Kajin,1 Star" in output_csv.read_text(encoding="utf-8")
    assert unresolved_csv.read_text(encoding="utf-8").splitlines() == [
        "name,michelin_category,address,location,latitude,longitude,michelin_url,phone_number,website_url,text_query"
    ]


def test_michelin_place_id_matcher_uses_llm_for_initial_manual_review(tmp_path: Path) -> None:
    source_csv = tmp_path / "michelin_my_maps.csv"
    output_csv = tmp_path / "resolved" / "base.csv"
    unresolved_csv = tmp_path / "unresolved" / "base.unresolved.csv"
    source_csv.write_text(
        "\n".join(
            [
                "Name,Address,Location,Award,Latitude,Longitude,Url",
                "Pang,\"181 Wenchang Street, Da'an District, Taipei, Taiwan\",\"Taipei, Taiwan\",Selected Restaurants,25.03261,121.55215,https://guide.michelin.com/tw/pang",
            ]
        ),
        encoding="utf-8",
    )
    client = FixedPlaceIdMatchCandidatesClient(
        (
            {
                "id": "place-pang",
                "displayName": {"text": "Pang Taqueria"},
                "formattedAddress": "181 Wenchang Street, Taipei",
                "location": {"latitude": 25.0339, "longitude": 121.55215},
                "primaryType": "restaurant",
            },
        )
    )
    calls: list[tuple[dict[str, object], tuple[dict[str, object], ...]]] = []

    def judge(row: dict[str, object], candidates: tuple[dict[str, object], ...]) -> dict[str, str | tuple[str, ...]]:
        calls.append((row, candidates))
        return {
            "recommended_outcome": "place_id",
            "recommended_place_id": "place-pang",
            "confidence": "high",
            "evidence": ("source name is contained in candidate name",),
            "conflicts": (),
        }

    result = resolve_michelin_place_ids_with_matcher_to_paths(
        source_csv,
        output_csv,
        unresolved_csv,
        client,
        1,
        QUERY_MODE_SIMPLE,
        judge,
    )

    assert result["resolved_rows"] == 1
    assert result["unresolved_rows"] == 0
    assert result["llm_judged_rows"] == 1
    assert len(calls) == 1
    assert calls[0][1][0]["id"] == "place-pang"
    assert "place-pang,Pang,Selected Restaurants" in output_csv.read_text(encoding="utf-8")
    report_rows = read_csv_dicts(tmp_path / "unresolved" / "base.validation-report.csv")
    assert len(report_rows) == 1
    assert report_rows[0]["predicted_outcome"] == "place_id"
    assert report_rows[0]["predicted_place_id"] == "place-pang"
    assert report_rows[0]["llm_recommended_outcome"] == "place_id"
    assert report_rows[0]["llm_recommended_place_id"] == "place-pang"
    assert report_rows[0]["llm_confidence"] == "high"


def test_matcher_escalates_to_address_query_and_resolves_over_unioned_pool(tmp_path: Path) -> None:
    source_csv = tmp_path / "michelin_my_maps.csv"
    output_csv = tmp_path / "resolved" / "base.csv"
    unresolved_csv = tmp_path / "unresolved" / "base.unresolved.csv"
    source_csv.write_text(
        "\n".join(
            [
                "Name,Address,Location,Award,Latitude,Longitude,Url",
                "Taben,\"28 Jilin Road, Zhongshan District, Taipei, Taiwan\",\"Taipei, Taiwan\",1 Star,25.0,121.0,https://guide.michelin.com/tw/taben",
            ]
        ),
        encoding="utf-8",
    )
    simple_query = "Taben, Taipei, Taiwan"
    strict_query = "Taben, 28 Jilin Road, Zhongshan District, Taipei, Taiwan, Taipei, Taiwan"
    client = QueryKeyedMatchCandidatesClient(
        {
            simple_query: (
                {
                    "id": "place-far-namesake",
                    "displayName": {"text": "Taben Annex"},
                    "formattedAddress": "999 Bade Road, Taipei",
                    "location": {"latitude": 25.2, "longitude": 121.6},
                    "primaryType": "restaurant",
                },
            ),
            strict_query: (
                {
                    "id": "place-taben",
                    "displayName": {"text": "Taben"},
                    "formattedAddress": "28 Jilin Road, Zhongshan District, Taipei, Taiwan",
                    "location": {"latitude": 25.0, "longitude": 121.0},
                    "primaryType": "restaurant",
                },
            ),
        }
    )

    result = resolve_michelin_place_ids_with_matcher_to_paths(
        source_csv, output_csv, unresolved_csv, client, 2, QUERY_MODE_SIMPLE, None
    )

    assert result["resolved_rows"] == 1
    assert result["unresolved_rows"] == 0
    assert result["llm_judged_rows"] == 0
    assert result["text_search_requests"] == 2
    assert client.queries == (simple_query, strict_query)
    assert "place-taben,Taben,1 Star" in output_csv.read_text(encoding="utf-8")


def test_matcher_does_not_escalate_when_primary_query_resolves(tmp_path: Path) -> None:
    source_csv = tmp_path / "michelin_my_maps.csv"
    output_csv = tmp_path / "resolved" / "base.csv"
    unresolved_csv = tmp_path / "unresolved" / "base.unresolved.csv"
    source_csv.write_text(
        "\n".join(
            [
                "Name,Address,Location,Award,Latitude,Longitude,Url",
                "Taben,\"28 Jilin Road, Zhongshan District, Taipei, Taiwan\",\"Taipei, Taiwan\",1 Star,25.0,121.0,https://guide.michelin.com/tw/taben",
            ]
        ),
        encoding="utf-8",
    )
    simple_query = "Taben, Taipei, Taiwan"
    client = QueryKeyedMatchCandidatesClient(
        {
            simple_query: (
                {
                    "id": "place-taben",
                    "displayName": {"text": "Taben"},
                    "formattedAddress": "28 Jilin Road, Zhongshan District, Taipei, Taiwan",
                    "location": {"latitude": 25.0, "longitude": 121.0},
                    "primaryType": "restaurant",
                },
            )
        }
    )

    result = resolve_michelin_place_ids_with_matcher_to_paths(
        source_csv, output_csv, unresolved_csv, client, 2, QUERY_MODE_SIMPLE, None
    )

    assert result["resolved_rows"] == 1
    assert result["text_search_requests"] == 1
    assert client.queries == (simple_query,)


def test_matcher_llm_adjudicates_over_unioned_pool_after_escalation(tmp_path: Path) -> None:
    source_csv = tmp_path / "michelin_my_maps.csv"
    output_csv = tmp_path / "resolved" / "base.csv"
    unresolved_csv = tmp_path / "unresolved" / "base.unresolved.csv"
    source_csv.write_text(
        "\n".join(
            [
                "Name,Address,Location,Award,Latitude,Longitude,Url",
                "Taben,\"28 Jilin Road, Zhongshan District, Taipei, Taiwan\",\"Taipei, Taiwan\",1 Star,25.0,121.0,https://guide.michelin.com/tw/taben",
            ]
        ),
        encoding="utf-8",
    )
    simple_query = "Taben, Taipei, Taiwan"
    strict_query = "Taben, 28 Jilin Road, Zhongshan District, Taipei, Taiwan, Taipei, Taiwan"
    client = QueryKeyedMatchCandidatesClient(
        {
            simple_query: (
                {
                    "id": "place-a",
                    "displayName": {"text": "Taben Annex"},
                    "formattedAddress": "999 Bade Road, Taipei",
                    "location": {"latitude": 25.2, "longitude": 121.6},
                    "primaryType": "restaurant",
                },
            ),
            strict_query: (
                {
                    "id": "place-b",
                    "displayName": {"text": "Taben House"},
                    "formattedAddress": "30 Jilin Road, Zhongshan District, Taipei, Taiwan",
                    "location": {"latitude": 25.0, "longitude": 121.0},
                    "primaryType": "restaurant",
                },
            ),
        }
    )
    seen_candidate_ids: list[frozenset[str]] = []

    def judge(row: dict[str, object], candidates: tuple[dict[str, object], ...]) -> dict[str, str | tuple[str, ...]]:
        seen_candidate_ids.append(frozenset(str(candidate["id"]) for candidate in candidates))
        return {
            "recommended_outcome": "place_id",
            "recommended_place_id": "place-b",
            "confidence": "high",
            "evidence": ("address and name align with the escalation candidate",),
            "conflicts": (),
        }

    result = resolve_michelin_place_ids_with_matcher_to_paths(
        source_csv, output_csv, unresolved_csv, client, 2, QUERY_MODE_SIMPLE, judge
    )

    assert result["resolved_rows"] == 1
    assert result["llm_judged_rows"] == 1
    assert result["text_search_requests"] == 2
    assert seen_candidate_ids == [frozenset({"place-a", "place-b"})]
    assert "place-b,Taben,1 Star" in output_csv.read_text(encoding="utf-8")


def test_michelin_place_id_lookup_reports_ambiguous_candidates(tmp_path: Path) -> None:
    source_csv = tmp_path / "michelin_my_maps.csv"
    output_csv = tmp_path / "michelin_taipei_place_ids.csv"
    source_csv.write_text(
        "\n".join(
            [
                "Name,Address,Location,Award,Latitude,Longitude,Url",
                "Sushi Kajin,\"28 Jilin Road, Zhongshan District, Taipei, 104, Taiwan\",\"Taipei, Taiwan\",1 Star,25.0,121.0,https://guide.michelin.com/tw/sushi-kajin",
            ]
        ),
        encoding="utf-8",
    )
    client = FixedPlaceCandidatesClient(
        (
            {
                "id": "place-sushi-kajin-1",
                "formattedAddress": "28 Jilin Road, Taipei",
                "location": {"latitude": 25.0, "longitude": 121.0},
            },
            {
                "id": "place-sushi-kajin-2",
                "formattedAddress": "28 Jilin Road, Taipei",
                "location": {"latitude": 25.0, "longitude": 121.0},
            },
        )
    )

    result = resolve_michelin_place_ids(source_csv, output_csv, client, 1, QUERY_MODE_STRICT)

    assert result["resolved_rows"] == 0
    assert result["unresolved_rows"] == 1
    report_rows = read_csv_dicts(tmp_path / "michelin_taipei_place_ids.validation-report.csv")
    assert report_rows[0]["reason"] == "ambiguous_candidates"
    assert report_rows[0]["valid_candidate_count"] == "2"
    assert report_rows[0]["valid_candidate_place_ids"] == "place-sushi-kajin-1|place-sushi-kajin-2"


def test_michelin_place_id_lookup_accepts_unresolved_csv_as_retry_source(tmp_path: Path) -> None:
    source_csv = tmp_path / "michelin_taipei_place_ids.unresolved.csv"
    output_csv = tmp_path / "michelin_taipei_place_ids.retry.csv"
    source_csv.write_text(
        "\n".join(
            [
                "name,michelin_category,address,location,latitude,longitude,michelin_url,text_query",
                "Longtail,Selected Restaurants,\"174, Section 2, Dunhua South Road, Da'an District, Taipei, 106, Taiwan\",\"Taipei, Taiwan\",25.0,121.0,https://guide.michelin.com/tw/longtail,\"Longtail, Taipei, Taiwan\"",
            ]
        ),
        encoding="utf-8",
    )
    client = StubPlaceIdClient(("place-longtail",))

    result = resolve_michelin_place_ids(source_csv, output_csv, client, 1, QUERY_MODE_STRICT)

    assert result == {
        "taipei_rows": 1,
        "resolved_rows": 1,
        "unresolved_rows": 0,
        "skipped_existing_rows": 0,
        "skipped_existing_unresolved_rows": 0,
        "text_search_requests": 1,
    }
    assert client.queries == (
        "Longtail, 174, Section 2, Dunhua South Road, Da'an District, Taipei, 106, Taiwan, Taipei, Taiwan",
    )
    assert "place-longtail,Longtail,Selected Restaurants" in output_csv.read_text(encoding="utf-8")


def test_michelin_place_id_lookup_supports_simple_retry_query_mode(tmp_path: Path) -> None:
    source_csv = tmp_path / "michelin_taipei_place_ids.unresolved.csv"
    output_csv = tmp_path / "michelin_taipei_place_ids.retry.csv"
    source_csv.write_text(
        "\n".join(
            [
                "name,michelin_category,address,location,latitude,longitude,michelin_url,text_query",
                "Wu Wang Tsai Chi,Bib Gourmand,\"Nanjichang Night Market, 29, Lane 313, Section 2, Zhonghua Road, Zhongzheng District, Taipei, 100, Taiwan\",\"Taipei, Taiwan\",25.0,121.0,https://guide.michelin.com/tw/wu-wang-tsai-chi,\"Wu Wang Tsai Chi, Taipei, Taiwan\"",
            ]
        ),
        encoding="utf-8",
    )
    client = StubPlaceIdClient(("place-wu-wang-tsai-chi",))

    result = resolve_michelin_place_ids(source_csv, output_csv, client, 1, QUERY_MODE_SIMPLE)

    assert result == {
        "taipei_rows": 1,
        "resolved_rows": 1,
        "unresolved_rows": 0,
        "skipped_existing_rows": 0,
        "skipped_existing_unresolved_rows": 0,
        "text_search_requests": 1,
    }
    assert client.queries == ("Wu Wang Tsai Chi, Taipei, Taiwan",)


def test_stacks_resolved_michelin_place_id_outputs(tmp_path: Path) -> None:
    first_csv = tmp_path / "michelin_taipei_place_ids.csv"
    second_csv = tmp_path / "michelin_taipei_place_ids.simple-retry3.csv"
    output_csv = tmp_path / "processed" / "michelin_taipei_place_ids.csv"
    first_csv.write_text(
        "\n".join(
            [
                "place_id,name,michelin_category,address,location,latitude,longitude,michelin_url",
                "place-sushi-kajin,Sushi Kajin,1 Star,\"28 Jilin Road, Taipei, Taiwan\",\"Taipei, Taiwan\",25.0,121.0,https://guide.michelin.com/tw/sushi-kajin",
            ]
        ),
        encoding="utf-8",
    )
    second_csv.write_text(
        "\n".join(
            [
                "michelin_url,longitude,latitude,location,address,michelin_category,name,place_id",
                "https://guide.michelin.com/tw/mochi-baby,121.1,25.1,\"Taipei, Taiwan\",\"111 Raohe Street, Taipei, Taiwan\",Selected Restaurants,Mochi Baby,place-mochi-baby",
            ]
        ),
        encoding="utf-8",
    )

    result = stack_resolved_place_id_datasets((first_csv, second_csv), output_csv)

    assert result == {"input_files": 2, "output_rows": 2}
    assert output_csv.read_text(encoding="utf-8").splitlines() == [
        "place_id,name,michelin_category,address,location,latitude,longitude,michelin_url",
        "place-sushi-kajin,Sushi Kajin,1 Star,\"28 Jilin Road, Taipei, Taiwan\",\"Taipei, Taiwan\",25.0,121.0,https://guide.michelin.com/tw/sushi-kajin",
        "place-mochi-baby,Mochi Baby,Selected Restaurants,\"111 Raohe Street, Taipei, Taiwan\",\"Taipei, Taiwan\",25.1,121.1,https://guide.michelin.com/tw/mochi-baby",
    ]
    assert_place_ids_are_distinct(output_csv)


def test_resolved_michelin_place_id_stack_rejects_schema_mismatch(tmp_path: Path) -> None:
    input_csv = tmp_path / "michelin_taipei_place_ids.csv"
    output_csv = tmp_path / "processed" / "michelin_taipei_place_ids.csv"
    input_csv.write_text(
        "\n".join(
            [
                "place_id,name,address,location,latitude,longitude,michelin_url,unexpected",
                "place-sushi-kajin,Sushi Kajin,\"28 Jilin Road, Taipei, Taiwan\",\"Taipei, Taiwan\",25.0,121.0,https://guide.michelin.com/tw/sushi-kajin,value",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema mismatch"):
        stack_resolved_place_id_datasets((input_csv,), output_csv)


def test_resolve_michelin_place_ids_writes_to_explicit_unresolved_directory(tmp_path: Path) -> None:
    source_csv = tmp_path / "michelin_my_maps.csv"
    output_csv = tmp_path / "resolved" / "base.csv"
    unresolved_csv = tmp_path / "unresolved" / "base.unresolved.csv"
    source_csv.write_text(
        "\n".join(
            [
                "Name,Address,Location,Award,Latitude,Longitude,Url",
                "Longtail,\"174, Section 2, Dunhua South Road, Da'an District, Taipei, 106, Taiwan\",\"Taipei, Taiwan\",1 Star,25.0,121.0,https://guide.michelin.com/tw/longtail",
            ]
        ),
        encoding="utf-8",
    )

    result = resolve_michelin_place_ids_to_paths(
        source_csv,
        output_csv,
        unresolved_csv,
        EmptyPlaceIdClient(),
        1,
        QUERY_MODE_STRICT,
    )

    assert result["unresolved_rows"] == 1
    assert output_csv.exists()
    assert unresolved_csv.exists()
    assert "Longtail" in unresolved_csv.read_text(encoding="utf-8")


def test_latest_unresolved_csv_path_uses_latest_nonempty_file(tmp_path: Path) -> None:
    unresolved_dir = tmp_path / "unresolved"
    unresolved_dir.mkdir()
    (unresolved_dir / "base.unresolved.csv").write_text(
        "\n".join(
            [
                "name,michelin_category,address,location,latitude,longitude,michelin_url,text_query",
                "Longtail,Selected Restaurants,\"174 Section 2 Dunhua South Road\",\"Taipei, Taiwan\",25.0,121.0,https://guide.michelin.com/tw/longtail,\"Longtail, Taipei, Taiwan\"",
            ]
        ),
        encoding="utf-8",
    )
    (unresolved_dir / "retry-001.unresolved.csv").write_text(
        "name,michelin_category,address,location,latitude,longitude,michelin_url,text_query\n",
        encoding="utf-8",
    )
    (unresolved_dir / "retry-001.validation-report.csv").write_text(
        "name,michelin_category,address,location,latitude,longitude,michelin_url,text_query,reason,candidate_count,valid_candidate_count,candidate_place_ids,valid_candidate_place_ids,candidate_addresses,candidate_distances_meters,candidate_failed_checks\n",
        encoding="utf-8",
    )
    (unresolved_dir / "michelin_taipei_place_ids.unresolved.csv").write_text(
        "\n".join(
            [
                "name,michelin_category,address,location,latitude,longitude,michelin_url,text_query",
                "Legacy,Selected Restaurants,\"1 Legacy Road\",\"Taipei, Taiwan\",25.0,121.0,https://guide.michelin.com/tw/legacy,\"Legacy, Taipei, Taiwan\"",
            ]
        ),
        encoding="utf-8",
    )
    (unresolved_dir / "retry-002.unresolved.csv").write_text(
        "\n".join(
            [
                "name,michelin_category,address,location,latitude,longitude,michelin_url,text_query",
                "Motoichi,1 Star,\"11 Alley 27 Lane 216\",\"Taipei, Taiwan\",25.0,121.0,https://guide.michelin.com/tw/motoichi,\"Motoichi, Taipei, Taiwan\"",
            ]
        ),
        encoding="utf-8",
    )

    assert latest_unresolved_csv_path(unresolved_dir) == unresolved_dir / "retry-002.unresolved.csv"


def test_retry_and_run_paths_are_directory_based(tmp_path: Path) -> None:
    resolved_dir = tmp_path / "resolved"
    unresolved_dir = tmp_path / "unresolved"
    resolved_dir.mkdir()
    (resolved_dir / "base.csv").write_text("", encoding="utf-8")
    (resolved_dir / "retry-001.csv").write_text("", encoding="utf-8")

    retry_output = next_retry_output_path(resolved_dir)

    assert retry_output == resolved_dir / "retry-002.csv"
    assert run_output_path(resolved_dir, "base") == resolved_dir / "base.csv"
    assert unresolved_output_path_for_resolved_output(retry_output, unresolved_dir) == unresolved_dir / "retry-002.unresolved.csv"


def test_stacks_resolved_michelin_place_id_directory(tmp_path: Path) -> None:
    resolved_dir = tmp_path / "resolved"
    output_csv = tmp_path / "processed" / "michelin_taipei_place_ids.csv"
    resolved_dir.mkdir()
    (resolved_dir / "base.csv").write_text(
        "\n".join(
            [
                "place_id,name,michelin_category,address,location,latitude,longitude,michelin_url",
                "place-sushi-kajin,Sushi Kajin,1 Star,\"28 Jilin Road, Taipei\",\"Taipei, Taiwan\",25.0,121.0,https://guide.michelin.com/tw/sushi-kajin",
            ]
        ),
        encoding="utf-8",
    )
    (resolved_dir / "retry-001.csv").write_text(
        "\n".join(
            [
                "place_id,name,michelin_category,address,location,latitude,longitude,michelin_url",
                "place-longtail,Longtail,Selected Restaurants,\"174 Dunhua South Road, Taipei\",\"Taipei, Taiwan\",25.1,121.1,https://guide.michelin.com/tw/longtail",
            ]
        ),
        encoding="utf-8",
    )

    result = stack_resolved_place_id_directory(resolved_dir, output_csv)

    assert result == {"input_files": 2, "output_rows": 2}
    assert output_csv.exists()
    assert_place_ids_are_distinct(output_csv)


def test_processed_place_id_dataset_uses_base_only_when_no_unresolved_rows(tmp_path: Path) -> None:
    resolved_dir = tmp_path / "resolved"
    unresolved_dir = tmp_path / "unresolved"
    output_csv = tmp_path / "processed" / "michelin_taipei_place_ids.csv"
    resolved_dir.mkdir()
    unresolved_dir.mkdir()
    (resolved_dir / "base.csv").write_text(
        "\n".join(
            [
                "place_id,name,michelin_category,address,location,latitude,longitude,michelin_url",
                "place-sushi-kajin,Sushi Kajin,1 Star,\"28 Jilin Road, Taipei\",\"Taipei, Taiwan\",25.0,121.0,https://guide.michelin.com/tw/sushi-kajin",
            ]
        ),
        encoding="utf-8",
    )
    (resolved_dir / "retry-001.csv").write_text(
        "\n".join(
            [
                "place_id,name,michelin_category,address,location,latitude,longitude,michelin_url",
                "place-stale-retry,Stale Retry,Selected Restaurants,\"1 Stale Road, Taipei\",\"Taipei, Taiwan\",25.1,121.1,https://guide.michelin.com/tw/stale-retry",
            ]
        ),
        encoding="utf-8",
    )
    (unresolved_dir / "base.unresolved.csv").write_text(
        "name,michelin_category,address,location,latitude,longitude,michelin_url,text_query\n",
        encoding="utf-8",
    )

    input_paths = processed_place_id_input_paths(resolved_dir, unresolved_dir)
    result = build_processed_place_id_dataset(resolved_dir, unresolved_dir, output_csv)

    assert input_paths == (resolved_dir / "base.csv",)
    assert result == {"input_files": 1, "output_rows": 1, "mode": "base-only"}
    assert "place-stale-retry" not in output_csv.read_text(encoding="utf-8")
    assert_place_ids_are_distinct(output_csv)


def test_processed_place_id_dataset_stacks_when_latest_unresolved_was_retried(tmp_path: Path) -> None:
    resolved_dir = tmp_path / "resolved"
    unresolved_dir = tmp_path / "unresolved"
    output_csv = tmp_path / "processed" / "michelin_taipei_place_ids.csv"
    resolved_dir.mkdir()
    unresolved_dir.mkdir()
    (resolved_dir / "base.csv").write_text(
        "\n".join(
            [
                "place_id,name,michelin_category,address,location,latitude,longitude,michelin_url",
                "place-sushi-kajin,Sushi Kajin,1 Star,\"28 Jilin Road, Taipei\",\"Taipei, Taiwan\",25.0,121.0,https://guide.michelin.com/tw/sushi-kajin",
            ]
        ),
        encoding="utf-8",
    )
    (resolved_dir / "retry-001.csv").write_text(
        "\n".join(
            [
                "place_id,name,michelin_category,address,location,latitude,longitude,michelin_url",
                "place-longtail,Longtail,Selected Restaurants,\"174 Dunhua South Road, Taipei\",\"Taipei, Taiwan\",25.1,121.1,https://guide.michelin.com/tw/longtail",
            ]
        ),
        encoding="utf-8",
    )
    (unresolved_dir / "base.unresolved.csv").write_text(
        "\n".join(
            [
                "name,michelin_category,address,location,latitude,longitude,michelin_url,text_query",
                "Longtail,Selected Restaurants,\"174 Dunhua South Road, Taipei\",\"Taipei, Taiwan\",25.1,121.1,https://guide.michelin.com/tw/longtail,\"Longtail, Taipei, Taiwan\"",
            ]
        ),
        encoding="utf-8",
    )
    (unresolved_dir / "retry-001.unresolved.csv").write_text(
        "name,michelin_category,address,location,latitude,longitude,michelin_url,text_query\n",
        encoding="utf-8",
    )

    result = build_processed_place_id_dataset(resolved_dir, unresolved_dir, output_csv)

    assert result == {"input_files": 2, "output_rows": 2, "mode": "stack-resolved"}
    assert_place_ids_are_distinct(output_csv)


def test_processed_michelin_place_id_csv_has_unique_place_ids() -> None:
    if not MICHELIN_PLACE_ID_DATASET_CSV.exists():
        pytest.skip("Local-only processed Michelin identity dataset is unavailable")
    rows = read_csv_dicts(MICHELIN_PLACE_ID_DATASET_CSV)
    place_ids = tuple(row["place_id"] for row in rows)

    assert len(place_ids) == len(set(place_ids))


def test_processed_michelin_place_id_csv_matches_unique_taipei_guide_restaurant_count() -> None:
    if not MICHELIN_GUIDE_SOURCE_CSV.exists() or not MICHELIN_PLACE_ID_DATASET_CSV.exists():
        pytest.skip("Local-only Guide snapshot or processed identity dataset is unavailable")
    source_rows = taipei_source_rows(read_source_rows(MICHELIN_GUIDE_SOURCE_CSV))
    unique_michelin_urls = {str(row["Url"]).strip() for row in source_rows}
    processed_rows = read_csv_dicts(MICHELIN_PLACE_ID_DATASET_CSV)

    config = load_config(Path("config/pipeline.yaml"), Path("config/features.yaml")).pipeline
    excluded = set(config.eligibility.unresolved_guide_exclusions)
    assert excluded <= unique_michelin_urls
    assert {row["michelin_url"] for row in processed_rows} == unique_michelin_urls - excluded


def test_processed_place_id_dataset_rejects_remaining_unresolved_rows(tmp_path: Path) -> None:
    resolved_dir = tmp_path / "resolved"
    unresolved_dir = tmp_path / "unresolved"
    output_csv = tmp_path / "processed" / "michelin_taipei_place_ids.csv"
    resolved_dir.mkdir()
    unresolved_dir.mkdir()
    (resolved_dir / "base.csv").write_text(
        "\n".join(
            [
                "place_id,name,michelin_category,address,location,latitude,longitude,michelin_url",
                "place-sushi-kajin,Sushi Kajin,1 Star,\"28 Jilin Road, Taipei\",\"Taipei, Taiwan\",25.0,121.0,https://guide.michelin.com/tw/sushi-kajin",
            ]
        ),
        encoding="utf-8",
    )
    (unresolved_dir / "base.unresolved.csv").write_text(
        "\n".join(
            [
                "name,michelin_category,address,location,latitude,longitude,michelin_url,text_query",
                "Longtail,Selected Restaurants,\"174 Dunhua South Road, Taipei\",\"Taipei, Taiwan\",25.1,121.1,https://guide.michelin.com/tw/longtail,\"Longtail, Taipei, Taiwan\"",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unresolved Michelin place_id rows remain"):
        build_processed_place_id_dataset(resolved_dir, unresolved_dir, output_csv)


def test_resolved_stack_rejects_duplicate_place_id_for_different_rows(tmp_path: Path) -> None:
    first_csv = tmp_path / "base.csv"
    second_csv = tmp_path / "retry-001.csv"
    output_csv = tmp_path / "processed" / "michelin_taipei_place_ids.csv"
    first_csv.write_text(
        "\n".join(
            [
                "place_id,name,michelin_category,address,location,latitude,longitude,michelin_url",
                "place-duplicate,Motoichi,1 Star,\"11 Alley 27 Lane 216\",\"Taipei, Taiwan\",25.0,121.0,https://guide.michelin.com/tw/motoichi",
            ]
        ),
        encoding="utf-8",
    )
    second_csv.write_text(
        "\n".join(
            [
                "place_id,name,michelin_category,address,location,latitude,longitude,michelin_url",
                "place-duplicate,Sushi Touryuumon,Selected Restaurants,\"16 Alley 27 Lane 216\",\"Taipei, Taiwan\",25.1,121.1,https://guide.michelin.com/tw/sushi-touryuumon",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple Michelin rows"):
        stack_resolved_place_id_datasets((first_csv, second_csv), output_csv)


def test_michelin_place_id_text_query_requires_name_and_address() -> None:
    row = {
        "Name": "Sushi Kajin",
        "Location": "Taipei, Taiwan",
        "Address": "28 Jilin Road, Zhongshan District, Taipei, 104, Taiwan",
    }

    assert (
        text_query_for_row(row, QUERY_MODE_STRICT)
        == "Sushi Kajin, 28 Jilin Road, Zhongshan District, Taipei, 104, Taiwan, Taipei, Taiwan"
    )
    assert text_query_for_row(row, QUERY_MODE_SIMPLE) == "Sushi Kajin, Taipei, Taiwan"

    with pytest.raises(ValueError, match="missing Address"):
        text_query_for_row({"Name": "Sushi Kajin", "Address": ""}, QUERY_MODE_STRICT)


def test_michelin_place_id_script_requires_explicit_live_cost_controls() -> None:
    script = load_add_michelin_place_ids_script()

    with pytest.raises(RuntimeError, match="confirm-live-api-cost"):
        script.validate_live_run_args(10, False)
    with pytest.raises(RuntimeError, match="max-requests"):
        script.validate_live_run_args(None, True)
    with pytest.raises(RuntimeError, match="at least 1"):
        script.validate_live_run_args(0, True)
    with pytest.raises(RuntimeError, match="<= 250"):
        script.validate_live_run_args(251, True)

    assert script.validate_live_run_args(10, True) == 10


def test_michelin_place_id_script_defaults_to_simple_query_mode() -> None:
    script = load_add_michelin_place_ids_script()

    assert script.DEFAULT_QUERY_MODE == "simple"


def test_michelin_place_id_script_uses_english_places_language(monkeypatch: pytest.MonkeyPatch) -> None:
    script = load_add_michelin_place_ids_script()
    captured_language_codes: list[str] = []

    def fake_from_environment(
        api_key_env_var: str,
        language_code: str,
        region_code: str,
        request_timeout_seconds: float,
        max_retries: int,
        retry_initial_backoff_seconds: float,
        retry_max_backoff_seconds: float,
    ) -> object:
        captured_language_codes.append(language_code)
        return object()

    monkeypatch.setattr(script.GooglePlacesClient, "from_environment", fake_from_environment)

    assert GOOGLE_LANGUAGE_CODE == "zh-TW"
    assert MICHELIN_PLACE_ID_LANGUAGE_CODE == "en"
    assert script.build_client() is not None
    assert captured_language_codes == ["en"]


def test_michelin_place_id_script_resolves_retry_latest_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = load_add_michelin_place_ids_script()
    resolved_dir = tmp_path / "resolved"
    unresolved_dir = tmp_path / "unresolved"
    resolved_dir.mkdir()
    unresolved_dir.mkdir()
    (resolved_dir / "retry-001.csv").write_text("", encoding="utf-8")
    (unresolved_dir / "retry-001.unresolved.csv").write_text(
        "\n".join(
            [
                "name,michelin_category,address,location,latitude,longitude,michelin_url,text_query",
                "Motoichi,1 Star,\"11 Alley 27 Lane 216\",\"Taipei, Taiwan\",25.0,121.0,https://guide.michelin.com/tw/motoichi,\"Motoichi, Taipei, Taiwan\"",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(script, "MICHELIN_PLACE_ID_RESOLVED_DIR", resolved_dir)
    monkeypatch.setattr(script, "MICHELIN_PLACE_ID_UNRESOLVED_DIR", unresolved_dir)

    source_csv, output_csv, unresolved_csv = script.resolve_command_paths(None, None, None, True)

    assert source_csv == unresolved_dir / "retry-001.unresolved.csv"
    assert output_csv == resolved_dir / "retry-002.csv"
    assert unresolved_csv == unresolved_dir / "retry-002.unresolved.csv"


def test_llm_place_id_judgment_accepts_high_confidence_candidate_for_manual_review() -> None:
    deterministic_decision = manual_review_decision(("candidate-place-id",))
    judgment = parse_place_id_llm_judgment(
        json.dumps(
            {
                "recommended_outcome": "place_id",
                "recommended_place_id": "candidate-place-id",
                "confidence": "high",
                "evidence": ["candidate name and address match source"],
                "conflicts": [],
            }
        )
    )

    decision = apply_llm_judgment(deterministic_decision, judgment)

    assert decision.outcome == "place_id"
    assert decision.place_id == "candidate-place-id"
    assert decision.reason == "insufficient_signals;llm_adjudicated"


def test_llm_place_id_judgment_rejects_non_candidate_place_id() -> None:
    deterministic_decision = manual_review_decision(("candidate-place-id",))
    judgment = parse_place_id_llm_judgment(
        json.dumps(
            {
                "recommended_outcome": "place_id",
                "recommended_place_id": "different-place-id",
                "confidence": "high",
                "evidence": ["unsupported place id"],
                "conflicts": [],
            }
        )
    )

    decision = apply_llm_judgment(deterministic_decision, judgment)

    assert decision.outcome == "manual_review"
    assert decision.place_id == ""
    assert decision.reason == "insufficient_signals;llm_place_id_not_in_candidates"


def test_llm_place_id_judgment_does_not_override_deterministic_match() -> None:
    deterministic_decision = MatchDecision(
        "place_id",
        "deterministic-place-id",
        "near+phone",
        (scored_candidate("different-place-id"),),
    )
    judgment = parse_place_id_llm_judgment(
        json.dumps(
            {
                "recommended_outcome": "place_id",
                "recommended_place_id": "different-place-id",
                "confidence": "high",
                "evidence": ["candidate looks plausible"],
                "conflicts": [],
            }
        )
    )

    decision = apply_llm_judgment(deterministic_decision, judgment)

    assert decision == deterministic_decision


def test_llm_place_id_judgment_resolves_on_method_agreement_without_high_confidence() -> None:
    deterministic_decision = manual_review_decision(("agreed-id", "other-id"))
    judgment = parse_place_id_llm_judgment(
        json.dumps(
            {
                "recommended_outcome": "place_id",
                "recommended_place_id": "agreed-id",
                "confidence": "medium",
                "evidence": ["address and name align with the top candidate"],
                "conflicts": [],
            }
        )
    )

    decision = apply_llm_judgment(deterministic_decision, judgment)

    assert decision.outcome == "place_id"
    assert decision.place_id == "agreed-id"
    assert decision.reason == "insufficient_signals;llm_adjudicated"


def test_llm_place_id_judgment_holds_when_medium_confidence_disagrees_with_top_candidate() -> None:
    deterministic_decision = manual_review_decision(("agreed-id", "other-id"))
    judgment = parse_place_id_llm_judgment(
        json.dumps(
            {
                "recommended_outcome": "place_id",
                "recommended_place_id": "other-id",
                "confidence": "medium",
                "evidence": ["weakly held opinion"],
                "conflicts": [],
            }
        )
    )

    decision = apply_llm_judgment(deterministic_decision, judgment)

    assert decision.outcome == "manual_review"
    assert decision.place_id == ""
    assert decision.reason == "insufficient_signals;llm_not_high_confidence"


def test_llm_place_id_judgment_does_not_force_no_safe_match_below_high_confidence() -> None:
    deterministic_decision = manual_review_decision(("candidate-place-id",))
    judgment = parse_place_id_llm_judgment(
        json.dumps(
            {
                "recommended_outcome": "no_safe_match",
                "recommended_place_id": "",
                "confidence": "medium",
                "evidence": [],
                "conflicts": ["candidate is the host hotel, not the restaurant"],
            }
        )
    )

    decision = apply_llm_judgment(deterministic_decision, judgment)

    assert decision.outcome == "manual_review"
    assert decision.reason == "insufficient_signals;llm_not_high_confidence"


def test_generic_source_name_uses_general_rules_not_hardcoded_names() -> None:
    assert generic_source_name("Unnamed Clay Oven Roll") is True
    assert generic_source_name("Chinese Cuisine") is True
    assert generic_source_name("Lin") is True
    assert generic_source_name("Mountain and Sea House") is False
    assert generic_source_name("Sushi Kajin") is False


def test_llm_place_id_matcher_only_calls_judge_for_manual_review() -> None:
    source_row = {
        "name": "Pang",
        "address": "181 Wenchang Street, Taipei",
        "latitude": 25.03261,
        "longitude": 121.55215,
        "phone_number": "",
        "website_url": "",
    }
    fixture_row = {
        "queries": [
            {
                "response": {
                    "places": [
                        {
                            "id": "candidate-place-id",
                            "displayName": {"text": "Pang Taqueria"},
                            "formattedAddress": "181 Wenchang Street, Taipei",
                            "location": {"latitude": 25.0339, "longitude": 121.55215},
                            "primaryType": "restaurant",
                        }
                    ]
                }
            }
        ]
    }
    calls: list[tuple[dict[str, object], tuple[dict[str, object], ...]]] = []

    def judge(row: dict[str, object], candidates: tuple[dict[str, object], ...]) -> dict[str, str | tuple[str, ...]]:
        calls.append((row, candidates))
        return {
            "recommended_outcome": "place_id",
            "recommended_place_id": "candidate-place-id",
            "confidence": "high",
            "evidence": ("source tokens are contained in candidate name",),
            "conflicts": (),
        }

    decision, judgment = decide_place_id_match_with_llm(source_row, fixture_row, judge)

    assert decision.outcome == "place_id"
    assert decision.place_id == "candidate-place-id"
    assert judgment is not None
    assert len(calls) == 1
    assert calls[0][1][0]["id"] == "candidate-place-id"


def test_llm_place_id_matcher_skips_judge_for_deterministic_match() -> None:
    source_row = {
        "name": "Sushi Kajin",
        "address": "28 Jilin Road, Zhongshan District, Taipei, Taiwan",
        "latitude": 25.0,
        "longitude": 121.0,
        "phone_number": "02 1234 5678",
        "website_url": "",
    }
    fixture_row = {
        "queries": [
            {
                "response": {
                    "places": [
                        {
                            "id": "deterministic-place-id",
                            "displayName": {"text": "Sushi Kajin"},
                            "formattedAddress": "28 Jilin Road, Zhongshan District, Taipei, Taiwan",
                            "location": {"latitude": 25.0, "longitude": 121.0},
                            "nationalPhoneNumber": "02 1234 5678",
                            "primaryType": "restaurant",
                        }
                    ]
                }
            }
        ]
    }

    def judge(row: dict[str, object], candidates: tuple[dict[str, object], ...]) -> dict[str, str | tuple[str, ...]]:
        raise AssertionError("Judge should not run after a deterministic match")

    decision, judgment = decide_place_id_match_with_llm(source_row, fixture_row, judge)

    assert decision.outcome == "place_id"
    assert decision.place_id == "deterministic-place-id"
    assert judgment is None


def test_llm_place_id_judgment_requires_strict_schema() -> None:
    with pytest.raises(ValueError, match="schema mismatch"):
        parse_place_id_llm_judgment(
            json.dumps(
                {
                    "recommended_outcome": "manual_review",
                    "recommended_place_id": "",
                    "confidence": "high",
                    "evidence": [],
                }
            )
        )

    with pytest.raises(ValueError, match="recommended_place_id is required"):
        parse_place_id_llm_judgment(
            json.dumps(
                {
                    "recommended_outcome": "place_id",
                    "recommended_place_id": "",
                    "confidence": "high",
                    "evidence": [],
                    "conflicts": [],
                }
            )
        )


def cached_place_details(place_id: str) -> PlaceDetails:
    return PlaceDetails.model_validate(
        {
            "id": place_id,
            "displayName": {"text": "Cached Restaurant"},
            "formattedAddress": "1 Taipei Road",
            "location": {"latitude": 25.033, "longitude": 121.5654},
            "rating": 4.7,
            "userRatingCount": 123,
            "priceLevel": "PRICE_LEVEL_MODERATE",
            "businessStatus": "OPERATIONAL",
            "primaryType": "restaurant",
            "types": ["restaurant", "food"],
            "googleMapsUri": "https://maps.google.com/?cid=1",
        }
    )


def assert_place_ids_are_distinct(csv_path: Path) -> None:
    rows = csv_path.read_text(encoding="utf-8").splitlines()
    place_ids = tuple(row.split(",", maxsplit=1)[0] for row in rows[1:])
    assert len(place_ids) == len(set(place_ids))


def read_csv_dicts(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def manual_review_decision(place_ids: tuple[str, ...]) -> MatchDecision:
    return MatchDecision("manual_review", "", "insufficient_signals", tuple(scored_candidate(place_id) for place_id in place_ids))


def scored_candidate(place_id: str) -> ScoredCandidate:
    return ScoredCandidate(place_id, 4, 0, 100.0, ("nearish", "restaurant_type"))


class StubPlaceIdClient:
    def __init__(self, place_ids: tuple[str, ...]) -> None:
        self._place_ids = place_ids
        self._index = 0
        self._queries: list[str] = []

    @property
    def queries(self) -> tuple[str, ...]:
        return tuple(self._queries)

    def search_text_place_candidates(
        self,
        text_query: str,
        request_budget_remaining: int,
    ) -> tuple[PlaceSearchResponse, tuple[dict[str, int], ...]]:
        self._queries.append(text_query)
        place_id = self._place_ids[self._index]
        self._index += 1
        return PlaceSearchResponse.model_validate(
            {
                "places": [
                    {
                        "id": place_id,
                        "formattedAddress": stub_formatted_address(place_id),
                        "location": {"latitude": 25.0, "longitude": 121.0},
                    }
                ]
            }
        ), (
            {"status_code": 200, "retry_count": 0},
        )


def stub_formatted_address(place_id: str) -> str:
    if place_id == "place-chia-i":
        return "2 Renhe Road, Gongliao District, New Taipei, Taiwan"
    if "longtail" in place_id:
        return "174 Section 2 Dunhua South Road, Taipei, Taiwan"
    if "wu-wang-tsai-chi" in place_id:
        return "Nanjichang Night Market, 29, Lane 313, Section 2, Zhonghua Road, Taipei, Taiwan"
    return "28 Jilin Road, Zhongshan District, Taipei, Taiwan"


class FixedPlaceCandidatesClient:
    def __init__(self, candidates: tuple[dict[str, object], ...]) -> None:
        self._candidates = candidates

    def search_text_place_candidates(
        self,
        text_query: str,
        request_budget_remaining: int,
    ) -> tuple[PlaceSearchResponse, tuple[dict[str, int], ...]]:
        return PlaceSearchResponse.model_validate({"places": list(self._candidates)}), (
            {"status_code": 200, "retry_count": 0},
        )


class FixedPlaceIdMatchCandidatesClient:
    def __init__(self, candidates: tuple[dict[str, object], ...]) -> None:
        self._candidates = candidates

    def search_text_place_id_match_candidates(
        self,
        text_query: str,
        request_budget_remaining: int,
    ) -> tuple[dict[str, object], tuple[dict[str, int], ...]]:
        return {"places": list(self._candidates)}, ({"status_code": 200, "retry_count": 0},)


class QueryKeyedMatchCandidatesClient:
    def __init__(self, places_by_query: dict[str, tuple[dict[str, object], ...]]) -> None:
        self._places_by_query = places_by_query
        self._queries: list[str] = []

    @property
    def queries(self) -> tuple[str, ...]:
        return tuple(self._queries)

    def search_text_place_id_match_candidates(
        self,
        text_query: str,
        request_budget_remaining: int,
    ) -> tuple[dict[str, object], tuple[dict[str, int], ...]]:
        self._queries.append(text_query)
        places = self._places_by_query.get(text_query, ())
        return {"places": list(places)}, ({"status_code": 200, "retry_count": 0},)


class EmptyPlaceIdClient:
    def search_text_place_candidates(
        self,
        text_query: str,
        request_budget_remaining: int,
    ) -> tuple[PlaceSearchResponse, tuple[dict[str, int], ...]]:
        return PlaceSearchResponse.model_validate({"places": []}), ({"status_code": 200, "retry_count": 0},)


class ExplodingPlaceIdClient:
    def search_text_place_candidates(
        self,
        text_query: str,
        request_budget_remaining: int,
    ) -> tuple[PlaceSearchResponse, tuple[dict[str, int], ...]]:
        raise AssertionError("Unexpected live lookup")


def load_add_michelin_place_ids_script() -> object:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "add_michelin_place_ids.py"
    spec = importlib.util.spec_from_file_location("add_michelin_place_ids", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load add_michelin_place_ids script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_backfill_price_fields_script() -> object:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "backfill_price_fields.py"
    spec = importlib.util.spec_from_file_location("backfill_price_fields", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load backfill_price_fields script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def price_details_payload(place_id: str, price_level: str | None, price_range: dict[str, object] | None) -> dict[str, object]:
    return {
        "id": place_id,
        "displayName": {"text": "Test Restaurant"},
        "formattedAddress": "1 Test Road",
        "location": {"latitude": 25.0, "longitude": 121.5},
        "priceLevel": price_level,
        "priceRange": price_range,
    }


def test_format_price_range_renders_both_bounds_as_bare_units() -> None:
    price_range = PriceRange.model_validate(
        {
            "startPrice": {"currencyCode": "TWD", "units": "1"},
            "endPrice": {"currencyCode": "TWD", "units": "200"},
        }
    )

    assert format_price_range(price_range) == "1-200"


@pytest.mark.parametrize(("units", "expected_range"), [("2000", "2000+"), ("500", "500+")])
def test_format_price_range_preserves_open_upper_bound(units: str, expected_range: str) -> None:
    start_only = PriceRange.model_validate({"startPrice": {"currencyCode": "TWD", "units": units}})

    assert format_price_range(start_only) == expected_range


def test_format_price_range_returns_none_for_unusable_and_absent_ranges() -> None:
    end_only = PriceRange.model_validate({"endPrice": {"currencyCode": "TWD", "units": "200"}})
    unitless = PriceRange.model_validate(
        {"startPrice": {"currencyCode": "TWD"}, "endPrice": {"currencyCode": "TWD", "units": "200"}}
    )

    unitless_end = PriceRange.model_validate(
        {"startPrice": {"currencyCode": "TWD", "units": "2000"}, "endPrice": {"currencyCode": "TWD"}}
    )

    assert format_price_range(unitless_end) is None
    assert format_price_range(end_only) is None
    assert format_price_range(unitless) is None
    assert format_price_range(None) is None


def test_create_schema_adds_price_range_column_to_pre_existing_database(tmp_path: Path) -> None:
    database_path = tmp_path / "restaurants.duckdb"
    legacy_connection = duckdb.connect(str(database_path))
    legacy_connection.execute(
        """
        CREATE TABLE restaurants (
            place_id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            address VARCHAR,
            latitude DOUBLE NOT NULL,
            longitude DOUBLE NOT NULL,
            rating DOUBLE,
            review_count INTEGER,
            price_level VARCHAR,
            website VARCHAR,
            phone VARCHAR,
            google_maps_url VARCHAR,
            business_status VARCHAR,
            primary_category VARCHAR,
            is_michelin BOOLEAN NOT NULL,
            last_updated TIMESTAMP NOT NULL
        )
        """
    )
    legacy_connection.close()

    store = DuckDbStore(database_path)
    try:
        store.create_schema()
    finally:
        store.close()

    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        columns = [str(row[0]) for row in connection.execute("DESCRIBE restaurants").fetchall()]
    finally:
        connection.close()

    assert "price_range" in columns


@pytest.mark.parametrize(
    ("range_payload", "expected_range"),
    [
        (
            {"startPrice": {"currencyCode": "TWD", "units": "1"}, "endPrice": {"currencyCode": "TWD", "units": "200"}},
            "1-200",
        ),
        ({"startPrice": {"currencyCode": "TWD", "units": "2000"}}, "2000+"),
    ],
)
def test_upsert_restaurant_overwrites_price_columns_and_stale_raw_response(
    tmp_path: Path, range_payload: dict[str, object], expected_range: str
) -> None:
    database_path = tmp_path / "restaurants.duckdb"
    stale_payload = price_details_payload("place-1", None, None)
    fresh_payload = price_details_payload(
        "place-1",
        "PRICE_LEVEL_INEXPENSIVE",
        range_payload,
    )

    store = DuckDbStore(database_path)
    try:
        store.create_schema()
        store.upsert_restaurant(PlaceDetails.model_validate(stale_payload), json.dumps(stale_payload), "stale-hash", False)
        store.upsert_restaurant(PlaceDetails.model_validate(fresh_payload), json.dumps(fresh_payload), "fresh-hash", False)
    finally:
        store.close()

    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        price_level, price_range = connection.execute("SELECT price_level, price_range FROM restaurants").fetchall()[0]
        raw_rows = connection.execute("SELECT response_hash, response_json FROM place_raw_responses").fetchall()
    finally:
        connection.close()

    assert price_level == "PRICE_LEVEL_INEXPENSIVE"
    assert price_range == expected_range
    assert len(raw_rows) == 1
    assert raw_rows[0][0] == "fresh-hash"
    assert json.loads(raw_rows[0][1])["priceRange"] is not None


def test_missing_price_state_skips_recently_refreshed_rows_and_returns_stalest_first(tmp_path: Path) -> None:
    database_path = tmp_path / "restaurants.duckdb"
    both_bounds = {"startPrice": {"currencyCode": "TWD", "units": "1"}, "endPrice": {"currencyCode": "TWD", "units": "200"}}
    rows = {
        "place-stalest-missing": price_details_payload("place-stalest-missing", None, None),
        "place-newer-missing": price_details_payload("place-newer-missing", None, None),
        "place-recently-tried": price_details_payload("place-recently-tried", None, None),
        "place-already-priced": price_details_payload("place-already-priced", "PRICE_LEVEL_MODERATE", both_bounds),
        "place-open-range": price_details_payload(
            "place-open-range", "PRICE_LEVEL_EXPENSIVE", {"startPrice": {"currencyCode": "TWD", "units": "2000"}}
        ),
        "place-open-range-missing-level": price_details_payload(
            "place-open-range-missing-level", None, {"startPrice": {"currencyCode": "TWD", "units": "2000"}}
        ),
    }

    store = DuckDbStore(database_path)
    try:
        store.create_schema()
        for place_id, payload in rows.items():
            store.upsert_restaurant(PlaceDetails.model_validate(payload), json.dumps(payload), f"hash-{place_id}", False)
    finally:
        store.close()

    ages_in_days = {
        "place-stalest-missing": 200,
        "place-newer-missing": 100,
        "place-recently-tried": 1,
        "place-already-priced": 200,
        "place-open-range": 200,
        "place-open-range-missing-level": 50,
    }
    connection = duckdb.connect(str(database_path))
    try:
        for place_id, age_in_days in ages_in_days.items():
            connection.execute(
                "UPDATE restaurants SET last_updated = ? WHERE place_id = ?",
                [datetime.now(UTC).replace(tzinfo=None) - timedelta(days=age_in_days), place_id],
            )
    finally:
        connection.close()

    store = DuckDbStore(database_path)
    try:
        selected = store.missing_price_state(10, 30)
    finally:
        store.close()

    assert [place_id for place_id, _price_level in selected] == [
        "place-stalest-missing",
        "place-newer-missing",
        "place-open-range-missing-level",
    ]


def test_request_progress_report_reports_throughput_and_eta() -> None:
    report = request_progress_report("price-backfill", 100, 1_000, 50.0)

    assert report["completed"] == 100
    assert report["total"] == 1_000
    assert report["calls_per_second"] == 2.0
    assert report["eta_seconds"] == 450


def test_request_progress_report_survives_a_zero_length_first_interval() -> None:
    report = request_progress_report("price-backfill", 0, 1_000, 0.0)

    assert report["calls_per_second"] == 0.0
    assert report["eta_seconds"] is None


def test_price_outcome_reports_recovered_price_level_and_partial_range() -> None:
    module = load_backfill_price_fields_script()
    both_bounds = PriceRange.model_validate(
        {"startPrice": {"currencyCode": "TWD", "units": "1"}, "endPrice": {"currencyCode": "TWD", "units": "200"}}
    )
    start_only = PriceRange.model_validate({"startPrice": {"currencyCode": "TWD", "units": "200"}})
    end_only = PriceRange.model_validate({"endPrice": {"currencyCode": "TWD", "units": "200"}})

    assert module.price_outcome(None, "PRICE_LEVEL_INEXPENSIVE", both_bounds) == (
        "price_level_recovered",
        "price_range_filled",
    )
    assert module.price_outcome("PRICE_LEVEL_INEXPENSIVE", "PRICE_LEVEL_INEXPENSIVE", start_only) == ("price_range_filled",)
    assert module.price_outcome("PRICE_LEVEL_INEXPENSIVE", "PRICE_LEVEL_INEXPENSIVE", end_only) == ("price_range_partial",)
    assert module.price_outcome("PRICE_LEVEL_INEXPENSIVE", None, None) == ("price_level_lost", "price_range_absent")

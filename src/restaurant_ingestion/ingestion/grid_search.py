"""Taipei tiled discovery using Google Text Search IDs-only calls."""

from restaurant_ingestion.clients.google_places import GooglePlacesClient, PlacesClientError, ensure_request_budget
from restaurant_ingestion.ingestion.checkpoints import FAILED, RUNNING, SUCCEEDED, search_checkpoint
from restaurant_ingestion.ingestion.progress import print_crossed_request_progress
from restaurant_ingestion.storage.duckdb_store import DuckDbStore


def generate_tiles(
    south_bound: float,
    west_bound: float,
    north_bound: float,
    east_bound: float,
    tile_size_degrees: float,
) -> tuple[dict[str, object], ...]:
    tiles: list[dict[str, object]] = []
    index = 0
    south = south_bound
    while south < north_bound:
        north = min(south + tile_size_degrees, north_bound)
        west = west_bound
        while west < east_bound:
            east = min(west + tile_size_degrees, east_bound)
            tiles.append(
                {
                    "index": index,
                    "rectangle": {
                        "low": {"latitude": south, "longitude": west},
                        "high": {"latitude": north, "longitude": east},
                    },
                }
            )
            index += 1
            west = east
        south = north
    return tuple(tiles)


def plan_taipei_discovery(
    south_bound: float,
    west_bound: float,
    north_bound: float,
    east_bound: float,
    tile_size_degrees: float,
    discovery_queries: tuple[str, ...],
) -> dict[str, int]:
    tile_count = len(generate_tiles(south_bound, west_bound, north_bound, east_bound, tile_size_degrees))
    return {
        "tiles": tile_count,
        "queries": len(discovery_queries),
        "first_page_requests": tile_count * len(discovery_queries),
    }


def run_taipei_discovery(
    client: GooglePlacesClient,
    store: DuckDbStore,
    south_bound: float,
    west_bound: float,
    north_bound: float,
    east_bound: float,
    tile_size_degrees: float,
    discovery_queries: tuple[str, ...],
    ordinary_restaurant_target: int,
    discovery_request_budget: int,
    progress_log_interval: int,
) -> dict[str, int | bool]:
    requests_made = 0
    candidates_seen = 0
    used_requests = store.api_call_count("text_search_ids_only")
    tiles = generate_tiles(south_bound, west_bound, north_bound, east_bound, tile_size_degrees)

    for tile in tiles:
        tile_index = int(tile["index"])
        rectangle = checked_rectangle(tile["rectangle"])
        for query in discovery_queries:
            page_token: str | None = None
            page_index = 0
            while True:
                checkpoint_id, payload_json = search_checkpoint(query, tile_index, page_index, rectangle)
                if store.checkpoint_status(checkpoint_id) == SUCCEEDED:
                    break
                ensure_request_budget(used_requests, discovery_request_budget, 1)
                store.upsert_checkpoint(checkpoint_id, "search_page", RUNNING, payload_json, page_token, None)
                try:
                    response, attempts = client.search_text_ids_only(
                        query,
                        rectangle,
                        page_token,
                        discovery_request_budget - used_requests,
                    )
                    previous_used_requests = used_requests
                    log_search_attempts(store, checkpoint_id, attempts, previous_used_requests)
                    used_requests += len(attempts)
                    print_crossed_request_progress(
                        "discovery",
                        previous_used_requests,
                        used_requests,
                        discovery_request_budget,
                        progress_log_interval,
                    )
                    requests_made += len(attempts)
                    for place in response.places:
                        store.upsert_candidate(place.id, checkpoint_id)
                        candidates_seen += 1
                    store.upsert_checkpoint(
                        checkpoint_id,
                        "search_page",
                        SUCCEEDED,
                        payload_json,
                        response.nextPageToken,
                        None,
                    )
                    if store.enriched_restaurant_count() >= ordinary_restaurant_target:
                        return {
                            "requests_made": requests_made,
                            "candidates_seen": candidates_seen,
                            "stopped_at_target": True,
                        }
                    if response.nextPageToken is None:
                        break
                    page_token = response.nextPageToken
                    page_index += 1
                except PlacesClientError as exc:
                    previous_used_requests = used_requests
                    log_search_attempts(store, checkpoint_id, exc.attempts, previous_used_requests)
                    used_requests += len(exc.attempts)
                    print_crossed_request_progress(
                        "discovery",
                        previous_used_requests,
                        used_requests,
                        discovery_request_budget,
                        progress_log_interval,
                    )
                    requests_made += len(exc.attempts)
                    store.upsert_checkpoint(checkpoint_id, "search_page", FAILED, payload_json, page_token, str(exc))
                    break
                except Exception as exc:
                    store.upsert_checkpoint(checkpoint_id, "search_page", FAILED, payload_json, page_token, str(exc))
                    break
    return {
        "requests_made": requests_made,
        "candidates_seen": candidates_seen,
        "stopped_at_target": False,
    }


def checked_rectangle(value: object) -> dict[str, dict[str, float]]:
    if not isinstance(value, dict):
        raise TypeError("tile rectangle must be a dictionary")
    low = value.get("low")
    high = value.get("high")
    if not isinstance(low, dict) or not isinstance(high, dict):
        raise TypeError("tile rectangle must contain low and high dictionaries")
    return {"low": checked_point(low), "high": checked_point(high)}


def checked_point(value: dict[object, object]) -> dict[str, float]:
    latitude = value.get("latitude")
    longitude = value.get("longitude")
    if not isinstance(latitude, float) or not isinstance(longitude, float):
        raise TypeError("tile point latitude and longitude must be floats")
    return {"latitude": latitude, "longitude": longitude}


def log_search_attempts(
    store: DuckDbStore,
    checkpoint_id: str,
    attempts: tuple[dict[str, int], ...],
    used_requests_before_call: int,
) -> None:
    for index, attempt in enumerate(attempts, start=1):
        store.log_api_call(
            f"{checkpoint_id}-{used_requests_before_call + index}",
            "places:searchText",
            "text_search_ids_only",
            int(attempt["status_code"]),
            int(attempt["retry_count"]),
            None,
            checkpoint_id,
        )

"""Google Places API (New) client with explicit cost controls."""

import os
import time
from typing import Any

import httpx
from pydantic import ValidationError

from restaurant_ingestion.models import PlaceDetails, PlaceSearchResponse

PLACES_BASE_URL = "https://places.googleapis.com/v1"
TEXT_SEARCH_URL = f"{PLACES_BASE_URL}/places:searchText"
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
PLACE_ID_CANDIDATE_PAGE_SIZE = 5
PLACE_ID_MATCH_FIELD_MASK = (
    "places.id",
    "places.displayName",
    "places.formattedAddress",
    "places.location",
    "places.nationalPhoneNumber",
    "places.internationalPhoneNumber",
    "places.websiteUri",
    "places.primaryType",
    "places.types",
)
NO_ATMOSPHERE_FIELDS = {
    "allowsDogs",
    "curbsidePickup",
    "delivery",
    "dineIn",
    "editorialSummary",
    "generativeSummary",
    "neighborhoodSummary",
    "outdoorSeating",
    "paymentOptions",
    "reviews",
    "reviewSummary",
    "servesBeer",
    "servesBreakfast",
    "servesBrunch",
    "servesCoffee",
    "servesDinner",
    "servesLunch",
    "servesVegetarianFood",
    "servesWine",
    "takeout",
}


class PlacesClientError(RuntimeError):
    def __init__(self, message: str, attempts: tuple[dict[str, int], ...]) -> None:
        super().__init__(message)
        self.attempts = attempts


class BudgetExceededError(PlacesClientError):
    pass


class GooglePlacesClient:
    def __init__(
        self,
        api_key: str,
        api_key_env_var: str,
        language_code: str,
        region_code: str,
        request_timeout_seconds: float,
        max_retries: int,
        retry_initial_backoff_seconds: float,
        retry_max_backoff_seconds: float,
    ) -> None:
        if not api_key:
            raise PlacesClientError(f"{api_key_env_var} is required", ())
        self._api_key = api_key
        self._language_code = language_code
        self._region_code = region_code
        self._max_retries = max_retries
        self._retry_initial_backoff_seconds = retry_initial_backoff_seconds
        self._retry_max_backoff_seconds = retry_max_backoff_seconds
        self._client = httpx.Client(timeout=request_timeout_seconds)

    @classmethod
    def from_environment(
        cls,
        api_key_env_var: str,
        language_code: str,
        region_code: str,
        request_timeout_seconds: float,
        max_retries: int,
        retry_initial_backoff_seconds: float,
        retry_max_backoff_seconds: float,
    ) -> "GooglePlacesClient":
        api_key = os.environ.get(api_key_env_var, "")
        return cls(
            api_key,
            api_key_env_var,
            language_code,
            region_code,
            request_timeout_seconds,
            max_retries,
            retry_initial_backoff_seconds,
            retry_max_backoff_seconds,
        )

    def close(self) -> None:
        self._client.close()

    def search_text_ids_only(
        self,
        text_query: str,
        rectangle: dict[str, dict[str, float]],
        page_token: str | None,
        request_budget_remaining: int,
    ) -> tuple[PlaceSearchResponse, tuple[dict[str, int], ...]]:
        payload: dict[str, object] = {
            "textQuery": text_query,
            "pageSize": 20,
            "locationRestriction": {"rectangle": rectangle},
            "languageCode": self._language_code,
            "regionCode": self._region_code,
        }
        if page_token is not None:
            payload["pageToken"] = page_token

        response_json_value, attempts = self._request(
            "POST",
            TEXT_SEARCH_URL,
            {"X-Goog-FieldMask": "places.id,nextPageToken"},
            payload,
            request_budget_remaining,
        )
        try:
            return PlaceSearchResponse.model_validate(response_json_value), attempts
        except ValidationError as exc:
            raise PlacesClientError("Google Places returned invalid search data", attempts) from exc

    def search_text_place_candidates(
        self,
        text_query: str,
        request_budget_remaining: int,
    ) -> tuple[PlaceSearchResponse, tuple[dict[str, int], ...]]:
        payload: dict[str, object] = {
            "textQuery": text_query,
            "pageSize": PLACE_ID_CANDIDATE_PAGE_SIZE,
            "languageCode": self._language_code,
            "regionCode": self._region_code,
        }

        response_json_value, attempts = self._request(
            "POST",
            TEXT_SEARCH_URL,
            {"X-Goog-FieldMask": "places.id,places.formattedAddress,places.location"},
            payload,
            request_budget_remaining,
        )
        try:
            return PlaceSearchResponse.model_validate(response_json_value), attempts
        except ValidationError as exc:
            raise PlacesClientError("Google Places returned invalid search data", attempts) from exc

    def search_text_place_id_match_candidates(
        self,
        text_query: str,
        request_budget_remaining: int,
    ) -> tuple[dict[str, Any], tuple[dict[str, int], ...]]:
        payload: dict[str, object] = {
            "textQuery": text_query,
            "pageSize": PLACE_ID_CANDIDATE_PAGE_SIZE,
            "languageCode": self._language_code,
            "regionCode": self._region_code,
        }

        return self._request(
            "POST",
            TEXT_SEARCH_URL,
            {"X-Goog-FieldMask": ",".join(PLACE_ID_MATCH_FIELD_MASK)},
            payload,
            request_budget_remaining,
        )

    def place_details(
        self,
        place_id: str,
        field_mask: tuple[str, ...],
        request_budget_remaining: int,
    ) -> tuple[PlaceDetails, dict[str, Any], tuple[dict[str, int], ...]]:
        validate_no_atmosphere_fields(field_mask)
        response_json_value, attempts = self._request(
            "GET",
            f"{PLACES_BASE_URL}/places/{place_id}",
            {"X-Goog-FieldMask": ",".join(field_mask)},
            None,
            request_budget_remaining,
        )
        try:
            return PlaceDetails.model_validate(response_json_value), response_json_value, attempts
        except ValidationError as exc:
            raise PlacesClientError(f"Google Places returned invalid details for place_id={place_id}", attempts) from exc

    def _request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        json_body: dict[str, object] | None,
        request_budget_remaining: int,
    ) -> tuple[dict[str, Any], tuple[dict[str, int], ...]]:
        request_headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self._api_key,
            **headers,
        }
        retry_count = 0
        backoff = self._retry_initial_backoff_seconds
        attempts: list[dict[str, int]] = []

        while retry_count <= self._max_retries:
            try:
                ensure_request_budget(len(attempts), request_budget_remaining, 1)
            except BudgetExceededError as exc:
                raise BudgetExceededError(str(exc), tuple(attempts)) from exc
            try:
                response = self._client.request(method, url, headers=request_headers, json=json_body)
            except httpx.HTTPError as exc:
                attempts.append({"status_code": 0, "retry_count": retry_count})
                retry_count += 1
                if retry_count > self._max_retries:
                    raise PlacesClientError("Google Places request failed after transport retries", tuple(attempts)) from exc
                time.sleep(backoff)
                backoff = min(backoff * 2.0, self._retry_max_backoff_seconds)
                continue

            attempts.append({"status_code": response.status_code, "retry_count": retry_count})
            response_body = response_json(response, tuple(attempts))
            if response.status_code < 400:
                return response_body, tuple(attempts)
            if response.status_code not in RETRYABLE_STATUSES:
                raise PlacesClientError(f"Google Places request failed with status={response.status_code}", tuple(attempts))
            retry_count += 1
            if retry_count <= self._max_retries:
                time.sleep(backoff)
                backoff = min(backoff * 2.0, self._retry_max_backoff_seconds)

        raise PlacesClientError("Google Places request failed after configured retries", tuple(attempts))


def response_json(response: httpx.Response, attempts: tuple[dict[str, int], ...]) -> dict[str, Any]:
    try:
        value: Any = response.json()
    except ValueError as exc:
        raise PlacesClientError(f"Google Places returned non-JSON response with status={response.status_code}", attempts) from exc
    if not isinstance(value, dict):
        raise PlacesClientError(f"Google Places returned non-object JSON with status={response.status_code}", attempts)
    return value


def ensure_request_budget(used: int, limit: int, requested: int) -> None:
    if used + requested > limit:
        raise BudgetExceededError(f"Request budget exceeded: limit={limit}, used={used}, requested={requested}", ())


def validate_no_atmosphere_fields(field_mask: tuple[str, ...]) -> None:
    blocked_fields = sorted(set(field_mask).intersection(NO_ATMOSPHERE_FIELDS))
    if blocked_fields:
        raise ValueError(f"Atmosphere fields are disabled for cost control: {blocked_fields}")

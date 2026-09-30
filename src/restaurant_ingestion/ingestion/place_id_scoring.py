"""Offline candidate scoring for bounded Michelin place_id proof fixtures."""

from dataclasses import dataclass
from difflib import SequenceMatcher
import math
import re
from typing import Any
from urllib.parse import urlparse

from restaurant_ingestion.ingestion.michelin_place_ids import distance_between_coordinates_meters, structured_addresses_match

NEAR_DISTANCE_METERS = 75.0
NEARISH_DISTANCE_METERS = 350.0
FAR_WITHOUT_CORROBORATION_METERS = 500.0
STRONG_NAME_SIMILARITY = 0.92
PARTIAL_NAME_SIMILARITY = 0.72
MIN_STRONG_SIGNALS = 2
AUTO_SCORE_WITH_PROXIMITY = 7
AUTO_SCORE_WITHOUT_PROXIMITY = 8
AMBIGUOUS_SCORE_MARGIN = 2
GENERIC_NAME_MAX_COMPACT_LENGTH = 3
CORROBORATING_SIGNALS = ("phone", "website", "structured_address")
PLACEHOLDER_NAME_TOKENS = frozenset(("unnamed", "unknown", "tbd", "na"))
LODGING_TYPES = frozenset(
    (
        "lodging",
        "hotel",
        "resort_hotel",
        "extended_stay_hotel",
        "bed_and_breakfast",
        "guest_house",
        "hostel",
        "inn",
        "motel",
        "japanese_inn",
        "budget_japanese_inn",
    )
)
GENERIC_NAME_TOKENS = frozenset(
    (
        "restaurant",
        "cuisine",
        "kitchen",
        "bistro",
        "cafe",
        "eatery",
        "diner",
        "canteen",
        "food",
        "bar",
        "grill",
        "house",
        "chinese",
        "japanese",
        "taiwanese",
        "cantonese",
        "sichuan",
        "shanghainese",
        "korean",
        "thai",
        "italian",
        "french",
        "western",
    )
)


@dataclass(frozen=True)
class ScoredCandidate:
    place_id: str
    score: int
    strong_signal_count: int
    distance_meters: float
    signals: tuple[str, ...]


@dataclass(frozen=True)
class MatchDecision:
    outcome: str
    place_id: str
    reason: str
    candidates: tuple[ScoredCandidate, ...]


def decide_place_id_match(row: dict[str, object], fixture_row: dict[str, object]) -> MatchDecision:
    candidates = tuple(score_candidate(row, candidate) for candidate in eligible_candidates(fixture_row))
    ranked = tuple(sorted(candidates, key=lambda candidate: (-candidate.score, candidate.distance_meters, candidate.place_id)))
    if not ranked:
        return MatchDecision("manual_review", "", "no_candidates", ())

    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    if best.score < minimum_auto_score(best) or best.strong_signal_count < MIN_STRONG_SIGNALS:
        return MatchDecision("manual_review", "", f"insufficient_signals:{'+'.join(best.signals)}", ranked)
    if second is not None and second.score >= best.score - AMBIGUOUS_SCORE_MARGIN:
        return MatchDecision("manual_review", "", f"ambiguous_top_candidates:{best.place_id}:{second.place_id}", ranked)
    if best.distance_meters > FAR_WITHOUT_CORROBORATION_METERS and not has_signal(best, CORROBORATING_SIGNALS):
        return MatchDecision("manual_review", "", f"far_without_external_id:{'+'.join(best.signals)}", ranked)
    return MatchDecision("place_id", best.place_id, "+".join(best.signals), ranked)


def score_candidate(row: dict[str, object], candidate: dict[str, Any]) -> ScoredCandidate:
    source_name = text_value(row.get("name", ""))
    source_address = text_value(row.get("address", ""))
    source_phone = text_value(row.get("phone_number", ""))
    source_website = text_value(row.get("website_url", ""))
    candidate_name = candidate_display_name(candidate)
    candidate_address = text_value(candidate.get("formattedAddress", ""))
    candidate_website = text_value(candidate.get("websiteUri", ""))
    distance_meters = candidate_distance_meters(row, candidate)
    signals: list[str] = []
    score = 0
    strong_signal_count = 0

    if distance_meters <= NEAR_DISTANCE_METERS:
        score += 3
        strong_signal_count += 1
        signals.append("near")
    elif distance_meters <= NEARISH_DISTANCE_METERS:
        score += 1
        signals.append("nearish")

    if candidate_address and structured_addresses_match(source_address, candidate_address):
        score += 3
        strong_signal_count += 1
        signals.append("structured_address")

    name_score = name_similarity(source_name, candidate_name)
    if name_score >= STRONG_NAME_SIMILARITY:
        score += 4
        strong_signal_count += 1
        signals.append("name")
    elif name_score >= PARTIAL_NAME_SIMILARITY and not generic_source_name(source_name):
        score += 2
        signals.append("partial_name")
    elif generic_source_name(source_name) and source_name_tokens_contained(source_name, candidate_name):
        score += 2
        signals.append("generic_name_contained")

    if source_phone and phone_matches(source_phone, candidate):
        score += 5
        strong_signal_count += 1
        signals.append("phone")

    if source_website and website_matches(source_website, candidate_website):
        score += 5
        strong_signal_count += 1
        signals.append("website")

    if source_website and website_path_matches(source_website, candidate_website):
        score += 2
        signals.append("website_path")

    if candidate_is_restaurant(candidate):
        score += 1
        signals.append("restaurant_type")

    return ScoredCandidate(
        text_value(candidate.get("id", "")),
        score,
        strong_signal_count,
        distance_meters,
        tuple(signals),
    )


def eligible_candidates(fixture_row: dict[str, object]) -> tuple[dict[str, Any], ...]:
    return tuple(candidate for candidate in deduped_candidates(fixture_row) if not candidate_is_lodging(candidate))


def deduped_candidates(fixture_row: dict[str, object]) -> tuple[dict[str, Any], ...]:
    candidates_by_id: dict[str, dict[str, Any]] = {}
    queries = fixture_row.get("queries", ())
    if not isinstance(queries, list):
        raise ValueError("fixture queries must be a list")
    for query in queries:
        if not isinstance(query, dict):
            raise ValueError("fixture query must be an object")
        response = query.get("response", {})
        if not isinstance(response, dict):
            raise ValueError("fixture response must be an object")
        places = response.get("places", [])
        if not isinstance(places, list):
            raise ValueError("fixture places must be a list")
        for place in places:
            if not isinstance(place, dict):
                raise ValueError("fixture place must be an object")
            place_id = text_value(place.get("id", ""))
            if place_id and place_id not in candidates_by_id:
                candidates_by_id[place_id] = place
    return tuple(candidates_by_id.values())


def candidate_distance_meters(row: dict[str, object], candidate: dict[str, Any]) -> float:
    location = candidate.get("location")
    if not isinstance(location, dict):
        return math.inf
    return distance_between_coordinates_meters(
        float(row["latitude"]),
        float(row["longitude"]),
        float(location["latitude"]),
        float(location["longitude"]),
    )


def candidate_display_name(candidate: dict[str, Any]) -> str:
    display_name = candidate.get("displayName", {})
    if isinstance(display_name, dict):
        return text_value(display_name.get("text", ""))
    return ""


def name_similarity(source_name: str, candidate_name: str) -> float:
    left = normalized_name(source_name)
    right = normalized_name(candidate_name)
    if not left or not right:
        return 0.0
    if left.replace(" ", "") == right.replace(" ", ""):
        return 1.0
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    overlap = len(left_tokens.intersection(right_tokens)) / max(len(left_tokens), len(right_tokens))
    return max(overlap, SequenceMatcher(None, left, right).ratio())


def minimum_auto_score(candidate: ScoredCandidate) -> int:
    if has_signal(candidate, ("near", "structured_address")):
        return AUTO_SCORE_WITH_PROXIMITY
    return AUTO_SCORE_WITHOUT_PROXIMITY


def source_name_tokens_contained(source_name: str, candidate_name: str) -> bool:
    source_tokens = set(normalized_name(source_name).split())
    candidate_tokens = set(normalized_name(candidate_name).split())
    return bool(source_tokens) and source_tokens.issubset(candidate_tokens)


def normalized_name(value: str) -> str:
    normalized = value.casefold().replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def generic_source_name(value: str) -> bool:
    normalized = normalized_name(value)
    tokens = normalized.split()
    if not tokens:
        return True
    if len(normalized.replace(" ", "")) <= GENERIC_NAME_MAX_COMPACT_LENGTH:
        return True
    if any(token in PLACEHOLDER_NAME_TOKENS for token in tokens):
        return True
    return all(token in GENERIC_NAME_TOKENS for token in tokens)


def phone_matches(source_phone: str, candidate: dict[str, Any]) -> bool:
    source_digits = phone_digits(source_phone)
    if not source_digits:
        return False
    candidate_phones = (
        text_value(candidate.get("nationalPhoneNumber", "")),
        text_value(candidate.get("internationalPhoneNumber", "")),
    )
    for candidate_phone in candidate_phones:
        candidate_digits = phone_digits(candidate_phone)
        if candidate_digits and (source_digits.endswith(candidate_digits) or candidate_digits.endswith(source_digits[-9:])):
            return True
    return False


def phone_digits(value: str) -> str:
    return re.sub(r"\D+", "", value)


def website_matches(source_website: str, candidate_website: str) -> bool:
    source_host = normalized_host(source_website)
    candidate_host = normalized_host(candidate_website)
    return bool(source_host) and source_host == candidate_host


def website_path_matches(source_website: str, candidate_website: str) -> bool:
    if not website_matches(source_website, candidate_website):
        return False
    source_tokens = url_path_tokens(source_website)
    candidate_tokens = url_path_tokens(candidate_website)
    if not source_tokens or not candidate_tokens:
        return False
    return bool(source_tokens.intersection(candidate_tokens))


def normalized_host(value: str) -> str:
    parsed = urlparse(value)
    host = parsed.netloc.casefold()
    return host.removeprefix("www.")


def url_path_tokens(value: str) -> set[str]:
    parsed = urlparse(value)
    tokens = re.findall(r"[a-z0-9]+", parsed.path.casefold())
    return {token for token in tokens if len(token) >= 3}


def candidate_is_restaurant(candidate: dict[str, Any]) -> bool:
    primary_type = text_value(candidate.get("primaryType", ""))
    types_value = candidate.get("types", ())
    types = tuple(text_value(item) for item in types_value) if isinstance(types_value, list) else ()
    return primary_type == "restaurant" or "restaurant" in types


def candidate_is_lodging(candidate: dict[str, Any]) -> bool:
    primary_type = text_value(candidate.get("primaryType", ""))
    types_value = candidate.get("types", ())
    types = tuple(text_value(item) for item in types_value) if isinstance(types_value, list) else ()
    return primary_type in LODGING_TYPES or any(item in LODGING_TYPES for item in types)


def has_signal(candidate: ScoredCandidate, signals: tuple[str, ...]) -> bool:
    return bool(set(candidate.signals).intersection(signals))


def text_value(value: object) -> str:
    return str(value).strip()

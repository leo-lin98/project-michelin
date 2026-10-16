"""Minimal restaurant identity resolution."""

from dataclasses import dataclass
from difflib import SequenceMatcher
from math import asin, cos, radians, sin, sqrt

import pandas as pd


@dataclass(frozen=True)
class IdentityMatchConfig:
    name_similarity_threshold: float
    distance_threshold_meters: float


@dataclass(frozen=True)
class IdentityResolutionResult:
    ordinary_without_guide_duplicates: pd.DataFrame
    alias_table: pd.DataFrame


def resolve_guide_ordinary_duplicates(
    guide_rows: pd.DataFrame,
    ordinary_rows: pd.DataFrame,
    config: IdentityMatchConfig,
) -> IdentityResolutionResult:
    aliases: list[dict[str, object]] = []
    duplicate_indices: set[int] = set()

    for ordinary_index, ordinary_row in ordinary_rows.iterrows():
        match = find_best_guide_match(ordinary_row, guide_rows, config)
        if match is not None:
            duplicate_indices.add(int(ordinary_index))
            aliases.append(match)

    deduped = ordinary_rows.drop(index=sorted(duplicate_indices)).reset_index(drop=True)
    alias_table = pd.DataFrame(
        aliases,
        columns=[
            "ordinary_source_id",
            "ordinary_name",
            "guide_source_id",
            "guide_name",
            "name_similarity",
            "distance_meters",
            "match_reason",
        ],
    )
    return IdentityResolutionResult(
        ordinary_without_guide_duplicates=deduped,
        alias_table=alias_table,
    )


def find_best_guide_match(
    ordinary_row: pd.Series,
    guide_rows: pd.DataFrame,
    config: IdentityMatchConfig,
) -> dict[str, object] | None:
    best_match: dict[str, object] | None = None
    best_score = 0.0

    for _, guide_row in guide_rows.iterrows():
        name_similarity = normalized_similarity(str(ordinary_row.get("name", "")), str(guide_row.get("name", "")))
        distance_meters = geo_distance_meters(
            ordinary_row.get("latitude"),
            ordinary_row.get("longitude"),
            guide_row.get("latitude"),
            guide_row.get("longitude"),
        )
        address_match = normalized_text(str(ordinary_row.get("address", ""))) == normalized_text(str(guide_row.get("address", "")))
        geo_match = distance_meters <= config.distance_threshold_meters
        name_match = name_similarity >= config.name_similarity_threshold

        if name_match and (geo_match or address_match) and name_similarity > best_score:
            best_score = name_similarity
            best_match = {
                "ordinary_source_id": ordinary_row.get("source_id", ""),
                "ordinary_name": ordinary_row.get("name", ""),
                "guide_source_id": guide_row.get("source_id", ""),
                "guide_name": guide_row.get("name", ""),
                "name_similarity": name_similarity,
                "distance_meters": distance_meters,
                "match_reason": "name_and_geo" if geo_match else "name_and_address",
            }
    return best_match


def normalized_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalized_text(left), normalized_text(right)).ratio()


def normalized_text(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def geo_distance_meters(
    left_latitude: object,
    left_longitude: object,
    right_latitude: object,
    right_longitude: object,
) -> float:
    left_lat = numeric_or_none(left_latitude)
    left_lon = numeric_or_none(left_longitude)
    right_lat = numeric_or_none(right_latitude)
    right_lon = numeric_or_none(right_longitude)
    if left_lat is None or left_lon is None or right_lat is None or right_lon is None:
        return float("inf")

    earth_radius_meters = 6_371_000.0
    lat_delta = radians(right_lat - left_lat)
    lon_delta = radians(right_lon - left_lon)
    left_lat_radians = radians(left_lat)
    right_lat_radians = radians(right_lat)
    haversine = sin(lat_delta / 2.0) ** 2 + cos(left_lat_radians) * cos(right_lat_radians) * sin(lon_delta / 2.0) ** 2
    return 2.0 * earth_radius_meters * asin(sqrt(haversine))


def numeric_or_none(value: object) -> float | None:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric_value):
        return None
    return numeric_value

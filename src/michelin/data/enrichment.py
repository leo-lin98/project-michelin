"""Uniform enrichment for guide and ordinary restaurants."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pandas as pd

from michelin.data.sources import filter_taipei_rows, read_table

FEATURE_COLUMNS: tuple[str, ...] = (
    "name",
    "address",
    "city",
    "latitude",
    "longitude",
    "price_level",
    "cuisine",
    "average_rating",
    "review_count",
    "feature_provider",
    "feature_snapshot_date",
)


@dataclass(frozen=True)
class OrdinaryPools:
    in_sample: pd.DataFrame
    out_of_sample: pd.DataFrame


def load_ordinary_snapshot(path: Path, snapshot_date: str, provider: str) -> pd.DataFrame:
    raw = read_table(path)
    return normalize_ordinary_rows(raw, snapshot_date, provider)


def normalize_ordinary_rows(raw: pd.DataFrame, snapshot_date: str, provider: str) -> pd.DataFrame:
    rows = raw.rename(columns=build_ordinary_rename_map(raw.columns))
    required_columns = {"name"}
    missing_columns = required_columns.difference(rows.columns)
    if missing_columns:
        raise ValueError(f"Ordinary snapshot missing columns: {sorted(missing_columns)}")

    normalized = pd.DataFrame(
        {
            "source_id": text_column(rows, "source_id"),
            "name": text_column(rows, "name"),
            "address": text_column(rows, "address"),
            "city": text_column(rows, "city").replace("", "unknown"),
            "latitude": number_column(rows, "latitude"),
            "longitude": number_column(rows, "longitude"),
            "price_level": text_column(rows, "price_level").replace("", "unknown"),
            "cuisine": text_column(rows, "cuisine").replace("", "unknown"),
            "average_rating": number_column(rows, "average_rating").fillna(-1.0),
            "review_count": number_column(rows, "review_count").fillna(-1).astype(int),
            "source": provider,
            "source_snapshot_date": snapshot_date,
            "award": "none",
        }
    )
    return filter_taipei_rows(normalized).sort_values(["name", "address"]).reset_index(drop=True)


def enrich_with_uniform_provider(
    rows: pd.DataFrame, features: pd.DataFrame, provider: str,
) -> pd.DataFrame:
    """Join provider-owned fields by case-sensitive Place ID, discarding source features."""
    required = {"source_id", *FEATURE_COLUMNS}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Provider features missing columns: {sorted(missing)}")
    if features["source_id"].duplicated().any():
        raise ValueError("Duplicate provider source_id values")
    if not features["feature_provider"].eq(provider).all():
        raise ValueError(f"Feature provenance mismatch: expected {provider}")
    metadata = rows.drop(columns=list(FEATURE_COLUMNS), errors="ignore")
    enriched = metadata.merge(features.loc[:, ["source_id", *FEATURE_COLUMNS]], on="source_id", how="left", validate="one_to_one", indicator=True)
    if enriched["_merge"].ne("both").any():
        raise ValueError("Guide rows are missing provider features; resolve and enrich every Guide Place ID")
    return enriched.drop(columns="_merge")


def sampling_key(source_id: str, seed: int) -> str:
    """Rank IDs independently of input order, names, labels, and global RNG state."""
    return sha256(f"{seed}:{source_id}".encode("utf-8")).hexdigest()


def split_ordinary_pools(
    ordinary_rows: pd.DataFrame, in_sample_limit: int, out_of_sample_limit: int, seed: int,
) -> OrdinaryPools:
    required = in_sample_limit + out_of_sample_limit
    if len(ordinary_rows) < required:
        raise ValueError(f"Insufficient ordinary rows: {required} required, {len(ordinary_rows)} available")
    ranked = ordinary_rows.assign(sampling_key=ordinary_rows["source_id"].map(lambda value: sampling_key(value, seed)))
    sorted_rows = ranked.sort_values(["sampling_key", "source_id"]).drop(columns="sampling_key").reset_index(drop=True)
    return OrdinaryPools(
        in_sample=sorted_rows.iloc[:in_sample_limit].copy().reset_index(drop=True),
        out_of_sample=sorted_rows.iloc[in_sample_limit:required].copy().reset_index(drop=True),
    )


def build_ordinary_rename_map(columns: pd.Index) -> dict[str, str]:
    aliases: dict[str, tuple[str, ...]] = {
        "source_id": ("source_id", "id", "osm_id", "place_id"),
        "name": ("name", "Name"),
        "address": ("address", "Address", "addr:full"),
        "city": ("city", "City", "addr:city"),
        "latitude": ("latitude", "Latitude", "lat"),
        "longitude": ("longitude", "Longitude", "lon", "lng"),
        "price_level": ("price_level", "price", "Price"),
        "cuisine": ("cuisine", "Cuisine", "amenity", "category"),
        "average_rating": ("average_rating", "rating", "Rating"),
        "review_count": ("review_count", "reviews", "ReviewCount"),
    }
    rename_map: dict[str, str] = {}
    for canonical_name, possible_names in aliases.items():
        for possible_name in possible_names:
            if possible_name in columns:
                rename_map[possible_name] = canonical_name
                break
    return rename_map


def text_column(rows: pd.DataFrame, column: str) -> pd.Series:
    if column in rows.columns:
        return rows[column].fillna("").astype(str).str.strip()
    return pd.Series([""] * len(rows), index=rows.index)


def number_column(rows: pd.DataFrame, column: str) -> pd.Series:
    if column in rows.columns:
        return pd.to_numeric(rows[column], errors="coerce")
    return pd.Series([pd.NA] * len(rows), index=rows.index, dtype="Float64")


def empty_feature_value(column: str) -> object:
    if column in {"average_rating", "review_count"}:
        return -1
    if column in {"latitude", "longitude"}:
        return pd.NA
    return "unknown"

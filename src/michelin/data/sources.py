"""Path A guide source loaders."""

from pathlib import Path
from typing import Any

import kagglehub
import pandas as pd

from michelin.config import Award

TAIPEI = "Taipei"

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "source_id": ("source_id", "id", "ID", "RestaurantId", "restaurant_id"),
    "name": ("name", "Name", "restaurant_name", "Restaurant Name", "Restaurant"),
    "address": ("address", "Address", "street_address", "Street Address"),
    "city": ("city", "City", "region", "Region"),
    "location": ("location", "Location", "area", "Area"),
    "latitude": ("latitude", "Latitude", "lat", "Lat"),
    "longitude": ("longitude", "Longitude", "lon", "lng", "Long"),
    "price_level": ("price_level", "Price", "price", "price_band"),
    "cuisine": ("cuisine", "Cuisine", "category", "Category"),
    "award": ("award", "Award", "distinction", "Distinction"),
    "year": ("year", "Year", "award_year", "Award Year"),
    "average_rating": ("average_rating", "rating", "Rating", "stars"),
    "review_count": ("review_count", "reviews", "ReviewCount", "review_count_total"),
}


def download_kaggle_dataset(dataset_name: str, output_dir: Path) -> Path:
    dataset_path = Path(kagglehub.dataset_download(dataset_name))
    output_dir.mkdir(parents=True, exist_ok=True)
    return dataset_path


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported table format for {path}")


def find_first_supported_table(directory: Path) -> Path:
    candidates = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in {".csv", ".json", ".jsonl"}
    )
    if not candidates:
        raise FileNotFoundError(f"No CSV/JSON/JSONL snapshot found in {directory}")
    return candidates[0]


def load_guide_snapshot(path: Path, snapshot_date: str, source_name: str) -> pd.DataFrame:
    raw = read_table(path)
    return normalize_guide_rows(raw, snapshot_date, source_name)


def load_wikipedia_starred_counts(path: Path) -> pd.DataFrame:
    raw = read_table(path)
    required_columns = {"year", "city", "starred_count"}
    missing_columns = required_columns.difference(raw.columns)
    if missing_columns:
        raise ValueError(f"Starred-count table missing columns: {sorted(missing_columns)}")
    counts = raw.loc[:, sorted(required_columns)].copy()
    counts["city"] = counts["city"].astype(str)
    counts["year"] = counts["year"].astype(int)
    counts["starred_count"] = counts["starred_count"].astype(int)
    return counts.sort_values(["year", "city"]).reset_index(drop=True)


def normalize_guide_rows(raw: pd.DataFrame, snapshot_date: str, source_name: str) -> pd.DataFrame:
    rows = raw.rename(columns=build_column_rename_map(raw.columns))
    required_columns = {"name", "award"}
    missing_columns = required_columns.difference(rows.columns)
    if missing_columns:
        raise ValueError(f"Guide snapshot missing columns: {sorted(missing_columns)}")

    normalized = pd.DataFrame(
        {
            "source_id": series_or_empty(rows, "source_id"),
            "name": clean_text_series(series_or_empty(rows, "name")),
            "address": clean_text_series(series_or_empty(rows, "address")),
            "city": infer_city(rows),
            "latitude": numeric_series(rows, "latitude"),
            "longitude": numeric_series(rows, "longitude"),
            "price_level": clean_text_series(series_or_empty(rows, "price_level")).replace("", "unknown"),
            "cuisine": clean_text_series(series_or_empty(rows, "cuisine")).replace("", "unknown"),
            "award": clean_text_series(series_or_empty(rows, "award")),
            "year": numeric_series(rows, "year").astype("Int64"),
            "average_rating": numeric_series(rows, "average_rating").fillna(-1.0),
            "review_count": numeric_series(rows, "review_count").fillna(-1).astype(int),
            "source": source_name,
            "source_snapshot_date": snapshot_date,
        }
    )
    normalized["award"] = normalized["award"].map(validate_award)
    normalized = filter_taipei_rows(normalized)
    return normalized.sort_values(["name", "award"]).reset_index(drop=True)


def build_column_rename_map(columns: pd.Index) -> dict[str, str]:
    rename_map: dict[str, str] = {}
    for canonical_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in columns:
                rename_map[alias] = canonical_name
                break
    return rename_map


def series_or_empty(rows: pd.DataFrame, column: str) -> pd.Series:
    if column in rows.columns:
        return rows[column]
    return pd.Series([""] * len(rows), index=rows.index)


def numeric_series(rows: pd.DataFrame, column: str) -> pd.Series:
    if column in rows.columns:
        return pd.to_numeric(rows[column], errors="coerce")
    return pd.Series([pd.NA] * len(rows), index=rows.index, dtype="Float64")


def clean_text_series(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip()


def infer_city(rows: pd.DataFrame) -> pd.Series:
    if "city" in rows.columns:
        city = clean_text_series(rows["city"])
    else:
        city = pd.Series([""] * len(rows), index=rows.index)

    if "location" in rows.columns:
        location = clean_text_series(rows["location"])
        city = city.mask(city == "", location.map(city_from_location))
    return city.replace("", "unknown")


def city_from_location(location: str) -> str:
    city = location.split(",")[0].strip().casefold()
    return {"taipei": "Taipei", "new taipei": "New Taipei"}.get(city, "unknown")


def validate_award(value: Any) -> str:
    text = str(value).strip()
    try:
        return Award(text).value
    except ValueError as exc:
        raise ValueError(f"Unsupported Michelin award: {text}") from exc


def filter_taipei_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Retain Taipei and New Taipei without inferring geography from names or IDs."""
    city_mask = rows["city"].astype(str).str.casefold().isin({"taipei", "new taipei"})
    return rows.loc[city_mask].copy()

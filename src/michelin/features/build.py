"""Validate provider-owned inputs and construct the shared, raw feature matrix."""

import math

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from michelin.config import AppConfig


class SharedRestaurant(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True, allow_inf_nan=False)
    restaurant_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    city: str = Field(min_length=1)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    price_range: str | None
    average_rating: float | None = Field(ge=0, le=5)
    review_count: float | None = Field(ge=0)
    cuisine: str | None
    feature_provider: str
    feature_snapshot_date: str

    @field_validator("review_count")
    @classmethod
    def integral_counts(cls, value: float | None) -> float | None:
        if value is not None and not value.is_integer():
            raise ValueError("Review count must be an integer")
        return value

    @field_validator("restaurant_id", "source_id", "city", "price_range", "cuisine")
    @classmethod
    def nonblank_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Identity and categorical values cannot be blank")
        return value


def validate_panel(panel: pd.DataFrame, config: AppConfig) -> None:
    """Reject malformed identities, ranges, city and provenance in every population."""
    config.features.require_phase_2_schema()
    required = set(SharedRestaurant.model_fields)
    missing = required.difference(panel.columns)
    if missing or panel.empty:
        raise ValueError(f"Panel empty or missing required columns: {sorted(missing)}")
    if panel["restaurant_id"].duplicated().any() or panel["source_id"].duplicated().any():
        raise ValueError("Panel identities must be unique")
    for record in panel.loc[:, sorted(required)].to_dict(orient="records"):
        validated = SharedRestaurant.model_validate({key: None if pd.isna(value) else value for key, value in record.items()})
        if validated.restaurant_id != "google_places:" + validated.source_id:
            raise ValueError(f"Restaurant identity mismatch: {validated.restaurant_id}")
        if validated.city not in config.pipeline.project.geography:
            raise ValueError(f"Unsupported city for {validated.restaurant_id}: {validated.city}")
        if validated.feature_provider != config.pipeline.data_sources.enrichment_provider:
            raise ValueError(f"Feature provenance mismatch for {validated.restaurant_id}")
    if pd.to_datetime(panel["feature_snapshot_date"], format="ISO8601", errors="coerce").isna().any():
        raise ValueError("Invalid provider feature_snapshot_date")


def build_shared_features(panel: pd.DataFrame, density: pd.Series, config: AppConfig) -> pd.DataFrame:
    """Select only configured provider features; never impute or learn from labels."""
    if config.features.feature_schema.optional_text:
        raise ValueError("Optional review text features have no shared source and must be disabled")
    validate_panel(panel, config)
    if not density.index.equals(pd.Index(panel["restaurant_id"], name="restaurant_id")):
        raise ValueError("Density identities/order do not match the panel")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 and float(value).is_integer() for value in density):
        raise ValueError("Neighborhood density must contain finite nonnegative integer counts")
    columns = (*config.features.feature_schema.numeric, *config.features.feature_schema.categorical)
    features = panel.set_index("restaurant_id").assign(neighborhood_density=density)
    return features.loc[:, columns].copy()

"""Typed configuration loading for the Michelin Taiwan pipeline."""

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class Award(StrEnum):
    """Michelin Guide award values used to derive the binary label."""

    ONE_STAR = "1 Star"
    TWO_STARS = "2 Stars"
    THREE_STARS = "3 Stars"
    BIB_GOURMAND = "Bib Gourmand"
    SELECTED_RESTAURANTS = "Selected Restaurants"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(StrictModel):
    name: str
    geography: str
    scope: str


class LabelConfig(StrictModel):
    positive_class: str
    negative_class: str
    starred_awards: tuple[Award, ...]
    hard_negative_awards: tuple[Award, ...]

    @field_validator("starred_awards")
    @classmethod
    def validate_starred_awards(cls, value: tuple[Award, ...]) -> tuple[Award, ...]:
        expected = {Award.ONE_STAR, Award.TWO_STARS, Award.THREE_STARS}
        if set(value) != expected:
            raise ValueError("starred_awards must contain exactly the three Star awards")
        return value

    @field_validator("hard_negative_awards")
    @classmethod
    def validate_hard_negative_awards(cls, value: tuple[Award, ...]) -> tuple[Award, ...]:
        expected = {Award.BIB_GOURMAND, Award.SELECTED_RESTAURANTS}
        if set(value) != expected:
            raise ValueError("hard_negative_awards must contain Bib Gourmand and Selected Restaurants")
        return value


class GroupConfig(StrictModel):
    in_sample: str
    out_of_sample: str


class SplitConfig(StrictModel):
    strategy: str
    test_size: float = Field(gt=0.0, lt=1.0)
    cv_strategy: str
    cv_folds: int = Field(ge=2)


class ModelConfig(StrictModel):
    explanation: str
    prediction: str


class ThresholdConfig(StrictModel):
    calibration: str
    similar_to_starred_cutoff: float = Field(ge=0.0, le=1.0)


class DataSourceConfig(StrictModel):
    guide_primary: str
    guide_crosscheck: str
    ordinary_base: str
    enrichment_provider: str


class PoolConfig(StrictModel):
    ordinary_in_sample: str
    ordinary_out_of_sample: str


class OutputConfig(StrictModel):
    docs_data_dir: Path
    reports_dir: Path


class PipelineConfig(StrictModel):
    project: ProjectConfig
    seed: int = Field(ge=0)
    labels: LabelConfig
    groups: GroupConfig
    split: SplitConfig
    models: ModelConfig
    threshold: ThresholdConfig
    data_sources: DataSourceConfig
    pools: PoolConfig
    outputs: OutputConfig


class FeatureSchemaConfig(StrictModel):
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    optional_text: tuple[str, ...]


class TransformConfig(StrictModel):
    numeric_imputer: str
    numeric_scaler: str
    categorical_encoder: str
    text_vectorizer: str


class LeakageControlConfig(StrictModel):
    excluded_fields: tuple[str, ...]
    fit_transforms_on: str
    require_feature_provenance_parity: bool


class FeatureConfig(StrictModel):
    feature_schema: FeatureSchemaConfig
    transforms: TransformConfig
    leakage_controls: LeakageControlConfig


class AppConfig(StrictModel):
    pipeline: PipelineConfig
    features: FeatureConfig


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(f"Unable to read YAML config at {path}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"YAML config at {path} must contain a mapping at the top level")
    return raw


def load_config(pipeline_path: Path, features_path: Path) -> AppConfig:
    pipeline_data = load_yaml_mapping(pipeline_path)
    features_data = load_yaml_mapping(features_path)

    try:
        return AppConfig(
            pipeline=PipelineConfig.model_validate(pipeline_data),
            features=FeatureConfig.model_validate(features_data),
        )
    except ValidationError as exc:
        raise ValueError("Invalid Michelin pipeline configuration") from exc

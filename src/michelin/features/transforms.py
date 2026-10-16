"""Immutable train-fitted imputation, scaling and categorical encoding."""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from michelin.config import FeatureConfig


@dataclass(frozen=True)
class NumericTransform:
    name: str
    median: float
    mean: float
    scale: float


@dataclass(frozen=True)
class CategoryTransform:
    name: str
    categories: tuple[str | None, ...]


@dataclass(frozen=True)
class FittedTransforms:
    numeric: tuple[NumericTransform, ...]
    categorical: tuple[CategoryTransform, ...]
    feature_names: tuple[str, ...]


def numeric_values(rows: pd.DataFrame, name: str) -> np.ndarray:
    values = rows[name]
    if any(isinstance(value, (complex, np.complexfloating)) or (not pd.isna(value) and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)))) for value in values):
        raise ValueError(f"Feature {name} requires numeric values or nulls")
    array = values.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(array).any():
        raise ValueError(f"Feature {name} contains infinite values")
    observed = array[~np.isnan(array)]
    if name == "average_rating" and ((observed < 0).any() or (observed > 5).any()):
        raise ValueError("average_rating must be in [0, 5]")
    if name in {"review_count", "neighborhood_density"} and ((observed < 0).any() or (observed != np.floor(observed)).any()):
        raise ValueError(f"{name} must contain nonnegative integer counts")
    return array


def categorical_values(rows: pd.DataFrame, name: str) -> tuple[str | None, ...]:
    if any(not pd.isna(value) and (not isinstance(value, str) or not value.strip()) for value in rows[name]):
        raise ValueError(f"Feature {name} requires nonblank string categories or nulls")
    return tuple(None if pd.isna(value) else value for value in rows[name])


def fit_transforms(train: pd.DataFrame, config: FeatureConfig) -> FittedTransforms:
    """Learn solely from the caller's training rows, without changing input or state."""
    config.require_phase_2_schema()
    if config.feature_schema.optional_text:
        raise ValueError("Optional review text features have no shared source and must be disabled")
    if train.empty:
        raise ValueError("Cannot fit transforms on empty training rows")
    numerics: list[NumericTransform] = []
    categories: list[CategoryTransform] = []
    names: list[str] = []
    for name in config.feature_schema.numeric:
        values = numeric_values(train, name)
        observed = values[~np.isnan(values)]
        if not len(observed):
            raise ValueError(f"Feature {name} has no training median; collect observed training values")
        median = float(np.median(observed))
        imputed = np.where(np.isnan(values), median, values)
        mean, scale = float(np.mean(imputed)), float(np.std(imputed))
        if not all(math.isfinite(value) for value in (median, mean, scale)):
            raise ValueError(f"Feature {name} cannot be scaled to finite values")
        numerics.append(NumericTransform(name, median, mean, scale if scale > 0 else 1.0))
        names.append(name)
    for name in config.feature_schema.categorical:
        values = categorical_values(train, name)
        learned = (None, *sorted({value for value in values if value is not None}))
        categories.append(CategoryTransform(name, learned))
        names.extend(f"{name}__missing" if value is None else f"{name}__value:{value}" for value in learned)
    return FittedTransforms(tuple(numerics), tuple(categories), tuple(names))


def transform_features(rows: pd.DataFrame, fitted: FittedTransforms) -> pd.DataFrame:
    """Apply frozen training state; unknown categories activate no learned category."""
    if any(state.name == "price_level" for state in fitted.numeric):
        raise ValueError("Phase 2 requires categorical price_range instead of numeric price_level")
    columns: list[np.ndarray] = []
    for state in fitted.numeric:
        values = numeric_values(rows, state.name)
        columns.append((np.where(np.isnan(values), state.median, values) - state.mean) / state.scale)
    for state in fitted.categorical:
        values = categorical_values(rows, state.name)
        columns.extend(np.asarray([value == category for value in values], dtype=float) for category in state.categories)
    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        raise ValueError("Transformed feature matrix contains nonfinite values")
    return pd.DataFrame(matrix, index=rows.index.copy(), columns=fitted.feature_names)

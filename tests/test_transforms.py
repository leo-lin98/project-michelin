"""Independent Phase 2 checks of train-only, immutable feature transforms."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from michelin.config import FeatureConfig, load_config
from michelin.features.transforms import fit_transforms, transform_features


@pytest.fixture
def feature_config() -> FeatureConfig:
    return load_config(Path("config/pipeline.yaml"), Path("config/features.yaml")).features


@pytest.fixture
def training() -> pd.DataFrame:
    return pd.DataFrame({
        "price_range": ["200-400", "2000+", None, "200-400"],
        "price_level": [0.0, 2.0, 4.0, np.nan],
        "average_rating": [1.0, 3.0, 5.0, np.nan],
        "review_count": [10.0, 20.0, 30.0, np.nan],
        "neighborhood_density": [7.0, 7.0, 7.0, 7.0],
        "cuisine": ["noodle_shop", "restaurant", None, "restaurant"],
        "city": ["Taipei", "Taipei", "New Taipei", "Taipei"],
    }, index=["a", "b", "c", "d"])


def test_train_median_population_scaling_and_frozen_held_out_encoding(
    training: pd.DataFrame, feature_config: FeatureConfig,
) -> None:
    """Train-only learning: analytic medians/scales and categories survive held-out extremes."""
    before = training.copy(deep=True)
    fitted = fit_transforms(training, feature_config)
    transformed_train = transform_features(training, fitted)
    expected = np.array([-np.sqrt(2), 0, np.sqrt(2), 0])
    for name in ("average_rating", "review_count"):
        np.testing.assert_allclose(transformed_train[name], expected)
    np.testing.assert_array_equal(transformed_train["neighborhood_density"], np.zeros(4))
    held_out = pd.DataFrame({
        "price_range": [None, "400-600"], "average_rating": [np.nan, 5.0],
        "review_count": [np.nan, 1020.0], "neighborhood_density": [7.0, 9.0],
        "cuisine": [None, "unseen_type"], "city": ["Taipei", "unseen_city"],
    }, index=["missing", "unseen"])
    held_before = held_out.copy(deep=True)
    state_before = repr(fitted)
    result = transform_features(held_out, fitted)
    assert tuple(result.columns) == tuple(transformed_train.columns) == fitted.feature_names
    assert result.index.equals(held_out.index)
    np.testing.assert_allclose(result.loc["missing", list(feature_config.feature_schema.numeric)], 0)
    assert result.loc["unseen", "review_count"] == pytest.approx(1000 / np.sqrt(50))
    assert result.loc["unseen", "neighborhood_density"] == 2
    cuisine_columns = [name for name in result.columns if name.startswith("cuisine__")]
    city_columns = [name for name in result.columns if name.startswith("city__")]
    assert result.loc["missing", cuisine_columns].sum() == 1
    assert result.loc["unseen", cuisine_columns + city_columns].sum() == 0
    assert np.isfinite(result.to_numpy()).all()
    assert all(dtype.kind == "f" for dtype in result.dtypes)
    pd.testing.assert_frame_equal(training, before)
    pd.testing.assert_frame_equal(held_out, held_before)
    assert repr(fitted) == state_before
    pd.testing.assert_frame_equal(transform_features(training, fitted), transformed_train)
    with pytest.raises(FrozenInstanceError):
        setattr(fitted, "feature_names", ())
    with pytest.raises(FrozenInstanceError):
        setattr(fitted.numeric[0], "median", 100)


def test_missing_categories_have_explicit_encoding_without_training_missingness(
    training: pd.DataFrame, feature_config: FeatureConfig,
) -> None:
    train = training.assign(price_range="2000+", cuisine="restaurant", city="Taipei")
    fitted = fit_transforms(train, feature_config)
    rows = train.iloc[:1].assign(price_range=None, cuisine=None, city=None)
    output = transform_features(rows, fitted)
    for name in ("price_range", "cuisine", "city"):
        selected = output.loc[:, output.columns.str.startswith(name + "__")]
        assert selected.sum(axis=1).item() == 1
        assert output[name + "__missing"].item() == 1


@pytest.mark.parametrize("column", ["average_rating", "review_count", "neighborhood_density"])
def test_all_missing_numeric_training_fails_explicitly(
    training: pd.DataFrame, feature_config: FeatureConfig, column: str,
) -> None:
    with pytest.raises(ValueError, match=column):
        fit_transforms(training.assign(**{column: np.nan}), feature_config)


@pytest.mark.parametrize("column,value", [
    ("average_rating", 5.1),
    ("review_count", -1), ("review_count", 1.5), ("neighborhood_density", -1),
    ("average_rating", float("inf")), ("review_count", "20"),
    ("average_rating", True), ("review_count", complex(2, 1)),
])
def test_invalid_numeric_inputs_fail_at_fit_and_transform(
    training: pd.DataFrame, feature_config: FeatureConfig, column: str,
    value: float | str | bool | complex,
) -> None:
    invalid = training.assign(**{column: pd.Series([value] * len(training), index=training.index)})
    with pytest.raises(ValueError):
        fit_transforms(invalid, feature_config)
    fitted = fit_transforms(training, feature_config)
    with pytest.raises(ValueError):
        transform_features(invalid, fitted)


@pytest.mark.parametrize("column", ["price_range", "cuisine"])
@pytest.mark.parametrize("value", ["", "  ", 123])
def test_malformed_category_fails_explicitly(
    training: pd.DataFrame, feature_config: FeatureConfig, column: str, value: str | int,
) -> None:
    with pytest.raises(ValueError, match=column):
        fit_transforms(training.assign(**{column: value}), feature_config)
    fitted = fit_transforms(training, feature_config)
    with pytest.raises(ValueError, match=column):
        transform_features(training.assign(**{column: value}), fitted)


def test_range_bands_are_exact_categories_and_price_level_has_no_model_effect(
    training: pd.DataFrame, feature_config: FeatureConfig,
) -> None:
    fitted = fit_transforms(training, feature_config)
    encoded = transform_features(training, fitted)
    price_columns = [name for name in encoded.columns if name.startswith("price_range__")]
    assert price_columns == ["price_range__missing", "price_range__value:200-400", "price_range__value:2000+"]
    np.testing.assert_array_equal(encoded.loc[:, price_columns], [
        [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0],
    ])
    unseen = training.iloc[:1].assign(price_range="400-600")
    np.testing.assert_array_equal(transform_features(unseen, fitted).loc[:, price_columns], [[0, 0, 0]])
    changed = training.assign(price_level=9999)
    assert fit_transforms(changed, feature_config) == fitted
    pd.testing.assert_frame_equal(transform_features(changed, fitted), encoded)
    assert not any("price_level" in name for name in encoded.columns)


def test_all_null_training_prices_learn_only_missing_category(
    training: pd.DataFrame, feature_config: FeatureConfig,
) -> None:
    train = training.assign(price_range=None)
    fitted = fit_transforms(train, feature_config)
    output = transform_features(train, fitted)
    columns = [name for name in output if name.startswith("price_range__")]
    assert columns == ["price_range__missing"]
    np.testing.assert_array_equal(output[columns], [[1], [1], [1], [1]])
    unseen = train.iloc[:1].assign(price_range="2000+")
    np.testing.assert_array_equal(transform_features(unseen, fitted)[columns], [[0]])

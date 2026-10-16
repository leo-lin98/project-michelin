"""Requirement-derived checks for identity-aligned classification assembly."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from michelin.config import AppConfig, load_config
from michelin.model import dataset as dataset_module
from michelin.model.dataset import assemble_classification_dataset


@pytest.fixture
def config() -> AppConfig:
    return load_config(Path("config/pipeline.yaml"), Path("config/features.yaml"))


@pytest.fixture
def panel() -> pd.DataFrame:
    return pd.DataFrame({
        "restaurant_id": ["s3", "ordinary", "display-z", "bib", "s1", "selected", "display-a", "s2"],
        "award": ["3 Stars", "none", "none", "Bib Gourmand", "1 Star", "Selected Restaurants", "none", "2 Stars"],
        "class": pd.array([1, 0, None, 0, 1, 0, None, 1], dtype="Int64"),
        "group": ["in_sample", "in_sample", "out_of_sample", "in_sample", "in_sample", "in_sample", "out_of_sample", "in_sample"],
        "is_hard_negative": [False, False, False, True, False, True, False, False],
    })


@pytest.fixture
def features() -> pd.DataFrame:
    return pd.DataFrame({
        "average_rating": [3.0, 4.0, np.nan, 4.5, 4.1, 4.2, 2.5, 4.8],
        "review_count": [10, 20, None, 30, 40, 50, 60, 70],
        "neighborhood_density": [1, 2, 3, 4, 5, 6, 7, 8],
        "price_range": [None, "2000+", "200-400", "400-600", "2000+", "600-800", None, "2000+"],
        "cuisine": [None, "restaurant", "noodle_shop", "restaurant", "restaurant", "restaurant", "restaurant", "restaurant"],
        "city": ["Taipei", "Taipei", "New Taipei", "Taipei", "Taipei", "Taipei", "New Taipei", "Taipei"],
    }, index=pd.Index(["ordinary", "s3", "display-z", "bib", "s1", "selected", "display-a", "s2"], name="restaurant_id"))


def test_sorted_identity_join_preserves_all_award_labels_and_raw_values(
    panel: pd.DataFrame, features: pd.DataFrame, config: AppConfig,
) -> None:
    result = assemble_classification_dataset(panel, features.iloc[::-1], config)
    assert result.features.index.tolist() == ["bib", "ordinary", "s1", "s2", "s3", "selected"]
    assert result.labels.to_dict() == {"bib": 0, "ordinary": 0, "s1": 1, "s2": 1, "s3": 1, "selected": 0}
    assert result.hard_negative.to_dict() == {"bib": True, "ordinary": False, "s1": False, "s2": False, "s3": False, "selected": True}
    assert result.labels.index.equals(result.features.index)
    assert result.hard_negative.index.equals(result.features.index)
    assert result.labels.dtype == np.dtype("int64")
    assert result.hard_negative.dtype == np.dtype("bool")
    assert result.display_features.index.tolist() == ["display-a", "display-z"]
    assert not result.display_features.index.isin(result.labels.index).any()
    assert tuple(result.features.columns) == ("average_rating", "review_count", "neighborhood_density", "price_range", "cuisine", "city")
    pd.testing.assert_frame_equal(result.features, features.loc[["bib", "ordinary", "s1", "s2", "s3", "selected"]])
    pd.testing.assert_frame_equal(result.display_features, features.loc[["display-a", "display-z"]])


def test_assembly_preserves_nulls_without_fitting_transforms(
    panel: pd.DataFrame, features: pd.DataFrame, config: AppConfig,
) -> None:
    raw = features.assign(average_rating=np.nan, review_count=pd.NA, neighborhood_density=np.nan, price_range=None)
    result = assemble_classification_dataset(panel, raw, config)
    assert result.features[["average_rating", "review_count", "neighborhood_density", "price_range"]].isna().all().all()
    assert result.display_features[["average_rating", "review_count", "neighborhood_density", "price_range"]].isna().all().all()
    assert pd.isna(result.features.loc["ordinary", "cuisine"])


def test_assembly_and_returned_objects_do_not_mutate_inputs(
    panel: pd.DataFrame, features: pd.DataFrame, config: AppConfig,
) -> None:
    panel_before, features_before = panel.copy(deep=True), features.copy(deep=True)
    result = assemble_classification_dataset(panel, features, config)
    pd.testing.assert_frame_equal(panel, panel_before)
    pd.testing.assert_frame_equal(features, features_before)
    result.features.loc["bib", "average_rating"] = 0.0
    result.display_features.loc["display-a", "city"] = "changed"
    result.labels.loc["s1"] = 0
    result.hard_negative.loc["bib"] = False
    pd.testing.assert_frame_equal(panel, panel_before)
    pd.testing.assert_frame_equal(features, features_before)


def test_single_class_assembly_defers_split_feasibility(
    panel: pd.DataFrame, features: pd.DataFrame, config: AppConfig,
) -> None:
    subset = panel.loc[panel["award"].eq("none")]
    result = assemble_classification_dataset(subset, features.loc[subset["restaurant_id"]], config)
    assert result.labels.to_dict() == {"ordinary": 0}
    assert result.display_features.index.tolist() == ["display-a", "display-z"]


@pytest.mark.parametrize("scenario", ["duplicate_panel", "duplicate_features", "missing_features", "extra_features", "blank_panel", "blank_features", "null_panel", "null_features", "numeric_panel", "numeric_features"])
def test_invalid_identity_join_fails_explicitly(
    panel: pd.DataFrame, features: pd.DataFrame, config: AppConfig, scenario: str,
) -> None:
    if scenario == "duplicate_panel":
        panel = pd.concat([panel, panel.iloc[:1]], ignore_index=True)
    elif scenario == "duplicate_features":
        features = pd.concat([features, features.iloc[:1]])
    elif scenario == "missing_features":
        features = features.drop(index="display-z")
    elif scenario == "extra_features":
        features = pd.concat([features, features.iloc[:1].rename(index={"ordinary": "extra"})])
    elif scenario.endswith("panel"):
        value = {"blank_panel": " ", "null_panel": None, "numeric_panel": 123}[scenario]
        panel = panel.assign(restaurant_id=panel["restaurant_id"].astype(object))
        panel.loc[0, "restaurant_id"] = value
    else:
        value = {"blank_features": " ", "null_features": None, "numeric_features": 123}[scenario]
        features = features.rename(index={"ordinary": value})
    with pytest.raises(ValueError, match="identit"):
        assemble_classification_dataset(panel, features, config)


@pytest.mark.parametrize("restaurant_id,column,value", [
    ("s1", "group", "unsupported"), ("s1", "group", None),
    ("s1", "award", "Green Star"), ("s1", "award", None),
    ("s1", "class", 0), ("bib", "class", 1), ("ordinary", "class", 1),
    ("s1", "class", None), ("ordinary", "class", 2),
    ("display-a", "class", 0), ("display-a", "award", "1 Star"),
    ("bib", "is_hard_negative", False), ("ordinary", "is_hard_negative", True),
    ("display-a", "is_hard_negative", True), ("s1", "is_hard_negative", None),
])
def test_invalid_population_metadata_fails_explicitly(
    panel: pd.DataFrame, features: pd.DataFrame, config: AppConfig,
    restaurant_id: str, column: str, value: str | int | bool | None,
) -> None:
    invalid = panel.astype({column: object})
    invalid.loc[invalid["restaurant_id"].eq(restaurant_id), column] = value
    with pytest.raises(ValueError):
        assemble_classification_dataset(invalid, features, config)


@pytest.mark.parametrize("column", ["restaurant_id", "class", "group", "award", "is_hard_negative"])
def test_missing_panel_metadata_fails_explicitly(
    panel: pd.DataFrame, features: pd.DataFrame, config: AppConfig, column: str,
) -> None:
    with pytest.raises(ValueError, match=column):
        assemble_classification_dataset(panel.drop(columns=column), features, config)


@pytest.mark.parametrize("column", ["average_rating", "review_count", "neighborhood_density", "price_range", "cuisine", "city"])
def test_missing_feature_columns_fail_explicitly(
    panel: pd.DataFrame, features: pd.DataFrame, config: AppConfig, column: str,
) -> None:
    with pytest.raises(ValueError, match=column):
        assemble_classification_dataset(panel, features.drop(columns=column), config)


@pytest.mark.parametrize("column", ["class", "is_hard_negative", "award", "price_level"])
def test_unexpected_feature_columns_fail_explicitly(
    panel: pd.DataFrame, features: pd.DataFrame, config: AppConfig, column: str,
) -> None:
    with pytest.raises(ValueError, match=column):
        assemble_classification_dataset(panel, features.assign(**{column: 1}), config)


def test_feature_columns_are_returned_in_configured_order(
    panel: pd.DataFrame, features: pd.DataFrame, config: AppConfig,
) -> None:
    result = assemble_classification_dataset(panel, features.iloc[:, ::-1], config)
    assert tuple(result.features.columns) == ("average_rating", "review_count", "neighborhood_density", "price_range", "cuisine", "city")


def test_duplicate_feature_columns_fail_explicitly(
    panel: pd.DataFrame, features: pd.DataFrame, config: AppConfig,
) -> None:
    invalid = pd.concat([features, features[["average_rating"]]], axis=1)
    with pytest.raises(ValueError, match="column"):
        assemble_classification_dataset(panel, invalid, config)


@pytest.mark.parametrize("column,value", [
    ("average_rating", float("inf")), ("review_count", float("-inf")),
    ("neighborhood_density", float("inf")), ("average_rating", 5.1),
    ("review_count", -1), ("review_count", 1.5), ("neighborhood_density", -1),
    ("review_count", "10"), ("average_rating", True),
])
@pytest.mark.parametrize("restaurant_id", ["s1", "display-z"])
def test_invalid_numeric_features_fail_in_every_population(
    panel: pd.DataFrame, features: pd.DataFrame, config: AppConfig,
    column: str, value: float | int | str | bool, restaurant_id: str,
) -> None:
    invalid = features.astype({column: object})
    invalid.loc[restaurant_id, column] = value
    with pytest.raises(ValueError, match=column):
        assemble_classification_dataset(panel, invalid, config)


@pytest.mark.parametrize("column,value", [("price_range", " "), ("cuisine", 123), ("city", "")])
def test_invalid_categorical_features_fail_explicitly(
    panel: pd.DataFrame, features: pd.DataFrame, config: AppConfig,
    column: str, value: str | int,
) -> None:
    invalid = features.astype({column: object})
    invalid.loc["display-z", column] = value
    with pytest.raises(ValueError, match=column):
        assemble_classification_dataset(panel, invalid, config)


@pytest.fixture
def split_dataset(config: AppConfig) -> dataset_module.ClassificationDataset:
    identities = [f"negative-{number:03d}" for number in range(80)] + [f"starred-{number:03d}" for number in range(20)] + ["display"]
    panel = pd.DataFrame({
        "restaurant_id": identities,
        "award": ["none"] * 80 + ["1 Star"] * 20 + ["none"],
        "class": pd.array([0] * 80 + [1] * 20 + [None], dtype="Int64"),
        "group": ["in_sample"] * 100 + ["out_of_sample"],
        "is_hard_negative": [False] * 101,
    })
    features = pd.DataFrame({
        "average_rating": [4.0] * 101, "review_count": [None] * 101,
        "neighborhood_density": [2] * 101, "price_range": [None] * 101,
        "cuisine": ["restaurant"] * 101, "city": ["Taipei"] * 101,
    }, index=pd.Index(identities, name="restaurant_id"))
    return assemble_classification_dataset(panel, features, config)


def test_holdout_is_deterministic_complete_disjoint_and_stratified(
    split_dataset: dataset_module.ClassificationDataset, config: AppConfig,
) -> None:
    before = split_dataset.features.copy(deep=True)
    labels_before = split_dataset.labels.copy(deep=True)
    rng_before = np.random.get_state()
    result = dataset_module.stratified_holdout(split_dataset, config.pipeline)
    assert result == dataset_module.stratified_holdout(split_dataset, config.pipeline)
    assert isinstance(result.train_ids, tuple)
    assert isinstance(result.test_ids, tuple)
    assert len(result.train_ids) == 80
    assert len(result.test_ids) == 20
    assert set(result.train_ids).isdisjoint(result.test_ids)
    assert set(result.train_ids) | set(result.test_ids) == set(split_dataset.labels.index)
    assert split_dataset.labels.loc[list(result.train_ids)].value_counts().to_dict() == {0: 64, 1: 16}
    assert split_dataset.labels.loc[list(result.test_ids)].value_counts().to_dict() == {0: 16, 1: 4}
    assert "display" not in result.train_ids + result.test_ids
    pd.testing.assert_frame_equal(split_dataset.features, before)
    pd.testing.assert_series_equal(split_dataset.labels, labels_before)
    rng_after = np.random.get_state()
    assert rng_after[0] == rng_before[0]
    np.testing.assert_array_equal(rng_after[1], rng_before[1])
    assert rng_after[2:] == rng_before[2:]


def test_hard_negative_reporting_slice_intersects_holdout_without_changing_model_inputs(
    split_dataset: dataset_module.ClassificationDataset, config: AppConfig,
) -> None:
    panel = pd.DataFrame({
        "restaurant_id": [*split_dataset.labels.index, "display"],
        "award": ["Bib Gourmand"] * 20 + ["Selected Restaurants"] * 20 + ["none"] * 40 + ["1 Star"] * 20 + ["none"],
        "class": pd.array([0] * 80 + [1] * 20 + [None], dtype="Int64"),
        "group": ["in_sample"] * 100 + ["out_of_sample"],
        "is_hard_negative": [True] * 40 + [False] * 61,
    })
    raw = pd.concat([split_dataset.features, split_dataset.display_features])
    result = assemble_classification_dataset(panel, raw, config)
    holdout = dataset_module.stratified_holdout(result, config.pipeline)
    features_before, labels_before = result.features.copy(deep=True), result.labels.copy(deep=True)
    hard_negative_before = result.hard_negative.copy(deep=True)
    expected_ids = {f"negative-{number:03d}" for number in range(40)}
    assert set(result.hard_negative.index[result.hard_negative]) == expected_ids
    held_out_mask = result.hard_negative.loc[list(holdout.test_ids)]
    slice_ids = held_out_mask.index[held_out_mask]
    assert set(slice_ids) == expected_ids.intersection(holdout.test_ids)
    assert set(panel.set_index("restaurant_id").loc[slice_ids, "award"]) == {"Bib Gourmand", "Selected Restaurants"}
    assert result.labels.loc[slice_ids].eq(0).all()
    assert set(slice_ids).isdisjoint(holdout.train_ids)
    assert set(slice_ids).isdisjoint(result.display_features.index)
    pd.testing.assert_frame_equal(result.features, features_before)
    pd.testing.assert_series_equal(result.labels, labels_before)
    pd.testing.assert_series_equal(result.hard_negative, hard_negative_before)
    assert holdout == dataset_module.stratified_holdout(result, config.pipeline)


def test_splits_are_independent_of_input_order(
    split_dataset: dataset_module.ClassificationDataset, config: AppConfig,
) -> None:
    shuffled = replace(
        split_dataset, features=split_dataset.features.iloc[::-1],
        labels=split_dataset.labels.sample(frac=1, random_state=31),
    )
    first = dataset_module.stratified_holdout(split_dataset, config.pipeline)
    assert first == dataset_module.stratified_holdout(shuffled, config.pipeline)
    assert dataset_module.stratified_folds(split_dataset, first.train_ids, config.pipeline) == dataset_module.stratified_folds(shuffled, first.train_ids[::-1], config.pipeline)


@pytest.mark.parametrize("use_holdout_training", [True, False])
def test_folds_cover_only_chosen_universe_once_with_rounded_class_proportions(
    split_dataset: dataset_module.ClassificationDataset, config: AppConfig,
    use_holdout_training: bool,
) -> None:
    holdout = dataset_module.stratified_holdout(split_dataset, config.pipeline)
    universe = holdout.train_ids if use_holdout_training else tuple(split_dataset.labels.index)
    before = split_dataset.labels.copy(deep=True)
    rng_before = np.random.get_state()
    folds = dataset_module.stratified_folds(split_dataset, universe, config.pipeline)
    assert isinstance(folds, tuple)
    assert len(folds) == 5
    assert folds == dataset_module.stratified_folds(split_dataset, universe, config.pipeline)
    validation_ids = tuple(identity for fold in folds for identity in fold.test_ids)
    assert len(validation_ids) == len(set(validation_ids)) == len(universe)
    assert set(validation_ids) == set(universe)
    for fold in folds:
        assert set(fold.train_ids).isdisjoint(fold.test_ids)
        assert set(fold.train_ids) | set(fold.test_ids) == set(universe)
        assert "display" not in fold.train_ids + fold.test_ids
        if use_holdout_training:
            assert set(fold.train_ids + fold.test_ids).isdisjoint(holdout.test_ids)
        for identities in (fold.train_ids, fold.test_ids):
            positives = int(split_dataset.labels.loc[list(identities)].sum())
            assert abs(positives - len(identities) * 0.2) < 1
    pd.testing.assert_series_equal(split_dataset.labels, before)
    rng_after = np.random.get_state()
    assert rng_after[0] == rng_before[0]
    np.testing.assert_array_equal(rng_after[1], rng_before[1])
    assert rng_after[2:] == rng_before[2:]


def test_split_sizes_fold_count_and_seed_are_read_from_config(
    split_dataset: dataset_module.ClassificationDataset, config: AppConfig,
) -> None:
    changed = config.pipeline.model_copy(update={
        "seed": 42,
        "split": config.pipeline.split.model_copy(update={"test_size": 0.3, "cv_folds": 4}),
    })
    holdout = dataset_module.stratified_holdout(split_dataset, changed)
    assert len(holdout.test_ids) == 30
    assert int(split_dataset.labels.loc[list(holdout.test_ids)].sum()) == 6
    assert set(holdout.test_ids) != set(dataset_module.stratified_holdout(split_dataset, config.pipeline).test_ids)
    ids = tuple(split_dataset.labels.index)
    folds = dataset_module.stratified_folds(split_dataset, ids, changed)
    assert len(folds) == 4
    assert {len(fold.test_ids) for fold in folds} == {25}
    seed_only = config.pipeline.model_copy(update={"seed": 42})
    assert dataset_module.stratified_folds(split_dataset, ids, seed_only) != dataset_module.stratified_folds(split_dataset, ids, config.pipeline)


@pytest.mark.parametrize("identities,error", [
    (("unknown",), "unknown"), (("display",), "display"),
    (("negative-000", "negative-000"), "unique"), ((), "empty"),
])
def test_folds_reject_invalid_selected_identities(
    split_dataset: dataset_module.ClassificationDataset, config: AppConfig,
    identities: tuple[str, ...], error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        dataset_module.stratified_folds(split_dataset, identities, config.pipeline)


@pytest.mark.parametrize("positive_count", [0, 1, 2])
def test_holdout_rejects_class_counts_that_cannot_populate_both_partitions(
    split_dataset: dataset_module.ClassificationDataset, config: AppConfig, positive_count: int,
) -> None:
    identities = [f"negative-{number:03d}" for number in range(80)] + [f"starred-{number:03d}" for number in range(positive_count)]
    selected = replace(split_dataset, features=split_dataset.features.loc[identities], labels=split_dataset.labels.loc[identities])
    with pytest.raises(ValueError, match="class"):
        dataset_module.stratified_holdout(selected, config.pipeline)


def test_holdout_rejects_partition_too_small_for_both_classes(
    split_dataset: dataset_module.ClassificationDataset, config: AppConfig,
) -> None:
    changed = config.pipeline.model_copy(update={"split": config.pipeline.split.model_copy(update={"test_size": 0.01})})
    with pytest.raises(ValueError, match="class"):
        dataset_module.stratified_holdout(split_dataset, changed)


@pytest.mark.parametrize("positive_count", [0, 1, 4])
def test_folds_validate_class_counts_in_selected_subset(
    split_dataset: dataset_module.ClassificationDataset, config: AppConfig, positive_count: int,
) -> None:
    identities = tuple(f"negative-{number:03d}" for number in range(80)) + tuple(f"starred-{number:03d}" for number in range(positive_count))
    with pytest.raises(ValueError, match="class"):
        dataset_module.stratified_folds(split_dataset, identities, config.pipeline)


@pytest.mark.parametrize("field", ["strategy", "cv_strategy"])
def test_unsupported_split_strategies_fail_explicitly(
    split_dataset: dataset_module.ClassificationDataset, config: AppConfig, field: str,
) -> None:
    changed = config.pipeline.model_copy(update={"split": config.pipeline.split.model_copy(update={field: "unsupported"})})
    with pytest.raises(ValueError, match="unsupported"):
        if field == "strategy":
            dataset_module.stratified_holdout(split_dataset, changed)
        else:
            dataset_module.stratified_folds(split_dataset, tuple(split_dataset.labels.index), changed)


@pytest.mark.parametrize("invalid_boundary", ["duplicate_features", "duplicate_labels", "mismatched_labels", "nonbinary_labels", "null_labels", "display_overlap"])
@pytest.mark.parametrize("operation", ["holdout", "folds"])
def test_split_boundary_rejects_invalid_directly_constructed_dataset(
    split_dataset: dataset_module.ClassificationDataset, config: AppConfig,
    invalid_boundary: str, operation: str,
) -> None:
    if invalid_boundary == "duplicate_features":
        invalid = replace(split_dataset, features=pd.concat([split_dataset.features, split_dataset.features.iloc[:1]]))
    elif invalid_boundary == "duplicate_labels":
        invalid = replace(split_dataset, labels=pd.concat([split_dataset.labels, split_dataset.labels.iloc[:1]]))
    elif invalid_boundary == "mismatched_labels":
        invalid = replace(split_dataset, labels=split_dataset.labels.rename(index={"negative-000": "other"}))
    elif invalid_boundary == "nonbinary_labels":
        invalid = replace(split_dataset, labels=split_dataset.labels.replace({1: 2}))
    elif invalid_boundary == "null_labels":
        invalid = replace(split_dataset, labels=split_dataset.labels.replace({1: np.nan}))
    else:
        invalid = replace(split_dataset, display_features=split_dataset.features.iloc[:1])
    with pytest.raises(ValueError):
        if operation == "holdout":
            dataset_module.stratified_holdout(invalid, config.pipeline)
        else:
            dataset_module.stratified_folds(invalid, tuple(split_dataset.labels.index), config.pipeline)

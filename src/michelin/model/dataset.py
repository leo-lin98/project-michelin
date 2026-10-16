"""Validated raw classification features with separate labels and reporting slices."""

from dataclasses import dataclass
from math import ceil

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from michelin.config import AppConfig, PipelineConfig
from michelin.features.transforms import categorical_values, numeric_values


@dataclass(frozen=True)
class ClassificationDataset:
    features: pd.DataFrame
    labels: pd.Series
    hard_negative: pd.Series
    display_features: pd.DataFrame


@dataclass(frozen=True)
class DatasetSplit:
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]


def validate_dataset_identities(panel: pd.DataFrame, features: pd.DataFrame) -> None:
    """Require a one-to-one identity join across modeling and display populations."""
    if "restaurant_id" not in panel.columns:
        raise ValueError("Panel identities require restaurant_id")
    for name, identities in (("Panel", panel["restaurant_id"]), ("Feature", features.index)):
        if any(not isinstance(value, str) or not value.strip() for value in identities):
            raise ValueError(f"{name} identities must be nonblank strings")
        if identities.duplicated().any():
            raise ValueError(f"{name} identities must be unique")
    panel_ids, feature_ids = set(panel["restaurant_id"]), set(features.index)
    if panel_ids != feature_ids:
        raise ValueError(
            f"Feature identities do not match panel: missing={sorted(panel_ids - feature_ids)}, "
            f"extra={sorted(feature_ids - panel_ids)}"
        )


def validate_dataset_metadata(panel: pd.DataFrame, config: AppConfig) -> None:
    """Reject unsupported populations and labels that contradict award metadata."""
    required = {"class", "group", "award", "is_hard_negative"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Panel population metadata missing columns: {sorted(missing)}")
    groups = config.pipeline.groups
    if not panel["group"].isin([groups.in_sample, groups.out_of_sample]).all():
        raise ValueError("Panel contains unsupported or missing group metadata")
    starred = panel["award"].isin(config.pipeline.labels.starred_awards)
    hard_negative = panel["award"].isin(config.pipeline.labels.hard_negative_awards)
    if not (starred | hard_negative | panel["award"].eq("none")).fillna(False).all():
        raise ValueError("Panel contains unsupported or missing award metadata")
    modeling = panel["group"].eq(groups.in_sample)
    display = panel["group"].eq(groups.out_of_sample)
    if panel.loc[modeling, "class"].isna().any() or not panel.loc[modeling, "class"].isin([0, 1]).all():
        raise ValueError("Modeling class labels must be non-null binary values")
    if not panel.loc[modeling, "class"].eq(starred.loc[modeling].astype(int)).all():
        raise ValueError("Modeling class labels disagree with award metadata")
    if panel.loc[display, "class"].notna().any():
        raise ValueError("Display rows must have null class labels")
    if not panel.loc[display, "award"].eq("none").all():
        raise ValueError("Display rows must have ordinary award metadata")
    if panel["is_hard_negative"].isna().any() or not panel["is_hard_negative"].eq(hard_negative).all():
        raise ValueError("Hard-negative metadata disagrees with award metadata")


def validate_dataset_features(features: pd.DataFrame, config: AppConfig) -> None:
    """Validate raw values without requiring provider metadata or fitting transforms."""
    config.features.require_phase_2_schema()
    if config.features.feature_schema.optional_text:
        raise ValueError("Optional review text features have no shared source and must be disabled")
    schema = config.features.feature_schema
    expected = {*schema.numeric, *schema.categorical}
    if features.columns.duplicated().any():
        raise ValueError("Feature columns must be unique")
    missing, unexpected = expected.difference(features.columns), set(features.columns).difference(expected)
    if missing or unexpected:
        raise ValueError(f"Invalid feature columns: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for name in schema.numeric:
        numeric_values(features, name)
    for name in schema.categorical:
        categorical_values(features, name)


def assemble_classification_dataset(
    panel: pd.DataFrame, features: pd.DataFrame, config: AppConfig,
) -> ClassificationDataset:
    """Join by restaurant identity and return independent, sorted raw population frames."""
    validate_dataset_identities(panel, features)
    validate_dataset_metadata(panel, config)
    validate_dataset_features(features, config)
    metadata = panel.set_index("restaurant_id").sort_index()
    modeling = metadata.loc[metadata["group"].eq(config.pipeline.groups.in_sample)]
    display_ids = metadata.index[metadata["group"].eq(config.pipeline.groups.out_of_sample)]
    columns = (*config.features.feature_schema.numeric, *config.features.feature_schema.categorical)
    return ClassificationDataset(
        features=features.loc[modeling.index, columns].copy(deep=True),
        labels=modeling["class"].astype("int64").copy(deep=True),
        hard_negative=modeling["is_hard_negative"].astype(bool).copy(deep=True),
        display_features=features.loc[display_ids, columns].copy(deep=True),
    )


def split_labels(dataset: ClassificationDataset, restaurant_ids: tuple[str, ...]) -> pd.Series:
    """Validate modeling identities and align selected binary labels in canonical order."""
    for name, identities in (("Feature", dataset.features.index), ("Label", dataset.labels.index)):
        if any(not isinstance(value, str) or not value.strip() for value in identities):
            raise ValueError(f"{name} split identities must be nonblank strings")
        if identities.duplicated().any():
            raise ValueError(f"{name} split identities must be unique")
    modeling_ids = set(dataset.features.index)
    if modeling_ids != set(dataset.labels.index):
        raise ValueError("Split feature and label identities must match")
    if modeling_ids.intersection(dataset.display_features.index):
        raise ValueError("Display identities must be disjoint from modeling split identities")
    if dataset.labels.isna().any() or not dataset.labels.isin([0, 1]).all():
        raise ValueError("Split class labels must be non-null binary values")
    if not restaurant_ids:
        raise ValueError("Selected split identities must not be empty")
    if len(restaurant_ids) != len(set(restaurant_ids)):
        raise ValueError("Selected split identities must be unique")
    unknown = set(restaurant_ids).difference(modeling_ids)
    if unknown:
        raise ValueError(f"Selected split identities are unknown to modeling data: {sorted(unknown)}")
    return dataset.labels.loc[sorted(restaurant_ids)]


def validate_split_class_counts(labels: pd.Series, minimum: int, context: str) -> None:
    """Require enough members of each binary class for the requested partition."""
    counts = labels.value_counts().reindex([0, 1], fill_value=0)
    if (counts < minimum).any():
        raise ValueError(
            f"{context} requires at least {minimum} rows of each class; "
            f"class counts={counts.to_dict()}"
        )


def stratified_holdout(dataset: ClassificationDataset, config: PipelineConfig) -> DatasetSplit:
    """Return seeded stratified train/test identities without transforming features."""
    if config.split.strategy != "stratified_holdout":
        raise ValueError(f"Unsupported holdout strategy: {config.split.strategy}")
    labels = split_labels(dataset, tuple(dataset.features.index))
    validate_split_class_counts(labels, 2, "Stratified holdout")
    test_count = ceil(len(labels) * config.split.test_size)
    train_count = len(labels) - test_count
    if min(train_count, test_count) < 2:
        raise ValueError(
            "Stratified holdout needs room for both classes in each partition; "
            f"train rows={train_count}, test rows={test_count}, test_size={config.split.test_size}"
        )
    train_ids, test_ids = train_test_split(
        labels.index.tolist(), test_size=config.split.test_size,
        random_state=config.seed, shuffle=True, stratify=labels.to_numpy(),
    )
    validate_split_class_counts(labels.loc[train_ids], 1, "Holdout training partition")
    validate_split_class_counts(labels.loc[test_ids], 1, "Holdout test partition")
    return DatasetSplit(train_ids=tuple(sorted(train_ids)), test_ids=tuple(sorted(test_ids)))


def stratified_folds(
    dataset: ClassificationDataset, restaurant_ids: tuple[str, ...], config: PipelineConfig,
) -> tuple[DatasetSplit, ...]:
    """Partition an explicit modeling universe; fold test identities are validation rows."""
    if config.split.cv_strategy != "stratified_k_fold":
        raise ValueError(f"Unsupported CV strategy: {config.split.cv_strategy}")
    labels = split_labels(dataset, restaurant_ids)
    validate_split_class_counts(labels, config.split.cv_folds, "Stratified folds")
    splitter = StratifiedKFold(n_splits=config.split.cv_folds, shuffle=True, random_state=config.seed)
    return tuple(
        DatasetSplit(train_ids=tuple(labels.index[train]), test_ids=tuple(labels.index[test]))
        for train, test in splitter.split(labels.index, labels.to_numpy())
    )

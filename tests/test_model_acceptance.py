"""Independent requirements for the ordered Phase 4 modeling skeleton."""

from dataclasses import replace
from math import log
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss

from michelin.config import AppConfig, load_config
from michelin.features.transforms import transform_features
from michelin.model.dataset import ClassificationDataset, DatasetSplit
from michelin.model.evaluate import evaluate_probabilities
from michelin.model.train import predict_probabilities, train_skeleton


@pytest.fixture
def config() -> AppConfig:
    return load_config(Path("config/pipeline.yaml"), Path("config/features.yaml"))


@pytest.fixture
def modeling_case() -> tuple[ClassificationDataset, DatasetSplit]:
    identities = pd.Index([f"r-{i:03d}" for i in range(100)], name="restaurant_id")
    labels = pd.Series([int(i % 5 == 0) for i in range(100)], index=identities)
    features = pd.DataFrame({
        "average_rating": [4.9 if value else 2.0 for value in labels],
        "review_count": [1000 if value else 10 for value in labels],
        "neighborhood_density": [20 if value else 2 for value in labels],
        "price_range": ["2000+" if value else "200-400" for value in labels],
        "cuisine": ["restaurant"] * 100,
        "city": ["New Taipei" if i % 2 else "Taipei" for i in range(100)],
    }, index=identities)
    dataset = ClassificationDataset(features, labels, labels.eq(0), features.iloc[:1].rename(index={"r-000": "display"}))
    return dataset, DatasetSplit(tuple(identities[:80]), tuple(identities[80:]))


def test_repeat_training_is_deterministic_plain_logistic_and_pure(
    modeling_case: tuple[ClassificationDataset, DatasetSplit], config: AppConfig,
) -> None:
    dataset, split = modeling_case
    raw_before, labels_before = dataset.features.copy(deep=True), dataset.labels.copy(deep=True)
    np_state, python_state = np.random.get_state(), random.getstate()
    first, second = train_skeleton(dataset, split, config), train_skeleton(dataset, split, config)
    pd.testing.assert_series_equal(first.test_probabilities, second.test_probabilities)
    pd.testing.assert_series_equal(first.train_probabilities, second.train_probabilities)
    assert first.test_probabilities.index.tolist() == list(split.test_ids)
    assert first.train_probabilities.index.tolist() == list(split.train_ids)
    assert isinstance(first.model.estimator, LogisticRegression)
    assert first.model.estimator.class_weight is None
    assert first.model.estimator.random_state == config.pipeline.seed
    pd.testing.assert_frame_equal(dataset.features, raw_before)
    pd.testing.assert_series_equal(dataset.labels, labels_before)
    assert random.getstate() == python_state
    np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
    assert np.random.get_state()[2:] == np_state[2:]


def test_initial_prior_floor_ceiling_and_capacity_have_numerical_evidence(
    modeling_case: tuple[ClassificationDataset, DatasetSplit], config: AppConfig,
) -> None:
    dataset, split = modeling_case
    result = train_skeleton(dataset, split, config)
    diagnostics = result.diagnostics
    p = float(dataset.labels.loc[list(split.train_ids)].mean())
    assert diagnostics.initial_probability == pytest.approx(p, abs=1e-12)
    assert diagnostics.initial_loss == pytest.approx(-p * log(p) - (1 - p) * log(1 - p), abs=1e-12)
    np.testing.assert_allclose(diagnostics.constant_probabilities, p, rtol=0, atol=1e-6)
    assert diagnostics.constant_probabilities.index.tolist() == list(split.test_ids)
    truth = dataset.labels.loc[list(split.test_ids)]
    assert average_precision_score(truth, result.test_probabilities) > average_precision_score(truth, diagnostics.constant_probabilities)
    assert diagnostics.overfit_labels_match
    assert diagnostics.overfit_loss < 0.01
    assert diagnostics.increased_capacity_train_loss < diagnostics.base_train_loss


def test_holdout_labels_and_values_cannot_change_training_state(
    modeling_case: tuple[ClassificationDataset, DatasetSplit], config: AppConfig,
) -> None:
    dataset, split = modeling_case
    first = train_skeleton(dataset, split, config)
    changed_features = dataset.features.copy(deep=True)
    changed_features.loc[list(split.test_ids), "review_count"] = 999999
    changed_labels = dataset.labels.copy(deep=True)
    changed_labels.loc[list(split.test_ids)] = 1 - changed_labels.loc[list(split.test_ids)]
    changed = train_skeleton(replace(dataset, features=changed_features, labels=changed_labels), split, config)
    assert first.model.transforms == changed.model.transforms
    np.testing.assert_array_equal(first.model.estimator.coef_, changed.model.estimator.coef_)
    np.testing.assert_array_equal(first.model.estimator.intercept_, changed.model.estimator.intercept_)
    assert first.diagnostics.initial_probability == changed.diagnostics.initial_probability


def test_prediction_is_row_independent_and_rejects_or_ignores_internal_fields(
    modeling_case: tuple[ClassificationDataset, DatasetSplit], config: AppConfig,
) -> None:
    dataset, split = modeling_case
    result = train_skeleton(dataset, split, config)
    rows = dataset.features.loc[list(split.test_ids)]
    target = rows.index[0]
    alone = predict_probabilities(result.model, rows.loc[[target]])
    together = predict_probabilities(result.model, rows.iloc[::-1])
    assert alone.loc[target] == pytest.approx(together.loc[target], abs=1e-14)
    contaminated = rows.assign(Award="3 Stars", Description="internal", **{"class": 1})
    try:
        scores = predict_probabilities(result.model, contaminated)
    except ValueError:
        return
    pd.testing.assert_series_equal(scores, predict_probabilities(result.model, rows))


def test_watchlist_records_exact_raw_and_model_input_rows(
    modeling_case: tuple[ClassificationDataset, DatasetSplit], config: AppConfig,
) -> None:
    dataset, split = modeling_case
    result = train_skeleton(dataset, split, config)
    diagnostics = result.diagnostics
    assert not diagnostics.watchlist_raw.empty
    assert diagnostics.watchlist_raw.index.is_unique
    pd.testing.assert_frame_equal(diagnostics.watchlist_raw, dataset.features.loc[diagnostics.watchlist_raw.index])
    pd.testing.assert_frame_equal(diagnostics.watchlist_transformed, transform_features(diagnostics.watchlist_raw, result.model.transforms))
    assert diagnostics.watchlist_probabilities.index.equals(diagnostics.watchlist_raw.index)
    assert diagnostics.watchlist_probabilities.shape[1] >= 2
    assert np.isfinite(diagnostics.watchlist_probabilities.to_numpy()).all()


def test_full_population_metrics_and_hard_negative_city_slices_are_exact() -> None:
    identities = pd.Index(["a", "b", "c", "d", "e", "f"])
    labels = pd.Series([0, 1, 0, 1, 0, 0], index=identities)
    probabilities = pd.Series([0.1, 0.8, 0.6, 0.4, 0.2, 0.3], index=identities)
    hard_negative = pd.Series([True, False, True, False, False, False], index=identities)
    cities = pd.Series(["Taipei"] * 4 + ["New Taipei"] * 2, index=identities)
    report = evaluate_probabilities(labels, probabilities, hard_negative, cities, 0.5, 3)
    assert report.metrics["log_loss"] == pytest.approx(log_loss(labels, probabilities), abs=1e-14)
    assert report.metrics["average_precision"] == pytest.approx(average_precision_score(labels, probabilities), abs=1e-14)
    assert report.metrics["precision"] == 0.5
    assert report.metrics["recall"] == 0.5
    assert report.metrics["precision_at_k"] == pytest.approx(2 / 3)
    assert np.asarray(report.confusion_matrix).tolist() == [[3, 1], [1, 1]]
    assert report.predicted_positive_count == 2
    assert report.known_positive_count == 2
    assert report.hard_negative["count"] == 2
    assert report.hard_negative["error_rate"] == 0.5
    assert report.cities["New Taipei"]["roc_auc"] is None
    assert report.cities["New Taipei"]["undefined_reason"]


@pytest.mark.parametrize("scores", [[0.1, float("nan")], [-0.1, 0.8], [0.2, 1.1]])
def test_invalid_probabilities_fail_explicitly(scores: list[float]) -> None:
    labels = pd.Series([0, 1], index=["a", "b"])
    with pytest.raises(ValueError):
        evaluate_probabilities(labels, pd.Series(scores, index=labels.index), labels.eq(0), pd.Series(["Taipei"] * 2, index=labels.index), 0.5, 1)


def test_misaligned_probabilities_never_silently_use_position() -> None:
    labels = pd.Series([0, 1], index=["a", "b"])
    with pytest.raises(ValueError):
        evaluate_probabilities(labels, pd.Series([0.1, 0.9], index=["a", "other"]), labels.eq(0), pd.Series(["Taipei"] * 2, index=labels.index), 0.5, 1)

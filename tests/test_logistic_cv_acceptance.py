"""Specification-derived checks for fixed LR on the shared outer folds."""

from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from michelin.config import AppConfig, PipelineConfig
from michelin.model.dataset import ClassificationDataset, DatasetSplit
from michelin.model.evaluate import EvaluationReport
from michelin.model.settings import load_modeling_settings
from michelin.model.train import FittedClassifier, fit_explanation, predict_probabilities
from scripts import evaluate_logistic_cv as candidate
from test_model_acceptance import config, modeling_case


@pytest.fixture
def cv_case(modeling_case: tuple[ClassificationDataset, DatasetSplit]) -> ClassificationDataset:
    """Vary values within each class so full-population transforms cannot pass."""
    dataset, _ = modeling_case
    positions = np.arange(len(dataset.labels))
    features = dataset.features.assign(
        average_rating=3.0 + (positions % 17) / 10,
        review_count=np.where(positions % 9 == 0, np.nan, positions ** 2 + 5),
        neighborhood_density=(positions * 7) % 31,
        cuisine=[f"cuisine-{i % 7}" if i != 17 else "unique-held-out" for i in positions],
        city=["Taipei" if label or i % 2 == 0 else "New Taipei" for i, label in enumerate(dataset.labels)],
    )
    display = dataset.display_features.assign(review_count=10_000_000, cuisine="display-only")
    return replace(dataset, features=features, display_features=display)


def expected_folds(dataset: ClassificationDataset, config: AppConfig) -> tuple[DatasetSplit, ...]:
    """Reproduce required partition identities directly through sklearn."""
    labels = dataset.labels.sort_index()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.pipeline.seed)
    return tuple(
        DatasetSplit(tuple(labels.index[train]), tuple(labels.index[validation]))
        for train, validation in splitter.split(labels.index, labels.to_numpy())
    )


def independently_encoded(
    training: pd.DataFrame, scoring: pd.DataFrame, config: AppConfig,
) -> pd.DataFrame:
    """Apply independently calculated training-only transform statistics."""
    columns: dict[str, np.ndarray] = {}
    for name in config.features.feature_schema.numeric:
        observed = training[name].to_numpy(dtype=float)
        median = float(np.nanmedian(observed))
        imputed = np.where(np.isnan(observed), median, observed)
        scale = float(imputed.std())
        raw = scoring[name].to_numpy(dtype=float)
        columns[name] = (np.where(np.isnan(raw), median, raw) - imputed.mean()) / (scale if scale else 1.0)
    for name in config.features.feature_schema.categorical:
        columns[f"{name}__missing"] = scoring[name].isna().to_numpy(dtype=float)
        for category in sorted(training[name].dropna().unique()):
            columns[f"{name}__value:{category}"] = scoring[name].eq(category).to_numpy(dtype=float)
    return pd.DataFrame(columns, index=scoring.index)


def assert_report_matches(
    report: EvaluationReport, dataset: ClassificationDataset, scores: pd.Series,
) -> None:
    """Recalculate population and slice metrics without production evaluators."""
    labels = dataset.labels.loc[scores.index]
    predicted = scores.ge(0.2)  #experimentation: fixed cutoff 20%.
    k = max(1, int(labels.sum()))
    expected = {
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "precision_at_k": float(labels.loc[scores.sort_index().sort_values(ascending=False, kind="stable").head(k).index].mean()),
    }
    assert report.metrics == pytest.approx(expected, abs=1e-14)
    np.testing.assert_array_equal(report.confusion_matrix, confusion_matrix(labels, predicted, labels=[0, 1]))
    assert report.known_positive_count == int(labels.sum())
    assert report.predicted_positive_count == int(predicted.sum())
    hard = dataset.hard_negative.loc[scores.index]
    assert report.hard_negative["count"] == int(hard.sum())
    assert report.hard_negative["error_rate"] == pytest.approx(float(predicted.loc[hard].mean()))
    cities = dataset.features.loc[scores.index, "city"]
    assert set(report.cities) == set(cities)
    for city in cities.unique():
        selected = cities.index[cities.eq(city)]
        city_labels, city_scores = labels.loc[selected], scores.loc[selected]
        metrics = report.cities[city]
        assert metrics["log_loss"] == pytest.approx(log_loss(city_labels, city_scores, labels=[0, 1]), abs=1e-14)
        assert metrics["precision"] == pytest.approx(precision_score(city_labels, city_scores.ge(0.2), zero_division=0))  #experimentation: fixed cutoff 20%.
        assert metrics["recall"] == pytest.approx(recall_score(city_labels, city_scores.ge(0.2), zero_division=0))  #experimentation: fixed cutoff 20%.
        top = city_scores.sort_index().sort_values(ascending=False, kind="stable").head(k).index
        assert metrics["precision_at_k"] == pytest.approx(float(city_labels.loc[top].mean()))
        if city_labels.eq(1).any():
            assert metrics["average_precision"] == pytest.approx(average_precision_score(city_labels, city_scores))
        else:
            assert metrics["average_precision"] is None
            assert metrics["undefined_reason"]
        if city_labels.nunique() == 2:
            assert metrics["roc_auc"] == pytest.approx(roc_auc_score(city_labels, city_scores))
            assert metrics["undefined_reason"] is None
        else:
            assert metrics["roc_auc"] is None
            assert metrics["undefined_reason"]


def test_exact_outer_partition_and_deterministic_pure_result(cv_case: ClassificationDataset, config: AppConfig) -> None:
    before = replace(
        cv_case, features=cv_case.features.copy(deep=True), labels=cv_case.labels.copy(deep=True),
        hard_negative=cv_case.hard_negative.copy(deep=True), display_features=cv_case.display_features.copy(deep=True),
    )
    config_before = config.model_dump()
    numpy_before, python_before = np.random.get_state(), random.getstate()
    first = candidate.run_logistic_cv(cv_case, config)
    second = candidate.run_logistic_cv(cv_case, config)
    assert isinstance(first, candidate.LogisticCVResult)
    assert first.folds == second.folds == expected_folds(cv_case, config)
    assert len(first.folds) == len(first.fold_evaluations) == 5
    assert first.probabilities.index.tolist() == sorted(cv_case.labels.index)
    assert first.probabilities.index.is_unique
    assert set(first.probabilities.index).isdisjoint(cv_case.display_features.index)
    assert np.isfinite(first.probabilities).all()
    assert first.probabilities.between(0, 1).all()
    validation_ids = [identity for fold in first.folds for identity in fold.test_ids]
    assert len(validation_ids) == len(set(validation_ids)) == len(cv_case.labels)
    assert set(validation_ids) == set(cv_case.labels.index)
    for fold in first.folds:
        assert set(fold.train_ids).isdisjoint(fold.test_ids)
        assert set(fold.train_ids) | set(fold.test_ids) == set(cv_case.labels.index)
    pd.testing.assert_series_equal(first.probabilities, second.probabilities, check_exact=True)
    assert asdict(first.evaluation) == asdict(second.evaluation)
    assert asdict(first.calibration) == asdict(second.calibration)
    assert first.fold_evaluations == second.fold_evaluations
    with pytest.raises(FrozenInstanceError):
        setattr(first, "folds", ())
    pd.testing.assert_frame_equal(cv_case.features, before.features)
    pd.testing.assert_frame_equal(cv_case.display_features, before.display_features)
    pd.testing.assert_series_equal(cv_case.labels, before.labels)
    pd.testing.assert_series_equal(cv_case.hard_negative, before.hard_negative)
    assert config.model_dump() == config_before
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]


def test_predictions_reconstruct_independent_fold_lr(cv_case: ClassificationDataset, config: AppConfig) -> None:
    result = candidate.run_logistic_cv(cv_case, config)
    settings = load_modeling_settings(Path("config/modeling.yaml"))
    for fold in expected_folds(cv_case, config):
        training = cv_case.features.loc[list(fold.train_ids)]
        validation = cv_case.features.loc[list(fold.test_ids)]
        matrix = independently_encoded(training, training, config)
        labels = cv_case.labels.loc[list(fold.train_ids)]
        estimator = LogisticRegression(
            C=settings.logistic_c, max_iter=settings.logistic_max_iter, tol=settings.logistic_tolerance,
            solver="lbfgs", random_state=config.pipeline.seed, class_weight=None, warm_start=True,
        )
        prior = float(labels.mean())
        estimator.classes_ = np.array([0, 1])
        estimator.coef_ = np.zeros((1, matrix.shape[1]))
        estimator.intercept_ = np.array([np.log(prior / (1 - prior))])
        estimator.n_features_in_ = matrix.shape[1]
        estimator.feature_names_in_ = np.asarray(matrix.columns, dtype=object)
        estimator.fit(matrix, labels)
        expected = estimator.predict_proba(independently_encoded(training, validation, config))[:, 1]
        np.testing.assert_allclose(result.probabilities.loc[list(fold.test_ids)], expected, rtol=0, atol=1e-12)


def test_validation_changes_do_not_change_its_fitted_model(
    cv_case: ClassificationDataset, config: AppConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folds = expected_folds(cv_case, config)
    fitted: list[FittedClassifier] = []
    fit_ids: list[tuple[str, ...]] = []

    def capture_fit(dataset: ClassificationDataset, train_ids: tuple[str, ...], settings: AppConfig) -> FittedClassifier:
        model = fit_explanation(dataset, train_ids, settings)
        fitted.append(model)
        fit_ids.append(train_ids)
        return model

    def fixed_folds(dataset: ClassificationDataset, restaurant_ids: tuple[str, ...], settings: PipelineConfig) -> tuple[DatasetSplit, ...]:
        assert set(restaurant_ids) == set(cv_case.labels.index)
        return folds

    monkeypatch.setattr(candidate, "fit_explanation", capture_fit)
    monkeypatch.setattr(candidate, "stratified_folds", fixed_folds)
    candidate.run_logistic_cv(cv_case, config)
    target = list(folds[0].test_ids)
    altered_features, altered_labels = cv_case.features.copy(deep=True), cv_case.labels.copy(deep=True)
    altered_features.loc[target, "review_count"] = 9_999_999
    altered_features.loc[target, "cuisine"] = "validation-only-category"
    altered_labels.loc[target] = 1 - altered_labels.loc[target]
    changed = replace(
        cv_case, features=altered_features, labels=altered_labels,
        hard_negative=cv_case.hard_negative & altered_labels.eq(0),
        display_features=cv_case.display_features.assign(review_count=999_999_999),
    )
    candidate.run_logistic_cv(changed, config)
    assert fit_ids == [fold.train_ids for fold in folds] * 2
    assert len(fitted) == 10
    original, altered = fitted[0], fitted[5]
    assert original.transforms == altered.transforms
    np.testing.assert_array_equal(original.estimator.coef_, altered.estimator.coef_)
    np.testing.assert_array_equal(original.estimator.intercept_, altered.estimator.intercept_)
    original_rows = cv_case.features.loc[target]
    pd.testing.assert_series_equal(predict_probabilities(original, original_rows), predict_probabilities(altered, original_rows), check_exact=True)


def test_controlled_scores_reconcile_metrics_and_slices(
    cv_case: ClassificationDataset, config: AppConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = pd.Series(
        [0.5, 0.49999, 0.9, 0.1, 0.5, 0.2, 0.8, 0.5, 0.3, 0.6] * 10,
        index=cv_case.labels.index, name="probability",
    )
    scored_ids: list[str] = []

    def controlled_predictions(model: FittedClassifier, raw_rows: pd.DataFrame) -> pd.Series:
        assert set(raw_rows.index).isdisjoint(cv_case.display_features.index)
        scored_ids.extend(raw_rows.index)
        return scores.loc[raw_rows.index].copy(deep=True)

    monkeypatch.setattr(candidate, "predict_probabilities", controlled_predictions)
    result = candidate.run_logistic_cv(cv_case, config)
    assert len(scored_ids) == len(set(scored_ids)) == len(scores)
    pd.testing.assert_series_equal(result.probabilities, scores)
    assert_report_matches(result.evaluation, cv_case, scores)
    assert result.calibration.brier_score == pytest.approx(brier_score_loss(cv_case.labels, scores), abs=1e-14)
    assert sum(bin_report.count for bin_report in result.calibration.bins) == len(scores)
    for fold, report in zip(result.folds, result.fold_evaluations, strict=True):
        assert_report_matches(report, cv_case, scores.loc[list(fold.test_ids)])

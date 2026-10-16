"""Independent Phase 5 acceptance: honest calibrated scores and LR explanations."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from michelin.config import AppConfig
from michelin.features.transforms import transform_features
from michelin.model.dataset import ClassificationDataset, DatasetSplit
from michelin.model.thresholds import similar_to_starred
from michelin.model.train import explain_rows, fit_calibrated, predict_calibrated, train_calibrated
from test_model_acceptance import config, modeling_case


def test_threshold_is_single_explicit_inclusive_probability_cutoff() -> None:
    probabilities = pd.Series([0.0, 0.199999, 0.2, 1.0], index=["a", "b", "c", "d"])  #experimentation: test the 20% boundary.
    before = probabilities.copy(deep=True)
    flags = similar_to_starred(probabilities, 0.2)  #experimentation: fixed cutoff 20%.
    pd.testing.assert_series_equal(flags, pd.Series([False, False, True, True], index=probabilities.index))
    pd.testing.assert_series_equal(probabilities, before)


@pytest.mark.parametrize("cutoff", [-0.1, 1.1, float("nan")])
def test_threshold_rejects_invalid_cutoff(cutoff: float) -> None:
    with pytest.raises(ValueError):
        similar_to_starred(pd.Series([0.2], index=["a"]), cutoff)


def test_nested_oof_scores_reproduce_models_excluding_each_validation_row(modeling_case: tuple[ClassificationDataset, DatasetSplit], config: AppConfig) -> None:
    dataset, split = modeling_case
    result = train_calibrated(dataset, split, config)
    assert isinstance(result.explanation.estimator, LogisticRegression)
    assert isinstance(result.prediction.estimator, GradientBoostingClassifier)
    assert isinstance(result.prediction.calibrator, LogisticRegression)
    assert result.cutoff == config.pipeline.threshold.similar_to_starred_cutoff == 0.2  #experimentation: fixed cutoff 20%.
    assert result.oof_probabilities.index.tolist() == dataset.features.index.tolist()
    assert result.holdout_probabilities.index.tolist() == list(split.test_ids)
    assert "display" not in result.oof_probabilities.index
    assert np.isfinite(result.oof_probabilities).all()
    assert result.oof_probabilities.between(0, 1).all()
    validation_ids = []
    for fold in result.oof_folds:
        validation_ids.extend(fold.validation_ids)
        assert set(fold.validation_ids).isdisjoint(fold.train_ids)
        assert set(fold.validation_ids).isdisjoint(fold.calibration_ids)
        assert set(fold.train_ids) | set(fold.validation_ids) == set(dataset.labels.index)
        independent = fit_calibrated(dataset, fold.train_ids, config)
        expected = predict_calibrated(independent, dataset.features.loc[list(fold.validation_ids)])
        np.testing.assert_allclose(result.oof_probabilities.loc[list(fold.validation_ids)], expected, rtol=0, atol=1e-12)
    assert len(validation_ids) == len(set(validation_ids)) == len(dataset.labels)
    assert set(validation_ids) == set(dataset.labels.index)


def test_holdout_fit_is_independent_of_holdout_labels_values_and_display(modeling_case: tuple[ClassificationDataset, DatasetSplit], config: AppConfig) -> None:
    dataset, split = modeling_case
    first = fit_calibrated(dataset, split.train_ids, config)
    labels = dataset.labels.copy(deep=True)
    labels.loc[list(split.test_ids)] = 1 - labels.loc[list(split.test_ids)]
    raw = dataset.features.copy(deep=True)
    raw.loc[list(split.test_ids), "review_count"] = 999999
    changed = replace(dataset, labels=labels, features=raw, display_features=dataset.display_features.assign(review_count=999999))
    second = fit_calibrated(changed, split.train_ids, config)
    assert first.transforms == second.transforms
    np.testing.assert_array_equal(first.calibrator.coef_, second.calibrator.coef_)
    np.testing.assert_array_equal(first.calibrator.intercept_, second.calibrator.intercept_)
    original_rows = dataset.features.loc[list(split.test_ids)]
    pd.testing.assert_series_equal(predict_calibrated(first, original_rows), predict_calibrated(second, original_rows))


def test_holdout_and_full_refits_are_separate_and_reproducible(modeling_case: tuple[ClassificationDataset, DatasetSplit], config: AppConfig) -> None:
    dataset, split = modeling_case
    first, second = train_calibrated(dataset, split, config), train_calibrated(dataset, split, config)
    assert first.final_prediction is not first.prediction
    assert first.final_explanation is not first.explanation
    pd.testing.assert_series_equal(first.oof_probabilities, second.oof_probabilities)
    pd.testing.assert_series_equal(first.holdout_probabilities, second.holdout_probabilities)
    expected = predict_calibrated(fit_calibrated(dataset, split.train_ids, config), dataset.features.loc[list(split.test_ids)])
    pd.testing.assert_series_equal(first.holdout_probabilities, expected)
    full = fit_calibrated(dataset, tuple(dataset.features.index), config)
    pd.testing.assert_series_equal(predict_calibrated(first.final_prediction, dataset.display_features), predict_calibrated(full, dataset.display_features))


def test_lr_explanations_rank_top_five_signed_contributions_and_reconcile_logits(modeling_case: tuple[ClassificationDataset, DatasetSplit], config: AppConfig) -> None:
    dataset, split = modeling_case
    result = train_calibrated(dataset, split, config)
    raw = dataset.features.loc[list(split.test_ids)]
    explanation = explain_rows(result.explanation, raw)
    encoded = transform_features(raw, result.explanation.transforms)
    coefficients = result.explanation.estimator.coef_[0]
    np.testing.assert_allclose(explanation.global_importance.loc[encoded.columns], coefficients, rtol=0, atol=1e-12)
    np.testing.assert_allclose(explanation.contributions.loc[raw.index, encoded.columns], encoded.to_numpy() * coefficients, rtol=0, atol=1e-12)
    np.testing.assert_allclose(explanation.contributions.sum(axis=1) + result.explanation.estimator.intercept_[0], result.explanation.estimator.decision_function(encoded), rtol=0, atol=1e-12)
    for restaurant_id in raw.index:
        selected = explanation.top_five[restaurant_id]
        assert len(selected) == len(set(selected)) == 5
        absolute = explanation.contributions.loc[restaurant_id].abs()
        chosen = absolute.loc[list(selected)].to_numpy()
        assert np.all(chosen[:-1] >= chosen[1:])
        assert min(chosen) >= absolute.drop(index=list(selected)).max()
    caveat = explanation.caveat.casefold()
    assert "correlation" in caveat and "causation" in caveat
    assert "logistic" in caveat and ("gbm" in caveat or "gradient" in caveat)

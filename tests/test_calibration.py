"""Developer coverage derived from training-only calibration requirements."""

from dataclasses import replace
import random

import numpy as np
import pandas as pd
import pytest

from michelin.config import AppConfig
from michelin.model.dataset import ClassificationDataset, DatasetSplit, stratified_folds
from michelin.features.transforms import fit_transforms, transform_features
from michelin.model.settings import load_modeling_settings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from michelin.model.train import fit_calibrated, predict_calibrated, train_calibrated
from test_model_acceptance import config, modeling_case


def test_calibration_uses_inner_oof_not_training_predictions(modeling_case: tuple[ClassificationDataset, DatasetSplit], config: AppConfig) -> None:
    dataset, split = modeling_case
    model = fit_calibrated(dataset, split.train_ids, config)
    assert model.calibrator.class_weight is None
    assert model.estimator.n_iter_no_change is None
    assert model.estimator.random_state == config.pipeline.seed
    assert model.calibrator.n_features_in_ == 1
    alone = predict_calibrated(model, dataset.features.iloc[[0]])
    together = predict_calibrated(model, dataset.features.iloc[::-1])
    assert alone.iloc[0] == together.loc[alone.index[0]]
    with pytest.raises(ValueError, match="schema"):
        predict_calibrated(model, dataset.features.assign(Award="3 Stars"))


def test_diagnostics_are_training_only_and_rng_is_untouched(modeling_case: tuple[ClassificationDataset, DatasetSplit], config: AppConfig) -> None:
    dataset, split = modeling_case
    python_state, numpy_state = random.getstate(), np.random.get_state()
    before = dataset.features.copy(deep=True)
    result = train_calibrated(dataset, split, config)
    diagnostics = result.diagnostics
    assert len(diagnostics.stages) == 5
    for stage in diagnostics.stages:
        assert set(stage.probabilities.index) == set(split.train_ids)
        assert len(stage.fold_average_precision) == config.pipeline.split.cv_folds
    altered = dataset.labels.copy(deep=True)
    altered.loc[list(split.test_ids)] = 1 - altered.loc[list(split.test_ids)]
    second = train_calibrated(replace(dataset, labels=altered, hard_negative=altered.eq(0)), split, config)
    for first_stage, second_stage in zip(diagnostics.stages, second.diagnostics.stages, strict=True):
        pd.testing.assert_series_equal(first_stage.probabilities, second_stage.probabilities)
    pd.testing.assert_frame_equal(dataset.features, before)
    assert random.getstate() == python_state
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]


def test_unsupported_prediction_family_fails(modeling_case: tuple[ClassificationDataset, DatasetSplit], config: AppConfig) -> None:
    dataset, split = modeling_case
    invalid = config.model_copy(update={"pipeline": config.pipeline.model_copy(update={"models": config.pipeline.models.model_copy(update={"prediction": "unexpected"})})})
    with pytest.raises(ValueError, match="lightweight_gbm"):
        fit_calibrated(dataset, split.train_ids, invalid)


def test_platt_coefficients_match_independent_inner_oof_fit(modeling_case: tuple[ClassificationDataset, DatasetSplit], config: AppConfig) -> None:
    dataset, split = modeling_case
    settings = load_modeling_settings(Path("config/modeling.yaml"))
    pieces = []
    for fold in stratified_folds(dataset, split.train_ids, config.pipeline):
        raw = dataset.features.loc[list(fold.train_ids)]
        transforms = fit_transforms(raw, config.features)
        base = GradientBoostingClassifier(n_estimators=settings.prediction_rounds, learning_rate=settings.boosting_learning_rate, max_depth=settings.boosting_depth, random_state=config.pipeline.seed, n_iter_no_change=None, subsample=1.0).fit(transform_features(raw, transforms), dataset.labels.loc[list(fold.train_ids)])
        pieces.append(pd.Series(base.decision_function(transform_features(dataset.features.loc[list(fold.test_ids)], transforms)), index=list(fold.test_ids)))
    logits = pd.concat(pieces).loc[list(split.train_ids)].to_numpy().reshape(-1, 1)
    expected = LogisticRegression(C=settings.calibration_c, max_iter=settings.logistic_max_iter, tol=settings.logistic_tolerance, solver="lbfgs", class_weight=None, random_state=config.pipeline.seed).fit(logits, dataset.labels.loc[list(split.train_ids)])
    actual = fit_calibrated(dataset, split.train_ids, config)
    np.testing.assert_array_equal(expected.coef_, actual.calibrator.coef_)
    np.testing.assert_array_equal(expected.intercept_, actual.calibrator.intercept_)

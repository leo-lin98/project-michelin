"""Local approved phases 4–6 modeling, excluding publication artifacts."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from michelin.config import AppConfig
from michelin.export.parity import ParityReport, onnx_probabilities, verify_parity
from michelin.export.to_onnx import export_prediction
from michelin.features.transforms import transform_features
from michelin.model.dataset import ClassificationDataset, DatasetSplit
from michelin.model.train import CalibratedResult, SkeletonResult, predict_calibrated, train_calibrated, train_skeleton


@dataclass(frozen=True)
class ModelingRun:
    skeleton: SkeletonResult
    calibrated: CalibratedResult
    holdout_parity: ParityReport
    final_parity: ParityReport


def validate_skeleton(result: SkeletonResult, dataset: ClassificationDataset, split: DatasetSplit) -> None:
    """Enforce the measured prior, floor, ceiling and capacity checks before Phase 5."""
    prior = float(dataset.labels.loc[list(split.train_ids)].mean())
    entropy = -prior * np.log(prior) - (1 - prior) * np.log(1 - prior)
    diagnostic = result.diagnostics
    truth = dataset.labels.loc[result.test_probabilities.index]
    if not np.isclose(diagnostic.initial_probability, prior, rtol=0, atol=1e-12) or not np.isclose(diagnostic.initial_loss, entropy, rtol=0, atol=1e-12):
        raise ValueError("Phase 4 initialization must match training prior and base-rate entropy")
    if not np.allclose(diagnostic.constant_probabilities, prior, rtol=0, atol=1e-6):
        raise ValueError("Phase 4 input-independent floor must collapse to training prior")
    if average_precision_score(truth, result.test_probabilities) <= average_precision_score(truth, diagnostic.constant_probabilities.reindex(truth.index)):
        raise ValueError("Phase 4 skeleton must beat input-independent floor average precision")
    if not diagnostic.overfit_labels_match or not diagnostic.overfit_loss < 0.01:
        raise ValueError("Phase 4 handful ceiling must match labels with near-zero loss")
    if not diagnostic.increased_capacity_train_loss < diagnostic.base_train_loss:
        raise ValueError("Phase 4 increased capacity must reduce training loss")


def oof_base_rate(dataset: ClassificationDataset, result: CalibratedResult) -> pd.Series:
    """Predict each outer validation fold from its own training prevalence only."""
    return pd.concat(tuple(pd.Series(float(dataset.labels.loc[list(fold.train_ids)].mean()), index=list(fold.validation_ids)) for fold in result.oof_folds)).reindex(result.oof_probabilities.index)


def validate_calibrated(result: CalibratedResult, dataset: ClassificationDataset, split: DatasetSplit) -> None:
    """Require honest holdout and nested OOF ranking lift before any export."""
    holdout_floor = pd.Series(float(dataset.labels.loc[list(split.train_ids)].mean()), index=result.holdout_probabilities.index)
    for name, scores, floor in (("holdout", result.holdout_probabilities, holdout_floor), ("OOF", result.oof_probabilities, oof_base_rate(dataset, result))):
        truth = dataset.labels.loc[scores.index]
        if average_precision_score(truth, scores) <= average_precision_score(truth, floor):
            raise ValueError(f"Phase 5 {name} calibrated average precision must beat train-only base rate")


def run_modeling(dataset: ClassificationDataset, split: DatasetSplit, config: AppConfig, output: Path) -> ModelingRun:
    """Train without display labels and independently verify both calibrated exports."""
    skeleton = train_skeleton(dataset, split, config)
    validate_skeleton(skeleton, dataset, split)
    repeated_skeleton = train_skeleton(dataset, split, config)
    pd.testing.assert_series_equal(skeleton.train_probabilities, repeated_skeleton.train_probabilities, check_exact=True)
    pd.testing.assert_series_equal(skeleton.test_probabilities, repeated_skeleton.test_probabilities, check_exact=True)
    pd.testing.assert_frame_equal(skeleton.diagnostics.watchlist_probabilities, repeated_skeleton.diagnostics.watchlist_probabilities, check_exact=True)
    calibrated = train_calibrated(dataset, split, config)
    validate_calibrated(calibrated, dataset, split)
    raw = dataset.features.loc[list(split.test_ids)]
    holdout_path = output / "holdout.quantized.onnx"
    final_path = output / "final.quantized.onnx"
    export_prediction(calibrated.prediction, holdout_path, 17)
    export_prediction(calibrated.final_prediction, final_path, 17)
    holdout_parity = verify_parity(calibrated.holdout_probabilities, onnx_probabilities(holdout_path, transform_features(raw, calibrated.prediction.transforms)), 1e-5)
    final_parity = verify_parity(predict_calibrated(calibrated.final_prediction, raw), onnx_probabilities(final_path, transform_features(raw, calibrated.final_prediction.transforms)), 1e-5)
    return ModelingRun(skeleton, calibrated, holdout_parity, final_parity)

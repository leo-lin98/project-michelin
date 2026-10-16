"""Developer checks for real quantization and full approved local orchestration."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
import pytest
from onnx import numpy_helper

from michelin.config import AppConfig
from michelin.export.parity import onnx_probabilities
from michelin.export.to_onnx import export_prediction
from michelin.features.transforms import transform_features
from michelin.model.dataset import ClassificationDataset, DatasetSplit
from michelin.model.train import CalibratedClassifier, fit_calibrated, predict_calibrated
import michelin.pipeline as pipeline
from michelin.pipeline import run_modeling
from test_model_acceptance import config, modeling_case


def test_leaf_pairs_reduce_actual_storage_precision_without_changing_fitted_model(
    tmp_path: Path, modeling_case: tuple[ClassificationDataset, DatasetSplit], config: AppConfig,
) -> None:
    dataset, split = modeling_case
    model = fit_calibrated(dataset, split.train_ids, config)
    before = model.estimator.estimators_[0, 0].tree_.value.copy()
    path = tmp_path / "leaves.onnx"
    export_prediction(model, path, 17)
    graph = onnx.load(path)
    initializers = {item.name: numpy_helper.to_array(item) for item in graph.graph.initializer}
    restored_values = []
    original_values = []
    for index, estimator in enumerate(model.estimator.estimators_[:, 0]):
        leaves = estimator.tree_.children_left == -1
        high = initializers[f"tree_{index}_high"]
        low = initializers[f"tree_{index}_low"]
        assert high.dtype == np.float16 and low.dtype == np.float16
        assert high.nbytes + low.nbytes == int(leaves.sum()) * 4
        restored_values.extend(high.astype(float) + low.astype(float))
        original_values.extend(estimator.tree_.value[leaves, 0, 0])
    assert np.any(np.asarray(restored_values) != original_values)
    np.testing.assert_array_equal(before, model.estimator.estimators_[0, 0].tree_.value)
    with pytest.raises(ValueError, match="opset"):
        export_prediction(model, tmp_path / "unsupported.onnx", 18)


def test_local_modeling_run_uses_distinct_holdout_and_final_exports(
    tmp_path: Path, modeling_case: tuple[ClassificationDataset, DatasetSplit], config: AppConfig,
) -> None:
    dataset, split = modeling_case
    result = run_modeling(dataset, split, config, tmp_path)
    assert set(result.calibrated.holdout_probabilities.index) == set(split.test_ids)
    assert result.holdout_parity.rows == len(split.test_ids)
    assert result.final_parity.rows == len(split.test_ids)
    assert (tmp_path / "holdout.quantized.onnx").is_file()
    assert (tmp_path / "final.quantized.onnx").is_file()
    assert set(result.calibrated.oof_probabilities.index).isdisjoint(dataset.display_features.index)


def test_constant_feature_model_preserves_batch_probability_shape(
    tmp_path: Path, modeling_case: tuple[ClassificationDataset, DatasetSplit], config: AppConfig,
) -> None:
    dataset, split = modeling_case
    raw = dataset.features.assign(average_rating=4.0, review_count=10.0, neighborhood_density=2.0, city="Taipei", cuisine="Chinese", price_range="200-400")
    constant = ClassificationDataset(raw, dataset.labels, dataset.hard_negative, dataset.display_features)
    model = fit_calibrated(constant, split.train_ids, config)
    path = tmp_path / "constant.onnx"
    export_prediction(model, path, 17)
    heldout = raw.loc[list(split.test_ids)]
    actual = onnx_probabilities(path, transform_features(heldout, model.transforms))
    np.testing.assert_allclose(actual, predict_calibrated(model, heldout), rtol=0, atol=1e-5)


@pytest.mark.parametrize("failure", ["prior", "initial_loss", "floor", "ceiling", "capacity", "signal"])
def test_failed_skeleton_gate_prevents_calibrated_training(
    tmp_path: Path, modeling_case: tuple[ClassificationDataset, DatasetSplit], config: AppConfig,
    monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    dataset, split = modeling_case
    skeleton = pipeline.train_skeleton(dataset, split, config)
    diagnostics = skeleton.diagnostics
    if failure == "prior":
        diagnostics = replace(diagnostics, initial_probability=0.5)
    elif failure == "initial_loss":
        diagnostics = replace(diagnostics, initial_loss=0.0)
    elif failure == "floor":
        diagnostics = replace(diagnostics, constant_probabilities=diagnostics.constant_probabilities + 0.1)
    elif failure == "ceiling":
        diagnostics = replace(diagnostics, overfit_labels_match=False)
    elif failure == "capacity":
        diagnostics = replace(diagnostics, increased_capacity_train_loss=diagnostics.base_train_loss)
    else:
        skeleton = replace(skeleton, test_probabilities=diagnostics.constant_probabilities)
    failed = replace(skeleton, diagnostics=diagnostics)
    monkeypatch.setattr(pipeline, "train_skeleton", lambda dataset, split, config: failed)
    def forbidden_training(dataset: ClassificationDataset, split: DatasetSplit, config: AppConfig) -> None:
        raise AssertionError("Phase 5 ran after failed skeleton gate")
    monkeypatch.setattr(pipeline, "train_calibrated", forbidden_training)
    with pytest.raises(ValueError, match="Phase 4"):
        run_modeling(dataset, split, config, tmp_path)


def test_failed_calibrated_ranking_gate_prevents_export(
    tmp_path: Path, modeling_case: tuple[ClassificationDataset, DatasetSplit], config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, split = modeling_case
    calibrated = pipeline.train_calibrated(dataset, split, config)
    failed = replace(calibrated, holdout_probabilities=pd.Series(0.2, index=calibrated.holdout_probabilities.index))
    monkeypatch.setattr(pipeline, "train_calibrated", lambda dataset, split, config: failed)
    def forbidden_export(model: CalibratedClassifier, path: Path, opset: int) -> None:
        raise AssertionError("Phase 6 ran after failed calibrated ranking gate")
    monkeypatch.setattr(pipeline, "export_prediction", forbidden_export)
    with pytest.raises(ValueError, match="Phase 5"):
        run_modeling(dataset, split, config, tmp_path)

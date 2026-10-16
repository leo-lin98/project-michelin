"""Independent Phase 6 acceptance: real quantization and strict probability parity."""

from pathlib import Path

import numpy as np
import onnx
import pandas as pd
import pytest
from onnx import TensorProto

from michelin.config import AppConfig
from michelin.export.parity import onnx_probabilities, verify_parity
from michelin.export.to_onnx import export_prediction
from michelin.features.transforms import transform_features
from michelin.model.dataset import ClassificationDataset, DatasetSplit
from michelin.model.train import fit_calibrated, predict_calibrated
from test_model_acceptance import config, modeling_case


def test_quantized_prediction_graph_is_valid_compact_and_matches_python_holdout(
    tmp_path: Path, modeling_case: tuple[ClassificationDataset, DatasetSplit], config: AppConfig,
) -> None:
    dataset, split = modeling_case
    model = fit_calibrated(dataset, split.train_ids, config)
    raw = dataset.features.loc[list(split.test_ids)]
    encoded = transform_features(raw, model.transforms)
    python_probabilities = predict_calibrated(model, raw)
    path = tmp_path / "prediction.quantized.onnx"
    exported = export_prediction(model, path, 17)
    assert exported.artifact_path == path
    assert exported.feature_names == model.transforms.feature_names
    assert exported.quantized is True
    assert exported.quantization_method
    assert path.is_file() and path.stat().st_size > 0
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    reduced = {value.name for value in graph.graph.initializer if value.data_type in {TensorProto.FLOAT16, TensorProto.INT8, TensorProto.UINT8}}
    for node in graph.graph.node:
        if node.op_type == "Constant":
            if any(attribute.type == onnx.AttributeProto.TENSOR and attribute.t.data_type in {TensorProto.FLOAT16, TensorProto.INT8, TensorProto.UINT8} for attribute in node.attribute):
                reduced.update(node.output)
    assert reduced, "Quantization requires actual reduced-precision data, not metadata or a renamed FP32 file"
    output_dependencies = {value.name for value in graph.graph.output}
    for node in reversed(graph.graph.node):
        if output_dependencies.intersection(node.output):
            output_dependencies.update(node.input)
    assert reduced.intersection(output_dependencies), "Reduced-precision values must contribute to exported predictions"
    actual = onnx_probabilities(path, encoded)
    assert actual.index.equals(raw.index)
    assert actual.between(0, 1).all()
    independent_error = float(np.max(np.abs(actual.to_numpy() - python_probabilities.to_numpy())))
    assert independent_error <= 1e-5
    report = verify_parity(python_probabilities, actual, 1e-5)
    assert report.rows == len(raw)
    assert report.max_absolute_error == pytest.approx(independent_error, abs=1e-15)


def test_final_full_refit_export_also_has_probability_parity(
    tmp_path: Path, modeling_case: tuple[ClassificationDataset, DatasetSplit], config: AppConfig,
) -> None:
    dataset, split = modeling_case
    model = fit_calibrated(dataset, tuple(dataset.features.index), config)
    raw = dataset.features.loc[list(split.test_ids)]
    path = tmp_path / "final.quantized.onnx"
    export_prediction(model, path, 17)
    expected = predict_calibrated(model, raw)
    actual = onnx_probabilities(path, transform_features(raw, model.transforms))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)


def test_quantized_export_preserves_float32_tree_threshold_boundary_predictions(
    tmp_path: Path, modeling_case: tuple[ClassificationDataset, DatasetSplit], config: AppConfig,
) -> None:
    dataset, split = modeling_case
    raw = dataset.features.assign(
        average_rating=[2.5 + i * 0.01 + int(label) * 0.5 for i, label in enumerate(dataset.labels)],
        review_count=[i * i + int(label) * 3 for i, label in enumerate(dataset.labels)],
        neighborhood_density=[i % 7 + 2 for i in range(len(dataset.labels))],
        price_range="200-400", city="Taipei",
    )
    varied = ClassificationDataset(raw, dataset.labels, dataset.hard_negative, dataset.display_features)
    model = fit_calibrated(varied, split.train_ids, config)
    encoded = transform_features(raw.loc[list(split.train_ids)], model.transforms)
    boundary_rows: list[np.ndarray] = []
    nonrepresentable_thresholds = 0
    for estimator in model.estimator.estimators_[:12, 0]:
        tree = estimator.tree_
        feature = int(tree.feature[0])
        assert feature >= 0, "Fixture must produce a nontrivial learned root split"
        threshold = float(tree.threshold[0])
        rounded = np.float32(threshold)
        nonrepresentable_thresholds += int(float(rounded) != threshold)
        neighbors = (
            np.nextafter(rounded, np.float32(-np.inf)),
            rounded,
            np.nextafter(rounded, np.float32(np.inf)),
        )
        assert float(neighbors[0]) < threshold < float(neighbors[-1])
        for value in (*neighbors, threshold):
            row = encoded.iloc[0].to_numpy(copy=True)
            row[feature] = float(value)
            boundary_rows.append(row)
    assert nonrepresentable_thresholds > 0, "Fixture must exercise stored thresholds outside float32 representability"
    boundaries = pd.DataFrame(
        boundary_rows, columns=encoded.columns,
        index=[f"boundary-{i:03d}" for i in range(len(boundary_rows))],
    )
    logits = model.estimator.decision_function(boundaries).reshape(-1, 1)
    expected = pd.Series(model.calibrator.predict_proba(logits)[:, 1], index=boundaries.index, name="probability")
    path = tmp_path / "boundaries.quantized.onnx"
    export_prediction(model, path, 17)
    actual = onnx_probabilities(path, boundaries)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)
    assert verify_parity(expected, actual, 1e-5).max_absolute_error <= 1e-5


def test_onnx_inference_enforces_feature_names_identities_and_finite_dimensions(
    tmp_path: Path, modeling_case: tuple[ClassificationDataset, DatasetSplit], config: AppConfig,
) -> None:
    dataset, split = modeling_case
    model = fit_calibrated(dataset, split.train_ids, config)
    path = tmp_path / "schema.quantized.onnx"
    export_prediction(model, path, 17)
    encoded = transform_features(dataset.features.loc[list(split.test_ids)], model.transforms)
    expected = onnx_probabilities(path, encoded)
    try:
        reversed_scores = onnx_probabilities(path, encoded.iloc[:, ::-1])
    except ValueError:
        pass
    else:
        pd.testing.assert_series_equal(reversed_scores, expected)
    nonfinite = encoded.copy(deep=True)
    nonfinite.iloc[0, 0] = float("nan")
    for invalid in (encoded.iloc[:, :-1], encoded.rename(columns={encoded.columns[0]: "unknown"}), nonfinite, pd.concat([encoded, encoded.iloc[:1]])):
        with pytest.raises(ValueError):
            onnx_probabilities(path, invalid)


def test_parity_gate_rejects_probability_error_above_declared_tolerance() -> None:
    expected = pd.Series([0.1, 0.9], index=["a", "b"])
    with pytest.raises(ValueError, match="parity|tolerance|error"):
        verify_parity(expected, pd.Series([0.10002, 0.9], index=expected.index), 1e-5)


def test_parity_gate_accepts_exact_predictions_and_reports_zero() -> None:
    expected = pd.Series([0.1, 0.9], index=["a", "b"])
    report = verify_parity(expected, expected.copy(deep=True), 1e-5)
    assert report.max_absolute_error == 0.0
    assert report.rows == 2


@pytest.mark.parametrize("scenario", ["missing_row", "different_identity", "nan", "outside_probability", "empty"])
def test_parity_gate_rejects_invalid_boundaries(scenario: str) -> None:
    expected = pd.Series([0.1, 0.9], index=["a", "b"])
    actual = expected.copy(deep=True)
    if scenario == "missing_row":
        actual = actual.iloc[:1]
    elif scenario == "different_identity":
        actual = actual.rename(index={"b": "other"})
    elif scenario == "nan":
        actual.loc["b"] = float("nan")
    elif scenario == "outside_probability":
        actual.loc["b"] = 1.1
    else:
        expected, actual = expected.iloc[:0], actual.iloc[:0]
    with pytest.raises(ValueError):
        verify_parity(expected, actual, 1e-5)


@pytest.mark.parametrize("tolerance", [-1.0, float("nan"), float("inf")])
def test_parity_gate_rejects_invalid_tolerance(tolerance: float) -> None:
    expected = pd.Series([0.1, 0.9], index=["a", "b"])
    with pytest.raises(ValueError):
        verify_parity(expected, expected, tolerance)

"""Run the bounded local phases 4–6 workflow twice and save modeling evidence."""

import argparse
from dataclasses import asdict
from hashlib import file_digest
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import pandas as pd

from michelin.config import AppConfig, load_config
from michelin.features.transforms import transform_features
from michelin.model.dataset import ClassificationDataset, assemble_classification_dataset, stratified_holdout
from michelin.model.evaluate import evaluate_calibration, evaluate_probabilities
from michelin.model.settings import load_modeling_settings
from michelin.model.train import CalibratedClassifier, explain_rows
from michelin.pipeline import ModelingRun, oof_base_rate, run_modeling


def compare_runs(first: ModelingRun, second: ModelingRun) -> None:
    """Require exactly reproducible scores, diagnostics, evaluations and calibration."""
    for left, right in ((first.skeleton.train_probabilities, second.skeleton.train_probabilities), (first.skeleton.test_probabilities, second.skeleton.test_probabilities), (first.calibrated.holdout_probabilities, second.calibrated.holdout_probabilities), (first.calibrated.oof_probabilities, second.calibrated.oof_probabilities)):
        pd.testing.assert_series_equal(left, right, check_exact=True)
    for left, right in ((first.calibrated.holdout_evaluation, second.calibrated.holdout_evaluation), (first.calibrated.oof_evaluation, second.calibrated.oof_evaluation), (first.calibrated.holdout_calibration, second.calibrated.holdout_calibration), (first.calibrated.oof_calibration, second.calibrated.oof_calibration), (first.holdout_parity, second.holdout_parity), (first.final_parity, second.final_parity)):
        if left != right:
            raise ValueError("Repeated modeling metrics or ONNX parity are not exactly reproducible")
    pd.testing.assert_frame_equal(first.skeleton.diagnostics.watchlist_probabilities, second.skeleton.diagnostics.watchlist_probabilities, check_exact=True)
    for left, right in zip(first.calibrated.diagnostics.stages, second.calibrated.diagnostics.stages, strict=True):
        pd.testing.assert_series_equal(left.probabilities, right.probabilities, check_exact=True)


def export_evidence(model: CalibratedClassifier, path: Path, raw: pd.DataFrame) -> dict[str, float | int | str]:
    """Measure actual leaf rounding, native warmed execution and artifact storage."""
    graph = onnx.load(path)
    metadata = {item.key: item.value for item in graph.metadata_props}
    tensors = {item.name: numpy_helper.to_array(item) for item in graph.graph.initializer}
    single_error = 0.0
    paired_error = 0.0
    leaf_count = 0
    changed_leaves = 0
    for index, fitted in enumerate(model.estimator.estimators_[:, 0]):
        values = fitted.tree_.value[fitted.tree_.children_left == -1, 0, 0]
        high = tensors[f"tree_{index}_high"].astype(float)
        low = tensors[f"tree_{index}_low"].astype(float)
        single_error += float(np.max(np.abs(values - high)))
        paired_error += float(np.max(np.abs(values - high - low)))
        leaf_count += len(values)
        changed_leaves += int(np.count_nonzero(values != high + low))
    factor = model.estimator.learning_rate * abs(float(model.calibrator.coef_[0, 0])) / 4
    encoded = transform_features(raw, model.transforms).to_numpy(dtype=np.float32)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    for _ in range(10):
        session.run(["probability"], {"features": encoded[:1]})
    start = perf_counter()
    for _ in range(100):
        session.run(["probability"], {"features": encoded[:1]})
    elapsed = (perf_counter() - start) * 1000 / 100
    single_graph = onnx.ModelProto()
    single_graph.CopyFrom(graph)
    for tensor in single_graph.graph.initializer:
        if tensor.name.endswith("_low"):
            replacement = numpy_helper.from_array(np.zeros_like(numpy_helper.to_array(tensor)), tensor.name)
            tensor.CopyFrom(replacement)
    single_session = ort.InferenceSession(single_graph.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    logits = model.estimator.decision_function(transform_features(raw, model.transforms)).reshape(-1, 1)
    expected = model.calibrator.predict_proba(logits)[:, 1]
    single_scores = single_session.run(["probability"], {"features": encoded})[0]
    return {
        "artifact_bytes": path.stat().st_size, "opset": 17, "dimensions": encoded.shape[1],
        "quantization_method": metadata["quantization_method"], "leaf_count": leaf_count,
        "leaf_storage_bytes": leaf_count * 4, "original_double_leaf_bytes": leaf_count * 8,
        "ordinary_float32_leaf_bytes": leaf_count * 4, "nonexact_quantized_leaves": changed_leaves,
        "single_fp16_probability_error_bound": single_error * factor,
        "single_fp16_holdout_max_error": float(np.max(np.abs(single_scores - expected))),
        "paired_fp16_probability_error_bound": paired_error * factor,
        "native_warm_single_row_mean_ms": elapsed,
        "timing_scope": "CPUExecutionProvider session.run only, 100 warmed single-row calls; excludes session creation and preprocessing; no browser claim",
    }


def save_evidence(run: ModelingRun, dataset: ClassificationDataset, config: AppConfig, output: Path) -> None:
    """Save finite validation reports, watchlist inputs, OOF and LR explanations locally."""
    calibrated = run.calibrated
    split = stratified_holdout(dataset, config.pipeline)
    raw = dataset.features.loc[list(split.test_ids)]
    reports = {}
    for name, scores, report, calibration in (("holdout", calibrated.holdout_probabilities, calibrated.holdout_evaluation, calibrated.holdout_calibration), ("oof", calibrated.oof_probabilities, calibrated.oof_evaluation, calibrated.oof_calibration)):
        labels = dataset.labels.loc[scores.index]
        hard = dataset.hard_negative.loc[scores.index]
        cities = dataset.features.loc[scores.index, "city"]
        prior = float(dataset.labels.loc[list(split.train_ids)].mean()) if name == "holdout" else float(labels.mean())
        baseline_scores = pd.Series(prior, index=scores.index) if name == "holdout" else oof_base_rate(dataset, calibrated)
        floor = evaluate_probabilities(labels, baseline_scores, hard, cities, calibrated.cutoff, max(1, int(labels.sum())))
        majority = evaluate_probabilities(labels, pd.Series(0.0, index=scores.index), hard, cities, calibrated.cutoff, max(1, int(labels.sum())))
        ap, floor_ap = report.metrics["average_precision"], floor.metrics["average_precision"]
        loss, floor_loss = report.metrics["log_loss"], floor.metrics["log_loss"]
        if ap is None or floor_ap is None or ap <= floor_ap:
            raise ValueError(f"{name} model must beat train-only base-rate AP gate")
        reports[name] = {"evaluation": asdict(report), "calibration": asdict(calibration), "base_rate": asdict(floor), "majority": asdict(majority), "base_rate_ap_gate_pass": True, "log_loss_below_base_rate": loss is not None and floor_loss is not None and loss < floor_loss}
        scores.to_csv(output / f"{name}_probabilities.csv")
    diagnostics = run.skeleton.diagnostics
    diagnostics.watchlist_raw.to_csv(output / "watchlist_raw.csv")
    diagnostics.watchlist_transformed.to_csv(output / "watchlist_transformed.csv")
    diagnostics.watchlist_probabilities.to_csv(output / "watchlist_dynamics.csv")
    diagnostics.constant_probabilities.to_csv(output / "constant_probabilities.csv")
    run.skeleton.test_probabilities.to_csv(output / "skeleton_holdout_probabilities.csv")
    skeleton_truth = dataset.labels.loc[run.skeleton.test_probabilities.index]
    skeleton_evaluation = evaluate_probabilities(skeleton_truth, run.skeleton.test_probabilities, dataset.hard_negative.loc[skeleton_truth.index], dataset.features.loc[skeleton_truth.index, "city"], calibrated.cutoff, max(1, int(skeleton_truth.sum())))
    skeleton_floor = evaluate_probabilities(skeleton_truth, diagnostics.constant_probabilities, dataset.hard_negative.loc[skeleton_truth.index], dataset.features.loc[skeleton_truth.index, "city"], calibrated.cutoff, max(1, int(skeleton_truth.sum())))
    explanation = explain_rows(calibrated.explanation, raw)
    explanation.global_importance.to_csv(output / "holdout_lr_coefficients.csv")
    explanation.contributions.to_csv(output / "holdout_lr_contributions.csv")
    (output / "holdout_top_five.json").write_text(json.dumps(explanation.top_five, indent=2, allow_nan=False) + "\n")
    (output / "oof_provenance.json").write_text(json.dumps([asdict(fold) for fold in calibrated.oof_folds], indent=2, allow_nan=False) + "\n")
    uncalibrated = pd.Series(calibrated.prediction.estimator.predict_proba(transform_features(raw, calibrated.prediction.transforms))[:, 1], index=raw.index)
    settings = load_modeling_settings(Path("config/modeling.yaml"))
    stages = []
    for index, stage in enumerate(calibrated.diagnostics.stages, start=1):
        stages.append({"features": stage.features, "average_precision": stage.average_precision, "fold_average_precision": stage.fold_average_precision})
        stage.probabilities.to_csv(output / f"stage_{index}_training_oof.csv")
    result = {
        "exact_repeat": True, "display_excluded": True, "seed": config.pipeline.seed,
        "rows": {"modeling": len(dataset.features), "holdout": len(raw), "display": len(dataset.display_features)},
        "cutoff": calibrated.cutoff, **reports,
        "uncalibrated_holdout_calibration": asdict(evaluate_calibration(dataset.labels.loc[raw.index], uncalibrated, settings.calibration_bins)),
        "skeleton": {"evaluation": asdict(skeleton_evaluation), "floor_evaluation": asdict(skeleton_floor), "ordered_gates_pass": True, "initial_probability": diagnostics.initial_probability, "initial_loss": diagnostics.initial_loss, "overfit_loss": diagnostics.overfit_loss, "overfit_labels_match": diagnostics.overfit_labels_match, "base_train_loss": diagnostics.base_train_loss, "increased_capacity_train_loss": diagnostics.increased_capacity_train_loss},
        "training_only_stages": stages, "calibration_rationale": calibrated.diagnostics.rationale,
        "schedule": calibrated.diagnostics.schedule, "explanation_caveat": explanation.caveat,
        "holdout_export": {**asdict(run.holdout_parity), **export_evidence(calibrated.prediction, output / "holdout.quantized.onnx", raw)},
        "final_export": {**asdict(run.final_parity), **export_evidence(calibrated.final_prediction, output / "final.quantized.onnx", raw)},
        "limitations": "Final full-refit parity is numerical verification only. Nine holdout positives, 43 modeling positives, sampled ordinary negatives and no New Taipei positives limit performance/calibration conclusions. Fixed-cutoff recall is poor; calibration does not guarantee improvement. Browser latency remains untested.",
    }
    (output / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main() -> None:
    """Read explicit bounded snapshots; create two independent local run destinations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--pipeline-config", type=Path, required=True)
    parser.add_argument("--features-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--row-limit", type=int, required=True)
    args = parser.parse_args()
    if args.row_limit < 1:
        raise ValueError("row-limit must be positive")
    output = args.output_dir.resolve()
    allowed = Path("data/processed/modeling-phase6").resolve()
    if not output.is_relative_to(allowed):
        raise ValueError("Modeling evidence must remain under data/processed/modeling-phase6")
    config = load_config(args.pipeline_config, args.features_config)
    panel = pd.read_csv(args.panel, nrows=args.row_limit + 1, float_precision="round_trip")
    features = pd.read_csv(args.features, index_col="restaurant_id", nrows=args.row_limit + 1, float_precision="round_trip")
    if len(panel) > args.row_limit or len(features) > args.row_limit:
        raise ValueError("Input snapshots exceed explicit bounded row-limit")
    dataset = assemble_classification_dataset(panel, features, config)
    split = stratified_holdout(dataset, config.pipeline)
    output.mkdir(parents=True, exist_ok=False)
    first = run_modeling(dataset, split, config, output / "run1")
    second = run_modeling(dataset, split, config, output / "run2")
    compare_runs(first, second)
    for name in ("holdout.quantized.onnx", "final.quantized.onnx"):
        if (output / "run1" / name).read_bytes() != (output / "run2" / name).read_bytes():
            raise ValueError("Repeated ONNX artifacts differ")
    save_evidence(first, dataset, config, output / "run1")
    save_evidence(second, dataset, config, output / "run2")
    paths = (args.panel, args.features, args.pipeline_config, args.features_config, Path("config/modeling.yaml"), Path("pyproject.toml"), Path("uv.lock"), Path("src/michelin/model/train.py"), Path("src/michelin/export/to_onnx.py"), Path("src/michelin/export/parity.py"), Path("src/michelin/pipeline.py"), Path(__file__))
    hashes = {}
    for path in paths:
        with path.open("rb") as source:
            hashes[str(path)] = file_digest(source, "sha256").hexdigest()
    (output / "hashes.json").write_text(json.dumps(hashes, indent=2, allow_nan=False) + "\n")
    print(f"Two exact modeling runs passed; local validation: {output / 'run1/validation.json'}")


if __name__ == "__main__":
    main()

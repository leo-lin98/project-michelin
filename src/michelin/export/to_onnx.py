"""Export sklearn calibrated binary GBM with paired FP16 leaf storage."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from sklearn.dummy import DummyClassifier

from michelin.model.train import CalibratedClassifier


@dataclass(frozen=True)
class ExportResult:
    artifact_path: Path
    quantized: bool
    feature_names: tuple[str, ...]
    quantization_method: str


def export_prediction(model: CalibratedClassifier, path: Path, opset: int) -> ExportResult:
    """Store each learned leaf as FP16 high+residual; bound probability rounding error.

    sklearn casts inputs to float32 then compares against double thresholds.
    Standard Gather/LessOrEqual/Where operations preserve that behavior exactly.
    Leaf pairs use four bytes instead of the original eight; accumulation and
    calibration remain double precision. No training transforms enter this graph.
    """
    if not isinstance(model, CalibratedClassifier) or opset != 17:
        raise ValueError("Export requires a fitted CalibratedClassifier and opset 17")
    estimator = model.estimator
    if not isinstance(estimator.init_, DummyClassifier) or not np.array_equal(estimator.classes_, [0, 1]) or estimator.estimators_.shape[1] != 1:
        raise ValueError("Export supports binary GBM with fitted class-prior initialization")
    names = model.transforms.feature_names
    if len(names) != estimator.n_features_in_ or tuple(estimator.feature_names_in_) != names:
        raise ValueError("GBM feature order must match frozen transformed features")
    if not np.array_equal(model.calibrator.classes_, [0, 1]) or model.calibrator.coef_.shape != (1, 1):
        raise ValueError("Export requires a binary one-logit Platt calibrator")
    initial = float(np.log(estimator.init_.class_prior_[1] / estimator.init_.class_prior_[0]))
    slope = float(model.calibrator.coef_[0, 0])
    intercept = float(model.calibrator.intercept_[0])
    initializers: list[onnx.TensorProto] = []
    nodes: list[onnx.NodeProto] = [helper.make_node("Cast", ["features"], ["features_double"], to=TensorProto.DOUBLE)]
    roots: list[str] = []
    error_bound = 0.0
    for tree_index, fitted in enumerate(estimator.estimators_[:, 0]):
        tree = fitted.tree_
        prefix = f"tree_{tree_index}"
        leaves = np.flatnonzero(tree.children_left == -1)
        values = tree.value[leaves, 0, 0]
        high = values.astype(np.float16)
        low = (values - high.astype(np.float64)).astype(np.float16)
        restored = high.astype(np.float64) + low.astype(np.float64)
        if not np.isfinite(restored).all():
            raise ValueError("FP16 leaf pairs cannot represent fitted GBM leaf magnitudes")
        error_bound += float(np.max(np.abs(values - restored)))
        for suffix, array in (("high", high), ("low", low)):
            initializers.append(numpy_helper.from_array(array, f"{prefix}_{suffix}"))
            nodes.append(helper.make_node("Cast", [f"{prefix}_{suffix}"], [f"{prefix}_{suffix}_double"], to=TensorProto.DOUBLE))
        nodes.append(helper.make_node("Add", [f"{prefix}_high_double", f"{prefix}_low_double"], [f"{prefix}_leaves"]))
        leaf_positions = {int(node): i for i, node in enumerate(leaves)}
        for node in reversed(range(tree.node_count)):
            output = f"{prefix}_node_{node}"
            if node in leaf_positions:
                index = f"{output}_index"
                initializers.append(numpy_helper.from_array(np.asarray(leaf_positions[node], dtype=np.int64), index))
                nodes.append(helper.make_node("Gather", [f"{prefix}_leaves", index], [output], axis=0))
            else:
                feature, threshold = f"{output}_feature", f"{output}_threshold"
                initializers.extend((numpy_helper.from_array(np.asarray(tree.feature[node], dtype=np.int64), feature), numpy_helper.from_array(np.asarray(tree.threshold[node], dtype=np.float64), threshold)))
                nodes.extend((helper.make_node("Gather", ["features_double", feature], [f"{output}_input"], axis=1), helper.make_node("LessOrEqual", [f"{output}_input", threshold], [f"{output}_left"]), helper.make_node("Where", [f"{output}_left", f"{prefix}_node_{tree.children_left[node]}", f"{prefix}_node_{tree.children_right[node]}"], [output])))
        roots.append(f"{prefix}_node_0")
    probability_bound = error_bound * estimator.learning_rate * abs(slope) / 4
    if not np.isfinite(probability_bound) or probability_bound > 1e-5:
        raise ValueError(f"Quantized leaf probability error bound {probability_bound} exceeds 1e-5")
    initializers.extend(numpy_helper.from_array(np.asarray(value, dtype=np.float64), name) for name, value in (("initial", initial), ("learning_rate", estimator.learning_rate), ("platt_slope", slope), ("platt_intercept", intercept)))
    nodes.extend((helper.make_node("Sum", roots, ["tree_sum"]), helper.make_node("Mul", ["tree_sum", "learning_rate"], ["boosting_delta"]), helper.make_node("Add", ["initial", "boosting_delta"], ["decision_logit"]), helper.make_node("Mul", ["decision_logit", "platt_slope"], ["scaled_logit"]), helper.make_node("Add", ["scaled_logit", "platt_intercept"], ["calibrated_logit"]), helper.make_node("Sigmoid", ["calibrated_logit"], ["probability"])))
    initializers.append(numpy_helper.from_array(np.asarray([0], dtype=np.int64), "batch_axis"))
    nodes.insert(-1, helper.make_node("Shape", ["features"], ["input_shape"]))
    nodes.insert(-1, helper.make_node("Gather", ["input_shape", "batch_axis"], ["batch_shape"], axis=0))
    nodes.insert(-1, helper.make_node("Expand", ["calibrated_logit", "batch_shape"], ["batch_logits"]))
    nodes[-1].input[0] = "batch_logits"
    graph = helper.make_graph(nodes, "calibrated_gbm", [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, len(names)])], [helper.make_tensor_value_info("probability", TensorProto.DOUBLE, [None])], initializers)
    exported = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10)
    method = "paired_fp16_leaves_double_thresholds_accumulation"
    helper.set_model_props(exported, {"feature_names": json.dumps(names), "quantization_method": method, "probability_error_bound": str(probability_bound)})
    onnx.checker.check_model(exported)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(exported, path)
    return ExportResult(path, True, names, method)

"""Strict identity-aware numerical verification of calibrated ONNX probabilities."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from michelin.model.evaluate import validate_identities


@dataclass(frozen=True)
class ParityReport:
    max_absolute_error: float
    rows: int


def validate_probabilities(values: pd.Series) -> None:
    """Reject empty, ambiguous or invalid class-one probabilities."""
    validate_identities(values, "Parity probabilities")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, int, np.floating, np.integer)) for value in values):
        raise ValueError("Parity probabilities must be numeric values")
    numeric = values.to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or ((numeric < 0) | (numeric > 1)).any():
        raise ValueError("Parity probabilities must be finite and in [0, 1]")


def verify_parity(python_probabilities: pd.Series, onnx_probabilities: pd.Series, tolerance: float) -> ParityReport:
    """Align identical identity sets and enforce a finite absolute error tolerance."""
    if isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, (float, int, np.floating, np.integer)) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Parity tolerance must be finite and nonnegative")
    validate_probabilities(python_probabilities)
    validate_probabilities(onnx_probabilities)
    if set(python_probabilities.index) != set(onnx_probabilities.index):
        raise ValueError("Parity probability identities must match")
    error = float(np.max(np.abs(python_probabilities.to_numpy() - onnx_probabilities.reindex(python_probabilities.index).to_numpy())))
    if error > tolerance:
        raise ValueError(f"ONNX parity error {error} exceeds tolerance {tolerance}")
    return ParityReport(error, len(python_probabilities))


def onnx_probabilities(path: Path, transformed_rows: pd.DataFrame) -> pd.Series:
    """Run numeric encoded rows with exact feature order and preserved identities."""
    validate_identities(transformed_rows, "ONNX rows")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    names = json.loads(session.get_modelmeta().custom_metadata_map.get("feature_names", "null"))
    if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise ValueError("ONNX feature_names metadata must be a nonempty unique string list")
    if list(transformed_rows.columns) != names:
        raise ValueError("ONNX rows must match exact encoded feature names and order")
    if any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_complex_dtype(dtype) for dtype in transformed_rows.dtypes):
        raise ValueError("ONNX rows must contain numeric real-valued feature columns")
    matrix = transformed_rows.to_numpy(dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(names) or not np.isfinite(matrix).all():
        raise ValueError("ONNX rows must have finite float32 numeric features and correct dimensions")
    outputs = session.run(["probability"], {"features": matrix})
    probabilities = np.asarray(outputs[0])
    if probabilities.shape != (len(transformed_rows),):
        raise ValueError("ONNX probability output must have one value per input row")
    result = pd.Series(probabilities, index=transformed_rows.index.copy(), name="probability")
    validate_probabilities(result)
    return result

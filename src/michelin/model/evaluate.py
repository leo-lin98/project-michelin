"""Identity-aligned full-population classification metrics and reporting slices."""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, log_loss, precision_score, recall_score, roc_auc_score


@dataclass(frozen=True)
class EvaluationReport:
    metrics: dict[str, float | None]
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    predicted_positive_count: int
    known_positive_count: int
    hard_negative: dict[str, int | float | None]
    cities: dict[str, dict[str, float | str | None]]


def validate_identities(values: pd.Series | pd.DataFrame, context: str) -> None:
    """Require unique nonblank string restaurant identities."""
    if values.empty or not values.index.is_unique:
        raise ValueError(f"{context} identities must be nonempty and unique")
    if any(not isinstance(value, str) or not value.strip() for value in values.index):
        raise ValueError(f"{context} identities must be nonblank strings")


def population_metrics(labels: pd.Series, scores: pd.Series, cutoff: float, precision_k: int) -> dict[str, float | None]:
    """Compute unsmoothed metrics over every supplied row; rank ties by identity."""
    predicted = scores.ge(cutoff)
    ranked = scores.sort_index().sort_values(ascending=False, kind="stable").head(precision_k).index
    return {
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
        "average_precision": float(average_precision_score(labels, scores)) if labels.eq(1).any() else None,
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "precision_at_k": float(labels.loc[ranked].mean()),
        "roc_auc": float(roc_auc_score(labels, scores)) if labels.nunique() == 2 else None,
    }


def evaluate_probabilities(
    labels: pd.Series, probabilities: pd.Series, hard_negative: pd.Series,
    cities: pd.Series, cutoff: float, precision_k: int,
) -> EvaluationReport:
    """Validate and align identities before full holdout and city-slice evaluation."""
    if not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("Evaluation cutoff must be finite and in [0, 1]")
    if isinstance(precision_k, bool) or not isinstance(precision_k, int) or precision_k < 1:
        raise ValueError("precision_k must be a positive integer")
    for name, values in (("Labels", labels), ("Probabilities", probabilities), ("Hard negatives", hard_negative), ("Cities", cities)):
        validate_identities(values, name)
        if set(values.index) != set(labels.index):
            raise ValueError(f"{name} identities must match label identities")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Evaluation labels must be non-null binary values")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, int, np.floating, np.integer)) for value in probabilities):
        raise ValueError("Probabilities must be numeric values")
    if not np.isfinite(probabilities.to_numpy(dtype=float)).all() or not probabilities.between(0, 1).all():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if any(not isinstance(value, (bool, np.bool_)) for value in hard_negative):
        raise ValueError("Hard-negative flags must be non-null booleans")
    if any(not isinstance(value, str) or not value.strip() for value in cities):
        raise ValueError("Cities must be nonblank strings")
    scores, hard, regions = probabilities.loc[labels.index], hard_negative.loc[labels.index], cities.loc[labels.index]
    if labels.loc[hard].ne(0).any():
        raise ValueError("Hard-negative rows must have class 0 labels")
    predicted = scores.ge(cutoff)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    city_reports: dict[str, dict[str, float | str | None]] = {}
    for city in sorted(regions.unique()):
        identities = regions.index[regions.eq(city)]
        metrics = population_metrics(labels.loc[identities], scores.loc[identities], cutoff, precision_k)
        city_reports[city] = {**metrics, "undefined_reason": "ROC AUC requires both classes; average precision also requires positive labels" if labels.loc[identities].nunique() < 2 else None}
    return EvaluationReport(
        metrics=population_metrics(labels, scores, cutoff, precision_k),
        confusion_matrix=((int(matrix[0, 0]), int(matrix[0, 1])), (int(matrix[1, 0]), int(matrix[1, 1]))),
        predicted_positive_count=int(predicted.sum()), known_positive_count=int(labels.sum()),
        hard_negative={"count": int(hard.sum()), "error_rate": float(predicted.loc[hard].mean()) if hard.any() else None},
        cities=city_reports,
    )


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_probability: float | None
    positive_fraction: float | None


@dataclass(frozen=True)
class CalibrationReport:
    brier_score: float
    bins: tuple[ReliabilityBin, ...]


def evaluate_calibration(labels: pd.Series, probabilities: pd.Series, bin_count: int) -> CalibrationReport:
    """Report equal-width reliability bins including explicit empty bins and Brier loss."""
    validate_identities(labels, "Calibration labels")
    validate_identities(probabilities, "Calibration probabilities")
    if set(labels.index) != set(probabilities.index):
        raise ValueError("Calibration probabilities and label identities must match")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Calibration labels must be non-null binary values")
    if isinstance(bin_count, bool) or not isinstance(bin_count, int) or bin_count < 2:
        raise ValueError("Calibration bin_count must be an integer of at least 2")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, int, np.floating, np.integer)) for value in probabilities):
        raise ValueError("Calibration probabilities must be numeric values")
    scores = probabilities.loc[labels.index]
    if not np.isfinite(scores.to_numpy(dtype=float)).all() or not scores.between(0, 1).all():
        raise ValueError("Calibration probabilities must be finite and in [0, 1]")
    assignments = np.minimum((scores.to_numpy() * bin_count).astype(int), bin_count - 1)
    bins = tuple(
        ReliabilityBin(
            step / bin_count, (step + 1) / bin_count, int((assignments == step).sum()),
            float(scores.iloc[assignments == step].mean()) if (assignments == step).any() else None,
            float(labels.iloc[assignments == step].mean()) if (assignments == step).any() else None,
        ) for step in range(bin_count)
    )
    return CalibrationReport(float(((labels - scores) ** 2).mean()), bins)

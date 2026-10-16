"""Developer explanation and threshold boundary requirements."""

import pandas as pd
import pytest

from michelin.model.evaluate import evaluate_calibration
from michelin.model.thresholds import similar_to_starred


@pytest.mark.parametrize("values", [[float("nan")], [-0.01], [1.01], [True]])
def test_invalid_probabilities_rejected(values: list[float | bool]) -> None:
    with pytest.raises(ValueError):
        similar_to_starred(pd.Series(values, index=["r"]), 0.5)


def test_duplicate_identities_rejected() -> None:
    with pytest.raises(ValueError):
        similar_to_starred(pd.Series([0.4, 0.5], index=["r", "r"]), 0.5)


def test_brier_reports_probability_squared_error() -> None:
    labels = pd.Series([0, 1], index=["a", "b"])
    probabilities = pd.Series([0.2, 0.7], index=labels.index)
    report = evaluate_calibration(labels, probabilities, 10)
    assert sum(item.count for item in report.bins) == len(labels)
    assert report.brier_score == pytest.approx((0.2 ** 2 + 0.3 ** 2) / 2)


def test_calibration_reliability_endpoints_and_empty_bins() -> None:
    labels = pd.Series([0, 1], index=["a", "b"])
    report = evaluate_calibration(labels, pd.Series([0.0, 1.0], index=labels.index), 10)
    assert report.brier_score == 0
    assert report.bins[0].count == report.bins[-1].count == 1
    assert report.bins[1].mean_probability is None
    assert report.bins[1].positive_fraction is None

"""Developer evaluation checks derived from identity and threshold requirements."""

import pandas as pd
import pytest

from michelin.model.evaluate import evaluate_probabilities


def test_reordered_metadata_aligns_and_threshold_is_inclusive() -> None:
    labels = pd.Series([0, 1, 0], index=["c", "a", "b"])
    scores = pd.Series([0.5, 0.1, 0.2], index=["a", "b", "c"])
    hard = pd.Series([False, True, True], index=["a", "b", "c"])
    cities = pd.Series(["Taipei"] * 3, index=["b", "c", "a"])
    result = evaluate_probabilities(labels, scores, hard, cities, 0.5, 10)
    assert result.predicted_positive_count == 1
    assert result.metrics["recall"] == 1.0
    assert result.metrics["precision_at_k"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("cutoff,k", [(float("nan"), 1), (-1, 1), (1.1, 1), (0.5, 0)])
def test_invalid_evaluation_controls_rejected(cutoff: float, k: int) -> None:
    labels = pd.Series([0, 1], index=["a", "b"])
    with pytest.raises(ValueError):
        evaluate_probabilities(labels, labels.astype(float), labels.eq(0), pd.Series(["Taipei"] * 2, index=labels.index), cutoff, k)

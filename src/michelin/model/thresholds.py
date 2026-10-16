"""Fixed inclusive Similar-to-Starred probability threshold."""

import math

import numpy as np
import pandas as pd

from michelin.model.evaluate import validate_identities


def similar_to_starred(probabilities: pd.Series, cutoff: float) -> pd.Series:
    """Reject invalid inputs and flag probabilities at or above the explicit cutoff."""
    validate_identities(probabilities, "Threshold probabilities")
    if not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("Similar-to-Starred cutoff must be finite and in [0, 1]")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, int, np.floating, np.integer)) for value in probabilities):
        raise ValueError("Threshold probabilities must be numeric values")
    if not np.isfinite(probabilities.to_numpy(dtype=float)).all() or not probabilities.between(0, 1).all():
        raise ValueError("Threshold probabilities must be finite and in [0, 1]")
    return probabilities.ge(cutoff)

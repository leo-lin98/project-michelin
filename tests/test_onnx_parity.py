"""Developer checks for identity-safe probability parity."""

import pandas as pd
import pytest

from michelin.export.parity import verify_parity


def test_parity_realigns_identities_without_mutation() -> None:
    expected = pd.Series([0.2, 0.8], index=["a", "b"])
    actual = expected.iloc[::-1].copy()
    original = actual.copy()
    assert verify_parity(expected, actual, 1e-5).max_absolute_error == 0
    pd.testing.assert_series_equal(actual, original)


def test_parity_rejects_duplicate_ids() -> None:
    invalid = pd.Series([0.2, 0.8], index=["a", "a"])
    with pytest.raises(ValueError):
        verify_parity(invalid, invalid, 1e-5)


@pytest.mark.parametrize("values", [["0.2", "0.8"], [True, False]])
def test_parity_rejects_coercible_non_numeric_probabilities(values: list[str] | list[bool]) -> None:
    expected = pd.Series([0.2, 0.8], index=["a", "b"])
    with pytest.raises(ValueError, match="numeric"):
        verify_parity(expected, pd.Series(values, index=expected.index), 1e-5)

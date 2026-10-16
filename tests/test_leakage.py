"""Spec-derived leakage, provenance, descriptive EDA and statistical-gate checks."""

from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from michelin.config import AppConfig, load_config
from michelin.features.build import build_shared_features
from michelin.features.database import verify_panel_source_features
from michelin.features.diagnostics import build_report, missingness_tests


FEATURES = ("average_rating", "review_count", "neighborhood_density", "price_range", "cuisine", "city")


@pytest.fixture
def config() -> AppConfig:
    return load_config(Path("config/pipeline.yaml"), Path("config/features.yaml"))


@pytest.fixture
def panel() -> pd.DataFrame:
    rows = []
    for index in range(17):
        starred = index < 8
        display = index == 16
        hard = 8 <= index < 12
        rows.append({
            "restaurant_id": f"google_places:p{index}", "source_id": f"p{index}",
            "city": "Taipei" if starred else "New Taipei", "latitude": 25.0, "longitude": 121.5,
            "price_range": "2000+", "price_level": 2.0, "average_rating": 4.0, "review_count": 100.0,
            "cuisine": "restaurant", "feature_provider": "google_places",
            "feature_snapshot_date": "2026-09-01T00:00:00", "class": None if display else int(starred),
            "award": "1 Star" if starred else "Bib Gourmand" if hard else "none",
            "is_hard_negative": hard, "group": "out_of_sample" if display else "in_sample",
            "Description": "Guide-only text" if starred or hard else None,
            "FacilitiesAndServices": "Guide-only facilities" if starred else None,
        })
    return pd.DataFrame(rows)


def density_for(panel: pd.DataFrame) -> pd.Series:
    return pd.Series(2, index=pd.Index(panel["restaurant_id"], name="restaurant_id"))


def test_shared_matrix_is_exactly_six_features_and_metadata_cannot_change_it(
    panel: pd.DataFrame, config: AppConfig,
) -> None:
    panel = panel.assign(price_range=panel["price_range"].mask(panel.index == 0))
    before = panel.copy(deep=True)
    density = density_for(panel)
    result = build_shared_features(panel, density, config)
    assert tuple(result.columns) == FEATURES
    assert pd.isna(result.iloc[0]["price_range"])
    for group in (panel["class"].eq(1), panel["class"].eq(0), panel["class"].isna()):
        assert tuple(result.loc[panel.loc[group, "restaurant_id"]].columns) == FEATURES
    changed = panel.assign(**{
        "price_level": 9999, "class": 999, "award": "arbitrary metadata", "group": "changed",
        "is_hard_negative": False, "Description": "changed description",
        "FacilitiesAndServices": "changed facilities", "Award": "3 Stars",
    })
    pd.testing.assert_frame_equal(build_shared_features(changed, density, config), result)
    pd.testing.assert_frame_equal(panel, before)


@pytest.mark.parametrize("position", [0, 8, 12, 16])
def test_provenance_mismatch_rejected_in_every_population(
    panel: pd.DataFrame, config: AppConfig, position: int,
) -> None:
    altered = panel.assign(feature_provider=panel["feature_provider"].mask(panel.index == position, "different_provider"))
    with pytest.raises(ValueError, match="provenance"):
        build_shared_features(altered, density_for(altered), config)


@pytest.mark.parametrize("field,value", [
    ("average_rating", 6.0), ("review_count", -1.0), ("price_range", "  "),
    ("latitude", 91.0), ("longitude", np.nan), ("city", "Taoyuan"),
    ("restaurant_id", "wrong_identity"), ("feature_snapshot_date", "not-a-date"),
])
def test_raw_feature_validation_rejects_malformed_input(
    panel: pd.DataFrame, config: AppConfig, field: str, value: str | float,
) -> None:
    invalid = panel.assign(**{field: panel[field].mask(panel.index == 0, value)})
    with pytest.raises(ValueError):
        build_shared_features(invalid, density_for(invalid), config)


@pytest.mark.parametrize("field", ["price_range", "average_rating", "review_count", "cuisine", "city", "feature_snapshot_date"])
@pytest.mark.parametrize("position", [0, 8, 12, 16])
def test_panel_feature_tampering_is_detected_against_frozen_provider(panel: pd.DataFrame, field: str, position: int) -> None:
    reference = panel.iloc[[position]].copy(deep=True)
    tampered = panel.assign(**{field: panel[field].mask(panel.index == position)})
    with pytest.raises(ValueError, match=field):
        verify_panel_source_features(tampered.set_index("source_id"), reference)


def test_fisher_two_sided_four_field_holm_gate_ignores_display_rows(
    panel: pd.DataFrame, config: AppConfig,
) -> None:
    """Hypergeometric enumeration independently supplies the balanced-table Fisher oracle."""
    counts = (8, 7, 6, 0)
    fields = ("price_range", "average_rating", "review_count", "cuisine")
    test_panel = panel.copy(deep=True)
    for field, missing in zip(fields, counts, strict=True):
        test_panel[field] = test_panel[field].mask(
            (test_panel.index < missing) | ((test_panel.index >= 8) & (test_panel.index < 16 - missing))
            if missing else pd.Series(False, index=test_panel.index)
        )
    observed = missingness_tests(test_panel, config)
    assert tuple(test.feature for test in observed) == fields
    probabilities = [comb(8, k) ** 2 / comb(16, 8) for k in range(9)]
    expected_p = [sum(p for p in probabilities if p <= probabilities[k] + 1e-15) for k in counts[:3]] + [1.0]
    expected_adjusted = [4 * expected_p[0], 3 * expected_p[1], min(1, 2 * expected_p[2]), 1.0]
    for index, test in enumerate(observed):
        missing = counts[index]
        assert (test.starred_missing, test.starred_present) == (missing, 8 - missing)
        assert (test.not_starred_missing, test.not_starred_present) == ((8 - missing, missing) if missing else (0, 8))
        assert test.starred_missing_rate == missing / 8
        assert test.not_starred_missing_rate == ((8 - missing) / 8 if missing else 0)
        assert test.raw_p_value == pytest.approx(expected_p[index])
        assert test.adjusted_p_value == pytest.approx(expected_adjusted[index])
        assert test.significant == (expected_adjusted[index] <= 0.05)
    display = test_panel.iloc[[-1]].copy(deep=True)
    displays = pd.concat([display.assign(restaurant_id=f"google_places:display{i}", source_id=f"display{i}", **{field: None for field in fields}) for i in range(50)], ignore_index=True)
    expanded = pd.concat([test_panel.iloc[:-1], displays], ignore_index=True)
    assert missingness_tests(expanded, config) == observed
    features = build_shared_features(test_panel, density_for(test_panel), config)
    assert build_report(test_panel, features, config).validation_status == "blocked_missingness_leakage"


def test_eda_keeps_population_and_actual_city_counts_and_caveats(panel: pd.DataFrame, config: AppConfig) -> None:
    panel = panel.assign(price_range=panel["price_range"].mask(panel.index.isin([0, 8])).mask(panel.index.isin([1, 9]), "200-400"))
    features = build_shared_features(panel, density_for(panel), config)
    report = build_report(panel, features, config)
    assert report.validation_status == "passed"
    slices = {(summary.population, summary.city): summary for summary in report.slices}
    assert slices["starred", "Taipei"].rows == 8
    assert slices["starred", "New Taipei"].rows == 0
    assert slices["ordinary", "New Taipei"].rows == 4
    assert slices["bib_selected", "New Taipei"].rows == 4
    assert slices["display_ordinary", "New Taipei"].rows == 1
    assert slices["starred", "all"].numeric["average_rating"].mean == 4
    assert slices["ordinary", "all"].categorical_counts["city"] == {"New Taipei": 4}
    assert slices["starred", "all"].categorical_counts["price_range"] == {"200-400": 1, "2000+": 6}
    assert slices["bib_selected", "all"].categorical_counts["price_range"] == {"200-400": 1, "2000+": 2}
    assert slices["starred", "all"].missingness["price_range"] == 1
    assert "price_range" not in slices["starred", "all"].numeric
    assert "price_level" not in slices["starred", "all"].missingness
    caveats = " ".join(report.caveats).lower()
    assert "correlation" in caveats and "not causation" in caveats
    assert "municipality" in caveats and "proxy" in caveats
    assert "nonsignificant" in caveats and "do not prove independence" in caveats


@pytest.mark.parametrize("position", [0, 8, 12, 16])
def test_panel_cannot_invent_price_range_where_frozen_provider_has_null(
    panel: pd.DataFrame, position: int,
) -> None:
    reference = panel.iloc[[position]].assign(price_range=None)
    tampered = panel.assign(price_range=panel["price_range"].mask(panel.index == position, "200-400"))
    with pytest.raises(ValueError, match="price_range"):
        verify_panel_source_features(tampered.set_index("source_id"), reference)

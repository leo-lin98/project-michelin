from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from michelin.data.panel import PanelBuildConfig, build_labeled_panel, write_panel_outputs


def panel_config(tmp_path: Path) -> PanelBuildConfig:
    return PanelBuildConfig(
        cities=("Taipei", "New Taipei"),
        in_sample_group="in_sample",
        out_of_sample_group="out_of_sample",
        ordinary_in_sample_limit=2,
        ordinary_out_of_sample_limit=1,
        seed=42,
        feature_provider="google_places",
        processed_dir=tmp_path / "processed",
        interim_dir=tmp_path / "interim",
        expected_starred_count=2,
        expected_hard_negative_count=2,
    )


def guide_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": ["guide-1", "guide-2", "guide-3", "guide-4"],
            "award": ["1 Star", "3 Stars", "Bib Gourmand", "Selected Restaurants"],
            "name": ["Guide name"] * 4,
            "price_level": ["$$$"] * 4,
            "cuisine": ["Michelin cuisine"] * 4,
        }
    )


def feature_fixture() -> pd.DataFrame:
    ids = ["guide-1", "guide-2", "guide-3", "guide-4", "place-A", "place-a", "place-c", "outside", "bad"]
    return pd.DataFrame(
        {
            "source_id": ids,
            "name": [f"Google {place_id}" for place_id in ids],
            "address": ["1 Road"] * 9,
            "city": ["Taipei", "Taipei", "New Taipei", "New Taipei", "Taipei", "New Taipei", "Taipei", "Taoyuan", "Taipei"],
            "latitude": [25.0] * 8 + [float("inf")],
            "longitude": [121.5] * 9,
            "price_level": pd.array([2, 3, 1, 2, None, 1, 2, 1, 1], dtype="Int64"),
            "cuisine": ["chinese_restaurant"] * 9,
            "average_rating": [4.5] * 9,
            "review_count": pd.array([100] * 9, dtype="Int64"),
            "feature_provider": ["google_places"] * 9,
            "feature_snapshot_date": ["2026-09-11T12:00:00"] * 9,
        }
    )


def test_panel_uses_provider_features_and_keeps_all_guide_rows(tmp_path: Path) -> None:
    guide = guide_fixture()
    features = feature_fixture()
    before = features.copy(deep=True)
    result = build_labeled_panel(guide, features, panel_config(tmp_path))

    assert result.summary == {
        "rows": 7, "in_sample_rows": 6, "out_of_sample_rows": 1,
        "starred_rows": 2, "not_starred_rows": 4, "hard_negative_rows": 2,
    }
    panel = result.panel.set_index("source_id")
    assert panel.loc["guide-1", "name"] == "Google guide-1"
    assert panel.loc["guide-1", "price_level"] == 2
    assert panel.loc["guide-1", "cuisine"] == "chinese_restaurant"
    assert panel.loc["guide-3", "city"] == "New Taipei"
    assert (panel.loc[guide.source_id, "group"] == "in_sample").all()
    assert panel.loc[panel.group == "out_of_sample", "class"].isna().all()
    assert pd.isna(panel.loc["place-A", "price_level"])
    assert result.panel.restaurant_id.is_unique
    assert "outside" not in panel.index
    assert result.dlq.source_id.tolist() == ["bad"]
    pd.testing.assert_frame_equal(features, before)


def test_panel_sampling_and_outputs_are_repeatable_and_input_order_independent(tmp_path: Path) -> None:
    config = panel_config(tmp_path)
    first = build_labeled_panel(guide_fixture(), feature_fixture(), config)
    second = build_labeled_panel(guide_fixture().iloc[::-1], feature_fixture().iloc[::-1], config)
    pd.testing.assert_frame_equal(first.panel, second.panel)
    write_panel_outputs(first, config)
    before = (config.processed_dir / "labeled_restaurants.csv").read_bytes()
    write_panel_outputs(second, config)
    assert (config.processed_dir / "labeled_restaurants.csv").read_bytes() == before
    assert (config.interim_dir / "identity_aliases.csv").is_file()


@pytest.mark.parametrize("column,value", [("expected_starred_count", 999), ("expected_hard_negative_count", 999)])
def test_panel_reconciles_both_guide_populations(tmp_path: Path, column: str, value: int) -> None:
    config = replace(panel_config(tmp_path), **{column: value})
    with pytest.raises(ValueError, match="count reconciliation failed"):
        build_labeled_panel(guide_fixture(), feature_fixture(), config)


def test_panel_rejects_missing_guide_features(tmp_path: Path) -> None:
    features = feature_fixture().loc[lambda rows: rows.source_id != "guide-3"]
    with pytest.raises(ValueError, match="Guide.*features"):
        build_labeled_panel(guide_fixture(), features, panel_config(tmp_path))


def test_panel_rejects_provider_mismatch(tmp_path: Path) -> None:
    features = feature_fixture().assign(feature_provider="michelin")
    with pytest.raises(ValueError, match="provenance"):
        build_labeled_panel(guide_fixture(), features, panel_config(tmp_path))


def test_panel_rejects_duplicate_identity_across_pools(tmp_path: Path) -> None:
    features = pd.concat([feature_fixture(), feature_fixture().iloc[[4]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate.*source_id"):
        build_labeled_panel(guide_fixture(), features, panel_config(tmp_path))


def test_panel_fails_when_bounded_pools_cannot_be_filled(tmp_path: Path) -> None:
    config = replace(panel_config(tmp_path), ordinary_out_of_sample_limit=4)
    with pytest.raises(ValueError, match="ordinary.*required"):
        build_labeled_panel(guide_fixture(), feature_fixture(), config)


def test_panel_accepts_mixed_iso_timestamp_precision(tmp_path: Path) -> None:
    features = feature_fixture().assign(
        feature_snapshot_date=["2026-09-11T12:00:00.123456"] + ["2026-09-11T12:00:00"] * 8,
    )
    result = build_labeled_panel(guide_fixture(), features, panel_config(tmp_path))
    assert result.summary["rows"] == 7

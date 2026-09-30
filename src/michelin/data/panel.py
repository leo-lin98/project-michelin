"""Idempotent labeled dataset construction from shared provider features."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from michelin.config import Award
from michelin.data.enrichment import FEATURE_COLUMNS, enrich_with_uniform_provider, split_ordinary_pools


@dataclass(frozen=True)
class PanelBuildConfig:
    cities: tuple[str, ...]
    in_sample_group: str
    out_of_sample_group: str
    ordinary_in_sample_limit: int
    ordinary_out_of_sample_limit: int
    seed: int
    feature_provider: str
    processed_dir: Path
    interim_dir: Path
    expected_starred_count: int
    expected_hard_negative_count: int


@dataclass(frozen=True)
class PanelBuildResult:
    panel: pd.DataFrame
    dlq: pd.DataFrame
    alias_table: pd.DataFrame
    summary: dict[str, int]


STARRED_AWARDS = frozenset({Award.ONE_STAR.value, Award.TWO_STARS.value, Award.THREE_STARS.value})
HARD_NEGATIVE_AWARDS = frozenset({Award.BIB_GOURMAND.value, Award.SELECTED_RESTAURANTS.value})
OUTPUT_COLUMNS = (
    "restaurant_id", "source_id", *FEATURE_COLUMNS,
    "award", "class", "group", "is_hard_negative",
)


def build_labeled_panel(
    guide_rows: pd.DataFrame, feature_rows: pd.DataFrame, config: PanelBuildConfig,
) -> PanelBuildResult:
    """Keep all Guide rows, withhold ordinary display rows, and never impute features."""
    if guide_rows["source_id"].duplicated().any() or feature_rows["source_id"].duplicated().any():
        raise ValueError("Duplicate source_id values; resolve identities before building the panel")
    valid_features, dlq = split_invalid_rows(feature_rows)
    eligible = valid_features.loc[valid_features["city"].isin(config.cities)].copy()
    guide = enrich_with_uniform_provider(guide_rows, eligible, config.feature_provider)
    if not eligible["feature_provider"].eq(config.feature_provider).all():
        raise ValueError("Feature provenance mismatch in ordinary rows")
    ordinary = eligible.loc[~eligible["source_id"].isin(guide["source_id"])].copy()
    pools = split_ordinary_pools(ordinary, config.ordinary_in_sample_limit, config.ordinary_out_of_sample_limit, config.seed)
    guide_panel = guide.assign(
        **{"class": guide["award"].map(label_from_award), "group": config.in_sample_group,
           "is_hard_negative": guide["award"].isin(HARD_NEGATIVE_AWARDS)}
    )
    ordinary_in_sample = pools.in_sample.assign(
        **{"award": "none", "class": 0, "group": config.in_sample_group, "is_hard_negative": False}
    )
    ordinary_out_of_sample = pools.out_of_sample.assign(
        **{"award": "none", "class": pd.NA, "group": config.out_of_sample_group, "is_hard_negative": False}
    )
    panel = finalize_panel(pd.concat([guide_panel, ordinary_in_sample, ordinary_out_of_sample], ignore_index=True))
    summary = build_summary(panel, config.in_sample_group, config.out_of_sample_group)
    for label, actual, expected in (
        ("Starred", summary["starred_rows"], config.expected_starred_count),
        ("Hard-negative", summary["hard_negative_rows"], config.expected_hard_negative_count),
    ):
        if actual != expected:
            raise ValueError(f"{label} count reconciliation failed: expected {expected}, got {actual}")
    aliases = guide_rows.loc[:, ["source_id"]].merge(
        panel.loc[:, ["source_id", "restaurant_id"]], on="source_id", validate="one_to_one"
    ).sort_values("source_id").reset_index(drop=True)
    return PanelBuildResult(panel=panel, dlq=dlq, alias_table=aliases, summary=summary)


def write_panel_outputs(result: PanelBuildResult, config: PanelBuildConfig) -> None:
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.interim_dir.mkdir(parents=True, exist_ok=True)
    result.panel.to_csv(config.processed_dir / "labeled_restaurants.csv", index=False)
    result.dlq.to_csv(config.interim_dir / "panel_dlq.csv", index=False)
    result.alias_table.to_csv(config.interim_dir / "identity_aliases.csv", index=False)
    pd.DataFrame([result.summary]).to_json(config.processed_dir / "panel_summary.json", orient="records", indent=2)


def split_invalid_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"source_id", *FEATURE_COLUMNS}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Provider rows missing columns: {sorted(missing)}")
    valid = (
        rows["source_id"].fillna("").str.strip().ne("")
        & rows["name"].fillna("").str.strip().ne("")
        & rows["city"].fillna("").str.strip().ne("")
        & pd.to_numeric(rows["latitude"], errors="coerce").between(-90, 90)
        & pd.to_numeric(rows["longitude"], errors="coerce").between(-180, 180)
        & pd.to_datetime(rows["feature_snapshot_date"], format="ISO8601", errors="coerce").notna()
    )
    rejected = rows.loc[~valid].assign(dlq_reason="invalid_identity_or_snapshot")
    return rows.loc[valid].copy(), rejected.sort_values("source_id").reset_index(drop=True)


def label_from_award(award: str) -> int:
    if award in STARRED_AWARDS:
        return 1
    if award in HARD_NEGATIVE_AWARDS:
        return 0
    raise ValueError(f"Cannot derive class from award: {award}")


def finalize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    finalized = panel.assign(restaurant_id="google_places:" + panel["source_id"])
    finalized = finalized.assign(**{"class": pd.array(finalized["class"], dtype="Int64")})
    return finalized.loc[:, OUTPUT_COLUMNS].sort_values(["group", "restaurant_id"]).reset_index(drop=True)


def build_summary(panel: pd.DataFrame, in_sample_group: str, out_of_sample_group: str) -> dict[str, int]:
    in_sample = panel.loc[panel["group"] == in_sample_group]
    return {
        "rows": len(panel),
        "in_sample_rows": len(in_sample),
        "out_of_sample_rows": int(panel["group"].eq(out_of_sample_group).sum()),
        "starred_rows": int(in_sample["class"].eq(1).sum()),
        "not_starred_rows": int(in_sample["class"].eq(0).sum()),
        "hard_negative_rows": int(in_sample["is_hard_negative"].sum()),
    }

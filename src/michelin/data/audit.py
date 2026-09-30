"""Missing-feature counts for eligible candidates and the final bounded panel."""

from collections import Counter
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict


AUDITED_FEATURES = ("price_level", "average_rating", "review_count", "cuisine")


class MissingnessGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    city: str
    population: str
    group: str
    rows: int
    missing: dict[str, int]
    missing_fraction: dict[str, float]


class DataAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    business_statuses: tuple[str | None, ...]
    coverage_complete: bool
    published_reconciliation: Literal["deferred"]
    candidate_pool: tuple[MissingnessGroup, ...]
    final_panel: tuple[MissingnessGroup, ...] | None


def summarize_counts(counts: Counter[tuple[str, str, str, str]]) -> tuple[MissingnessGroup, ...]:
    groups = sorted({key[:3] for key in counts})
    return tuple(
        MissingnessGroup(
            city=city, population=population, group=group,
            rows=counts[(city, population, group, "rows")],
            missing={field: counts[(city, population, group, field)] for field in AUDITED_FEATURES},
            missing_fraction={field: counts[(city, population, group, field)] / counts[(city, population, group, "rows")] for field in AUDITED_FEATURES},
        )
        for city, population, group in groups
    )


def panel_missingness(panel: pd.DataFrame) -> tuple[MissingnessGroup, ...]:
    populations = panel["award"].map(
        lambda award: "ordinary" if award == "none" else "starred" if award in {"1 Star", "2 Stars", "3 Stars"} else "hard_negative"
    )
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for (city, population, group), rows in panel.assign(population=populations).groupby(["city", "population", "group"]):
        counts[(city, population, group, "rows")] = len(rows)
        for field in AUDITED_FEATURES:
            counts[(city, population, group, field)] = int(rows[field].isna().sum())
    return summarize_counts(counts)

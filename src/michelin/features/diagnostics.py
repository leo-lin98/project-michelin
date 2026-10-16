"""Descriptive population slices and an explicit missingness leakage gate."""

import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy.stats import fisher_exact

from michelin.config import AppConfig
from michelin.data.panel import HARD_NEGATIVE_AWARDS, STARRED_AWARDS


RAW_FIELDS = ("price_range", "average_rating", "review_count", "cuisine")


class MissingnessTest(BaseModel):
    model_config = ConfigDict(frozen=True)
    feature: str
    starred_missing: int
    starred_present: int
    not_starred_missing: int
    not_starred_present: int
    starred_missing_rate: float
    not_starred_missing_rate: float
    raw_p_value: float
    adjusted_p_value: float
    significant: bool


class NumericSummary(BaseModel):
    count: int
    missing: int
    mean: float | None
    median: float | None
    minimum: float | None
    maximum: float | None


class PopulationSummary(BaseModel):
    population: str
    city: str
    rows: int
    numeric: dict[str, NumericSummary]
    categorical_counts: dict[str, dict[str, int]]
    missingness: dict[str, int]


class FeatureReport(BaseModel):
    validation_status: str
    feature_names: tuple[str, ...]
    density_radius_m: float
    missingness_alpha: float
    missingness_tests: tuple[MissingnessTest, ...]
    slices: tuple[PopulationSummary, ...]
    caveats: tuple[str, ...]


def validate_populations(panel: pd.DataFrame, config: AppConfig) -> None:
    required = {"class", "group", "award", "is_hard_negative"}
    if not required.issubset(panel.columns):
        raise ValueError(f"Panel population metadata requires {sorted(required)}")
    modeling = panel.loc[panel["group"].eq(config.pipeline.groups.in_sample)]
    display = panel.loc[panel["group"].eq(config.pipeline.groups.out_of_sample)]
    if len(modeling) + len(display) != len(panel) or not modeling["class"].isin([0, 1]).all() or not display["class"].isna().all():
        raise ValueError("Invalid modeling/display populations or labels")
    if not display["award"].eq("none").all():
        raise ValueError("Display rows must be ordinary restaurants")
    if not panel["award"].isin({"none", *STARRED_AWARDS, *HARD_NEGATIVE_AWARDS}).all():
        raise ValueError("Unsupported award metadata in panel")
    if not modeling.loc[modeling["award"].isin(STARRED_AWARDS), "class"].eq(1).all() or not modeling.loc[~modeling["award"].isin(STARRED_AWARDS), "class"].eq(0).all():
        raise ValueError("Panel classes disagree with award metadata")
    if not panel["is_hard_negative"].eq(panel["award"].isin(HARD_NEGATIVE_AWARDS)).all():
        raise ValueError("Hard-negative metadata disagrees with award")
    if not modeling["class"].eq(1).any() or not modeling["class"].eq(0).any():
        raise ValueError("Missingness tests require both starred and not-starred modeling rows")


def missingness_tests(panel: pd.DataFrame, config: AppConfig) -> tuple[MissingnessTest, ...]:
    """Compare raw missingness in modeling rows using two-sided Fisher and Holm."""
    config.features.require_phase_2_schema()
    validate_populations(panel, config)
    modeling = panel.loc[panel["group"].eq(config.pipeline.groups.in_sample)]
    starred = modeling.loc[modeling["class"].eq(1)]
    negative = modeling.loc[modeling["class"].eq(0)]
    counts = [(int(starred[field].isna().sum()), int(negative[field].isna().sum())) for field in RAW_FIELDS]
    p_values = [float(fisher_exact([[sm, len(starred) - sm], [nm, len(negative) - nm]], alternative="two-sided").pvalue) for sm, nm in counts]
    adjusted = [1.0] * len(p_values)
    cumulative = 0.0
    for rank, position in enumerate(sorted(range(len(p_values)), key=lambda index: (p_values[index], index))):
        cumulative = max(cumulative, min(1.0, (len(p_values) - rank) * p_values[position]))
        adjusted[position] = cumulative
    return tuple(MissingnessTest(
        feature=field, starred_missing=sm, starred_present=len(starred) - sm,
        not_starred_missing=nm, not_starred_present=len(negative) - nm,
        starred_missing_rate=sm / len(starred), not_starred_missing_rate=nm / len(negative),
        raw_p_value=p_values[index], adjusted_p_value=adjusted[index],
        significant=adjusted[index] <= config.features.leakage_controls.missingness_alpha,
    ) for index, (field, (sm, nm)) in enumerate(zip(RAW_FIELDS, counts, strict=True)))


def summarize_population(rows: pd.DataFrame, population: str, city: str, config: AppConfig) -> PopulationSummary:
    numeric = {}
    for field in config.features.feature_schema.numeric:
        observed = rows[field].dropna()
        numeric[field] = NumericSummary(count=len(observed), missing=int(rows[field].isna().sum()),
            mean=float(observed.mean()) if len(observed) else None,
            median=float(observed.median()) if len(observed) else None,
            minimum=float(observed.min()) if len(observed) else None,
            maximum=float(observed.max()) if len(observed) else None)
    return PopulationSummary(population=population, city=city, rows=len(rows), numeric=numeric,
        categorical_counts={field: {str(key): int(value) for key, value in rows[field].dropna().value_counts().sort_index().items()} for field in config.features.feature_schema.categorical},
        missingness={field: int(rows[field].isna().sum()) for field in RAW_FIELDS})


def build_report(panel: pd.DataFrame, features: pd.DataFrame, config: AppConfig) -> FeatureReport:
    tests = missingness_tests(panel, config)
    rows = panel.set_index("restaurant_id").assign(neighborhood_density=features["neighborhood_density"])
    modeling = rows.loc[rows["group"].eq(config.pipeline.groups.in_sample)]
    populations = {
        "starred": modeling.loc[modeling["class"].eq(1)],
        "ordinary": modeling.loc[modeling["award"].eq("none")],
        "bib_selected": modeling.loc[modeling["award"].isin(HARD_NEGATIVE_AWARDS)],
        "display_ordinary": rows.loc[rows["group"].eq(config.pipeline.groups.out_of_sample)],
    }
    summaries = tuple(summarize_population(population_rows if city == "all" else population_rows.loc[population_rows["city"].eq(city)], population, city, config)
        for population, population_rows in populations.items() for city in ("all", *config.pipeline.project.geography))
    city_counts = {city: int(populations["starred"]["city"].eq(city).sum()) for city in config.pipeline.project.geography}
    return FeatureReport(validation_status="blocked_missingness_leakage" if any(test.significant for test in tests) else "passed",
        feature_names=tuple(features.columns), density_radius_m=config.features.density_radius_m,
        missingness_alpha=config.features.leakage_controls.missingness_alpha,
        missingness_tests=tests, slices=summaries, caveats=(
            "Descriptive associations are correlation, not causation; no model importances are computed in Phase 2.",
            "Nonsignificant missingness tests do not prove independence; power depends on sample size.",
            f"Municipality confounding: starred counts by city are {city_counts}; city can proxy the observed label geography.",
            "Cuisine is the shared Google primary-category proxy; unsupported text, other-guide awards and tenure are omitted.",
        ))

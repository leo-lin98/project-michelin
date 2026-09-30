"""Read-only DuckDB export with Guide coverage and shared Google feature provenance."""

import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from tempfile import mkdtemp
from typing import Literal

import duckdb
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from michelin.config import Award, PipelineConfig
from michelin.data.audit import AUDITED_FEATURES, DataAudit, MissingnessGroup, panel_missingness, summarize_counts
from michelin.data.snapshot import sha256_file
from michelin.data.enrichment import FEATURE_COLUMNS
from michelin.data.panel import HARD_NEGATIVE_AWARDS, STARRED_AWARDS, PanelBuildConfig, build_labeled_panel, write_panel_outputs


PRICE_LEVELS = {
    "PRICE_LEVEL_UNSPECIFIED": None,
    "PRICE_LEVEL_FREE": 0,
    "PRICE_LEVEL_INEXPENSIVE": 1,
    "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3,
    "PRICE_LEVEL_VERY_EXPENSIVE": 4,
}
FETCH_BATCH_SIZE = 1000


class GuideCoverageError(ValueError):
    """Scoped Guide identities or provider records remain unresolved."""


class GuideRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1)
    city: Literal["Taipei", "New Taipei"]
    award: Award
    michelin_url: str = Field(min_length=1)


class DatabaseRestaurant(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
    source_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    address: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    average_rating: float | None = Field(ge=0, le=5)
    review_count: int | None = Field(ge=0)
    price_level: Literal[
        "PRICE_LEVEL_UNSPECIFIED", "PRICE_LEVEL_FREE", "PRICE_LEVEL_INEXPENSIVE",
        "PRICE_LEVEL_MODERATE", "PRICE_LEVEL_EXPENSIVE", "PRICE_LEVEL_VERY_EXPENSIVE",
    ] | None
    primary_category: str | None
    fetched_at: datetime
    is_michelin: bool
    business_status: Literal["OPERATIONAL", "CLOSED_TEMPORARILY", "CLOSED_PERMANENTLY", "FUTURE_OPENING"] | None


def city_from_address(address: str) -> str | None:
    """Use explicit municipality names; ambiguous addresses require review."""
    parts = {part.strip().casefold() for part in address.split(",")}
    if "新北市" in address or parts.intersection({"new taipei city", "new taipei"}):
        return "New Taipei"
    if "台北市" in address or "臺北市" in address or parts.intersection({"taipei city", "taipei"}):
        return "Taipei"
    return None


def read_scoped_guide(path: Path, cities: tuple[str, ...]) -> tuple[GuideRow, ...]:
    """Stream the global snapshot and retain only the configured Guide universe."""
    retained: list[GuideRow] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Name", "Location", "Award", "Url"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Guide snapshot {path} requires columns {sorted(required)}")
        for row in reader:
            location = tuple(part.strip() for part in row["Location"].split(","))
            if len(location) != 2 or location[0] not in cities or location[1] != "Taiwan":
                continue
            retained.append(GuideRow.model_validate({
                "name": row["Name"], "city": location[0], "award": row["Award"], "michelin_url": row["Url"],
            }))
    if not retained:
        raise ValueError(f"Guide snapshot {path} has no rows for {cities}")
    urls = [row.michelin_url for row in retained]
    if len(set(urls)) != len(urls):
        raise ValueError(f"Guide snapshot {path} contains duplicate Michelin URLs")
    return tuple(retained)


def read_guide_place_ids(path: Path, guide: tuple[GuideRow, ...]) -> dict[str, str]:
    wanted = {row.michelin_url for row in guide}
    mappings: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"place_id", "michelin_url"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"Resolved mapping {path} requires place_id and michelin_url columns")
        for row in reader:
            url = row["michelin_url"]
            if url not in wanted:
                continue
            place_id = row["place_id"].strip()
            if not place_id or place_id.lower() == "nan":
                continue
            if url in mappings and mappings[url] != place_id:
                raise ValueError(f"Conflicting Place IDs for Guide URL {url}")
            mappings[url] = place_id
    if len(set(mappings.values())) != len(mappings):
        raise ValueError("Multiple Guide restaurants share a Place ID; resolve the identity collision")
    return mappings


def guide_coverage_report(
    connection: duckdb.DuckDBPyConnection, guide: tuple[GuideRow, ...], mappings: dict[str, str], business_statuses: tuple[str | None, ...],
    exclusions: tuple[str, ...],
) -> pd.DataFrame:
    if set(exclusions).intersection(mappings):
        raise ValueError("An excluded unresolved Guide entry now has a mapping; review its exclusion before building")
    present = dict(connection.execute(
        "SELECT place_id, business_status FROM restaurants WHERE place_id IN (SELECT unnest(?))", [list(mappings.values())]
    ).fetchall())
    return pd.DataFrame([
        {"name": row.name, "city": row.city, "award": row.award.value, "michelin_url": row.michelin_url,
         "place_id": mappings.get(row.michelin_url, ""),
         "status": "excluded_by_user" if row.michelin_url in exclusions else
                   "missing_place_id" if row.michelin_url not in mappings else
                   "missing_database_features" if mappings[row.michelin_url] not in present else
                   "excluded_business_status" if present[mappings[row.michelin_url]] not in business_statuses else "ready"}
        for row in guide
    ]).sort_values(["city", "name"])


def provider_feature_row(row: DatabaseRestaurant, city: str) -> dict[str, str | int | float | None]:
    """Use the same Google primary-type proxy and ordinal price mapping for every row."""
    return {
        "source_id": row.source_id, "name": row.name, "address": row.address,
        "city": city, "latitude": row.latitude, "longitude": row.longitude,
        "average_rating": row.average_rating, "review_count": row.review_count,
        "price_level": None if row.price_level is None else PRICE_LEVELS[row.price_level],
        "cuisine": row.primary_category, "feature_provider": "google_places",
        "feature_snapshot_date": row.fetched_at.isoformat(),
    }


@dataclass(frozen=True)
class FeatureReadResult:
    features: pd.DataFrame
    missingness: tuple[MissingnessGroup, ...]


def read_bounded_features(
    connection: duckdb.DuckDBPyConnection, guide_awards: dict[str, str], config: PipelineConfig, dlq_path: Path,
) -> FeatureReadResult:
    """Stream rows in seeded ID order; retain all Guide rows plus the bounded ordinary pools."""
    cursor = connection.execute(
        """SELECT r.place_id AS source_id, r.name, r.address, r.latitude, r.longitude,
                  r.rating AS average_rating, r.review_count, r.price_level, r.primary_category,
                  raw.fetched_at, r.is_michelin, r.business_status
           FROM restaurants r LEFT JOIN place_raw_responses raw USING (place_id)
           ORDER BY sha256(? || ':' || r.place_id), r.place_id""", [str(config.seed)]
    )
    columns = [column[0] for column in cursor.description]
    limit = config.pools.ordinary_in_sample_limit + config.pools.ordinary_out_of_sample_limit
    ordinary_count = 0
    counts: Counter[tuple[str, str, str, str]] = Counter()
    features: list[dict[str, str | int | float | None]] = []
    dlq_path.parent.mkdir(parents=True, exist_ok=True)
    with dlq_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_id", "reason"])
        writer.writeheader()
        while batch := cursor.fetchmany(FETCH_BATCH_SIZE):
            for values in batch:
                try:
                    row = DatabaseRestaurant.model_validate(dict(zip(columns, values, strict=True)))
                except ValidationError as exc:
                    writer.writerow({"source_id": values[0], "reason": "; ".join(
                        f"{'.'.join(map(str, error['loc']))}: {error['type']}" for error in exc.errors(include_input=False)
                    )})
                    continue
                city = city_from_address(row.address)
                if city not in config.project.geography:
                    writer.writerow({"source_id": row.source_id, "reason": "outside_scope_or_unresolved_city"})
                    continue
                if row.business_status not in config.eligibility.business_statuses:
                    writer.writerow({"source_id": row.source_id, "reason": f"excluded_business_status:{row.business_status or 'unknown'}"})
                    continue
                feature = provider_feature_row(row, city)
                award = guide_awards.get(row.source_id)
                population = "ordinary_candidate" if award is None else "starred" if award in STARRED_AWARDS else "hard_negative"
                counts[(city, population, "eligible_pool", "rows")] += 1
                for field in AUDITED_FEATURES:
                    if feature[field] is None:
                        counts[(city, population, "eligible_pool", field)] += 1
                if row.source_id not in guide_awards:
                    if row.is_michelin:
                        raise ValueError(f"Database Guide Place ID {row.source_id} is absent from the selected Guide snapshot")
                    if ordinary_count >= limit:
                        continue
                    ordinary_count += 1
                features.append(feature)
    frame = pd.DataFrame(features, columns=["source_id", *FEATURE_COLUMNS])
    typed = frame.assign(
        price_level=pd.array(frame["price_level"], dtype="Int64"),
        review_count=pd.array(frame["review_count"], dtype="Int64"),
        average_rating=pd.array(frame["average_rating"], dtype="Float64"),
    )
    return FeatureReadResult(features=typed, missingness=summarize_counts(counts))


def export_database_panel(
    database_path: Path, guide_path: Path, mapping_path: Path, output_dir: Path, config: PipelineConfig,
) -> dict[str, int]:
    """Invalidate previous outputs before rebuilding; failed attempts publish no final panel."""
    archive_panel_outputs(output_dir)
    try:
        return build_database_panel(database_path, guide_path, mapping_path, output_dir, config)
    except Exception:
        archive_panel_outputs(output_dir)
        raise


def archive_panel_outputs(output_dir: Path) -> None:
    """Preserve prior or partial panel files away from downstream input paths."""
    paths = (
        Path("processed/labeled_restaurants.csv"),
        Path("processed/panel_summary.json"),
        Path("processed/panel_manifest.json"),
        Path("interim/identity_aliases.csv"),
        Path("interim/panel_dlq.csv"),
    )
    existing = tuple(path for path in paths if (output_dir / path).exists())
    if not existing:
        return
    archive_root = output_dir / "interim" / "previous_panels"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_dir = Path(mkdtemp(prefix="panel-", dir=archive_root))
    for path in existing:
        archived_path = archive_dir / path
        archived_path.parent.mkdir(parents=True, exist_ok=True)
        (output_dir / path).rename(archived_path)


def build_database_panel(
    database_path: Path, guide_path: Path, mapping_path: Path, output_dir: Path, config: PipelineConfig,
) -> dict[str, int]:
    """Export local modeling data only after every scoped Guide entry has shared features."""
    if config.data_sources.ordinary_base != "google_places" or config.data_sources.enrichment_provider != "google_places":
        raise ValueError("DuckDB export requires google_places as ordinary source and feature provider")
    guide = read_scoped_guide(guide_path, config.project.geography)
    mappings = read_guide_place_ids(mapping_path, guide)
    interim = output_dir / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database_path), read_only=True) as connection:
        coverage = guide_coverage_report(connection, guide, mappings, config.eligibility.business_statuses, config.eligibility.unresolved_guide_exclusions)
        coverage.to_csv(interim / "guide_coverage.csv", index=False)
        guide_awards = {mappings[row.michelin_url]: row.award.value for row in guide if row.michelin_url in mappings}
        read_result = read_bounded_features(connection, guide_awards, config, interim / "database_dlq.csv")
    missing = int(coverage["status"].isin({"missing_place_id", "missing_database_features"}).sum())
    audit = DataAudit(
        business_statuses=config.eligibility.business_statuses, coverage_complete=missing == 0,
        published_reconciliation=config.validation.published_reconciliation,
        candidate_pool=read_result.missingness, final_panel=None,
    )
    (interim / "missingness_audit.json").write_text(audit.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if missing:
        raise GuideCoverageError(f"Guide coverage incomplete: {missing}/{len(guide)} rows need resolution/enrichment; see {interim / 'guide_coverage.csv'}")
    eligible_guide = coverage.loc[coverage["status"].eq("ready")]
    guide_rows = eligible_guide.loc[:, ["place_id", "award"]].rename(columns={"place_id": "source_id"})
    panel_config = PanelBuildConfig(
        cities=config.project.geography, in_sample_group=config.groups.in_sample,
        out_of_sample_group=config.groups.out_of_sample,
        ordinary_in_sample_limit=config.pools.ordinary_in_sample_limit,
        ordinary_out_of_sample_limit=config.pools.ordinary_out_of_sample_limit,
        seed=config.seed, feature_provider=config.data_sources.enrichment_provider,
        processed_dir=output_dir / "processed", interim_dir=output_dir / "interim",
        expected_starred_count=int(eligible_guide["award"].isin(STARRED_AWARDS).sum()),
        expected_hard_negative_count=int(eligible_guide["award"].isin(HARD_NEGATIVE_AWARDS).sum()),
    )
    result = build_labeled_panel(guide_rows, read_result.features, panel_config)
    write_panel_outputs(result, panel_config)
    final_audit = audit.model_copy(update={"final_panel": panel_missingness(result.panel)})
    (interim / "missingness_audit.json").write_text(final_audit.model_dump_json(indent=2) + "\n", encoding="utf-8")
    manifest = {
        "database": str(database_path), "guide_snapshot": str(guide_path), "place_id_mapping": str(mapping_path),
        "cities": config.project.geography, "seed": config.seed, "feature_provider": "google_places",
        "cuisine_definition": "Google primary_category (primaryType), retained as a shared category proxy",
        "snapshot_policy": "per_record_fetched_at", "collection_start": result.panel["feature_snapshot_date"].min(),
        "collection_end": result.panel["feature_snapshot_date"].max(),
        "reconciliation_basis": "eligible Guide rows reconcile to the local snapshot; excluded Guide rows are recorded separately",
        "published_reconciliation": config.validation.published_reconciliation,
        "input_sha256": {"database": sha256_file(database_path), "guide": sha256_file(guide_path), "mapping": sha256_file(mapping_path)},
        "eligibility": config.eligibility.model_dump(mode="json"),
        "guide_excluded_rows": int(coverage["status"].isin({"excluded_business_status", "excluded_by_user"}).sum()),
        "distribution": "local modeling artifact; not approved for public map redistribution",
        "summary": result.summary,
    }
    (panel_config.processed_dir / "panel_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result.summary

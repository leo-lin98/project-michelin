"""Run Phase 1 acceptance checks against frozen inputs, including repeat builds."""

import csv
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd
from pydantic import BaseModel, ConfigDict

from michelin.config import PipelineConfig, load_config
from michelin.data.audit import DataAudit
from michelin.data.database import GuideCoverageError, export_database_panel
from michelin.data.panel import HARD_NEGATIVE_AWARDS, STARRED_AWARDS
from michelin.data.snapshot import sha256_file, verify_snapshot


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    status: Literal["passed", "failed", "blocked", "deferred"]
    detail: str


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    snapshot: str
    phase_1_ready: bool
    checks: tuple[CheckResult, ...]


def check_google_provenance(connection: duckdb.DuckDBPyConnection) -> CheckResult:
    mismatches = connection.execute(
        """SELECT count(*) FROM restaurants r LEFT JOIN place_raw_responses raw USING (place_id)
           WHERE raw.place_id IS NULL
              OR r.place_id IS DISTINCT FROM json_extract_string(raw.response_json, '$.id')
              OR r.name IS DISTINCT FROM json_extract_string(raw.response_json, '$.displayName.text')
              OR r.address IS DISTINCT FROM json_extract_string(raw.response_json, '$.formattedAddress')
              OR r.latitude IS DISTINCT FROM TRY_CAST(json_extract(raw.response_json, '$.location.latitude') AS DOUBLE)
              OR r.longitude IS DISTINCT FROM TRY_CAST(json_extract(raw.response_json, '$.location.longitude') AS DOUBLE)
              OR r.rating IS DISTINCT FROM TRY_CAST(json_extract(raw.response_json, '$.rating') AS DOUBLE)
              OR r.review_count IS DISTINCT FROM TRY_CAST(json_extract(raw.response_json, '$.userRatingCount') AS INTEGER)
              OR r.price_level IS DISTINCT FROM json_extract_string(raw.response_json, '$.priceLevel')
              OR r.primary_category IS DISTINCT FROM json_extract_string(raw.response_json, '$.primaryType')
              OR r.business_status IS DISTINCT FROM json_extract_string(raw.response_json, '$.businessStatus')"""
    ).fetchone()[0]
    return CheckResult(name="google_feature_provenance", status="failed" if mismatches else "passed", detail=f"{mismatches} restaurant rows differ from their raw Google response")


def check_raw_history(connection: duckdb.DuckDBPyConnection) -> CheckResult:
    missing, bad_hashes = connection.execute(
        """SELECT
            (SELECT count(*) FROM place_raw_responses r WHERE NOT EXISTS (
                SELECT 1 FROM place_raw_response_history h WHERE h.place_id=r.place_id
                    AND h.response_hash=r.response_hash AND h.fetched_at=r.fetched_at
                    AND h.response_json=r.response_json)),
            (SELECT count(*) FROM place_raw_response_history
             WHERE response_hash != sha256(CAST(response_json AS VARCHAR)))"""
    ).fetchone()
    return CheckResult(name="raw_history", status="failed" if missing or bad_hashes else "passed", detail=f"{missing} current responses absent from history; {bad_hashes} invalid content hashes")


def check_panel_invariants(path: Path, config: PipelineConfig) -> CheckResult:
    panel = pd.read_csv(path, dtype={"class": "Int64", "price_level": "Int64", "review_count": "Int64"})
    modeling = panel.loc[panel["group"].eq(config.groups.in_sample)]
    display = panel.loc[panel["group"].eq(config.groups.out_of_sample)]
    guide = panel.loc[panel["award"].ne("none")]
    ordinary = modeling.loc[modeling["award"].eq("none")]
    valid = (
        panel["source_id"].is_unique and panel["restaurant_id"].is_unique
        and panel["city"].isin(config.project.geography).all()
        and panel["group"].isin({config.groups.in_sample, config.groups.out_of_sample}).all()
        and modeling["class"].isin({0, 1}).all() and display["class"].isna().all()
        and display["award"].eq("none").all() and guide["group"].eq(config.groups.in_sample).all()
        and modeling.loc[modeling["award"].isin(STARRED_AWARDS), "class"].eq(1).all()
        and modeling.loc[modeling["award"].isin(HARD_NEGATIVE_AWARDS), "class"].eq(0).all()
        and ordinary["class"].eq(0).all()
        and panel["is_hard_negative"].eq(panel["award"].isin(HARD_NEGATIVE_AWARDS)).all()
        and len(display) == config.pools.ordinary_out_of_sample_limit
        and len(ordinary) == config.pools.ordinary_in_sample_limit
        and panel["feature_provider"].eq(config.data_sources.enrichment_provider).all()
    )
    return CheckResult(name="pool_identity_and_labels", status="passed" if valid else "failed", detail=f"Checked {len(modeling)} modeling and {len(display)} display rows")


def build_once(snapshot: Path, destination: Path, config: PipelineConfig) -> CheckResult:
    try:
        export_database_panel(snapshot / "restaurants.duckdb", snapshot / "guide.csv", snapshot / "place_ids.csv", destination, config)
    except GuideCoverageError as exc:
        return CheckResult(name="guide_coverage", status="blocked", detail=str(exc))
    except ValueError as exc:
        return CheckResult(name="dataset_build", status="failed", detail=str(exc))
    return CheckResult(name="guide_coverage", status="passed", detail="Eligible and excluded Guide counts reconcile to the frozen local snapshot")


def output_hashes(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): sha256_file(path) for path in sorted(directory.rglob("*")) if path.is_file()}


def validate_snapshot(snapshot: Path, destination: Path) -> ValidationReport:
    """Keep unavailable final-panel checks blocked; deferred edition work never appears passed."""
    verify_snapshot(snapshot)
    config = load_config(snapshot / "pipeline.yaml", snapshot / "features.yaml").pipeline
    destination.mkdir(parents=True, exist_ok=False)
    checks = [CheckResult(name="snapshot_integrity", status="passed", detail="All frozen file hashes match")]
    with duckdb.connect(str(snapshot / "restaurants.duckdb"), read_only=True) as connection:
        checks.extend((check_google_provenance(connection), check_raw_history(connection)))
        restaurant_count = connection.execute("SELECT count(*) FROM restaurants").fetchone()[0]
    if all(check.status == "passed" for check in checks):
        first = build_once(snapshot, destination / "build_a", config)
        second = build_once(snapshot, destination / "build_b", config)
        checks.append(first)
        repeated = first.status == second.status and output_hashes(destination / "build_a") == output_hashes(destination / "build_b")
        checks.append(CheckResult(name="repeat_build", status="passed" if repeated else "failed", detail="Compared all emitted artifacts from two builds; blocked builds compare diagnostics only"))
        audit_path = destination / "build_a" / "interim" / "missingness_audit.json"
        if audit_path.exists():
            audit = DataAudit.model_validate_json(audit_path.read_text(encoding="utf-8"))
            geography_valid = all(group.city in config.project.geography for group in audit.candidate_pool)
            with (destination / "build_a" / "interim" / "database_dlq.csv").open(newline="") as stream:
                rejected_count = 0
                rejection_reasons_valid = True
                for row in csv.DictReader(stream):
                    rejected_count += 1
                    rejection_reasons_valid = rejection_reasons_valid and bool(row["source_id"] and row["reason"])
            accounted = sum(group.rows for group in audit.candidate_pool) + rejected_count == restaurant_count
            checks.append(CheckResult(name="rejected_row_accounting", status="passed" if accounted and rejection_reasons_valid else "failed", detail=f"Eligible plus rejected/excluded rows account for {restaurant_count} source rows; rejection identities and reasons checked"))
            checks.append(CheckResult(name="eligible_geography_and_ranges", status="passed" if geography_valid else "failed", detail=f"Eligible rows passed provider schema/range and city checks; {rejected_count} rejected/excluded rows recorded"))
            checks.append(CheckResult(name="missingness_audit", status="passed", detail="Null counts/fractions recorded by city and population; final-panel audit remains null until a complete build"))
        if first.status == "passed":
            checks.append(check_panel_invariants(destination / "build_a" / "processed" / "labeled_restaurants.csv", config))
        else:
            checks.append(CheckResult(name="pool_identity_and_labels", status="blocked", detail="No final dataset exists until the build gate passes"))
    else:
        checks.append(CheckResult(name="dataset_build", status="blocked", detail="Frozen input provenance/history checks must pass before building"))
    checks.append(CheckResult(name="published_reconciliation", status="deferred", detail="Guide edition and independent published counts deferred by the user"))
    verify_snapshot(snapshot)
    report = ValidationReport(snapshot=str(snapshot), phase_1_ready=all(check.status in {"passed", "deferred"} for check in checks), checks=tuple(checks))
    (destination / "validation.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report

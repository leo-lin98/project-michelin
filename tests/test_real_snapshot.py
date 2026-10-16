"""Acceptance checks use the local frozen dataset when it is available."""

import csv
from pathlib import Path

import duckdb
import pytest

from michelin.data.audit import DataAudit
from michelin.data.validation import validate_snapshot


@pytest.mark.parametrize("snapshot_name", ["2026-09-11-phase-1", "phase1-six-excluded", "phase1-all-statuses"])
def test_real_snapshot_accounts_for_every_row_and_repeats_without_mutation(tmp_path: Path, snapshot_name: str) -> None:
    snapshot = Path("data/snapshots") / snapshot_name
    if not snapshot.exists():
        pytest.skip("Local-only frozen restaurant dataset is unavailable")
    report = validate_snapshot(snapshot, tmp_path / "validation")
    checks = {check.name: check.status for check in report.checks}
    for name in ("snapshot_integrity", "google_feature_provenance", "raw_history", "repeat_build", "eligible_geography_and_ranges", "missingness_audit"):
        assert checks[name] == "passed"
    interim = tmp_path / "validation" / "build_a" / "interim"
    audit = DataAudit.model_validate_json((interim / "missingness_audit.json").read_text())
    with (interim / "database_dlq.csv").open(newline="") as stream:
        rejected = list(csv.DictReader(stream))
    assert all(row["source_id"] and row["reason"] for row in rejected)
    assert len({row["source_id"] for row in rejected}) == len(rejected)
    with duckdb.connect(str(snapshot / "restaurants.duckdb"), read_only=True) as connection:
        total = connection.execute("SELECT count(*) FROM restaurants").fetchone()[0]
    assert sum(group.rows for group in audit.candidate_pool) + len(rejected) == total
    with (interim / "guide_coverage.csv").open(newline="") as stream:
        coverage = list(csv.DictReader(stream))
    unresolved = [row for row in coverage if row["status"] in {"missing_place_id", "missing_database_features"}]
    if unresolved:
        assert checks["guide_coverage"] == "blocked"
        assert checks["pool_identity_and_labels"] == "blocked"
        assert not report.phase_1_ready
        assert audit.final_panel is None
    else:
        assert checks["guide_coverage"] == "passed"
        assert checks["pool_identity_and_labels"] == "passed"
        assert audit.final_panel is not None
    assert checks["published_reconciliation"] == "deferred"

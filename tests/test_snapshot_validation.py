import json
from pathlib import Path

import duckdb
import pytest
import yaml
from pydantic import ValidationError

from michelin.data import validation
from michelin.data.snapshot import freeze_snapshot
from test_database_export import export_fixture


def test_complete_frozen_build_checks_final_pools_and_is_byte_identical(tmp_path: Path) -> None:
    database, guide, mappings = export_fixture(tmp_path)
    pipeline = yaml.safe_load(Path("config/pipeline.yaml").read_text())
    pipeline["pools"]["ordinary_in_sample_limit"] = 2
    pipeline["pools"]["ordinary_out_of_sample_limit"] = 1
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(yaml.safe_dump(pipeline))
    directory = tmp_path / "snapshot"
    freeze_snapshot(database, guide, mappings, pipeline_path, Path("config/features.yaml"), directory)
    report = validation.validate_snapshot(directory, tmp_path / "validation")
    checks = {check.name: check.status for check in report.checks}
    assert report.phase_1_ready
    assert checks["guide_coverage"] == "passed"
    assert checks["pool_identity_and_labels"] == "passed"
    assert checks["repeat_build"] == "passed"
    assert checks["published_reconciliation"] == "deferred"


def test_validation_reports_final_checks_blocked_but_still_checks_real_inputs(tmp_path: Path) -> None:
    database, guide, mappings = export_fixture(tmp_path)
    mappings.write_text("place_id,michelin_url\nstar,https://guide.example/star\n")
    directory = tmp_path / "snapshot"
    freeze_snapshot(database, guide, mappings, Path("config/pipeline.yaml"), Path("config/features.yaml"), directory)
    report = validation.validate_snapshot(directory, tmp_path / "validation")
    checks = {check.name: check.status for check in report.checks}
    assert report.phase_1_ready is False
    assert checks["snapshot_integrity"] == "passed"
    assert checks["google_feature_provenance"] == "passed"
    assert checks["repeat_build"] == "passed"
    assert checks["guide_coverage"] == "blocked"
    assert checks["pool_identity_and_labels"] == "blocked"
    assert checks["published_reconciliation"] == "deferred"


def test_validation_detects_feature_values_not_matching_raw_google_response(tmp_path: Path) -> None:
    database, guide, mappings = export_fixture(tmp_path)
    with duckdb.connect(str(database)) as connection:
        connection.execute("UPDATE restaurants SET rating=1 WHERE place_id='star'")
    directory = tmp_path / "snapshot"
    freeze_snapshot(database, guide, mappings, Path("config/pipeline.yaml"), Path("config/features.yaml"), directory)
    report = validation.validate_snapshot(directory, tmp_path / "validation")
    provenance = next(check for check in report.checks if check.name == "google_feature_provenance")
    assert provenance.status == "failed"
    assert report.phase_1_ready is False
    assert not (tmp_path / "validation" / "build_a" / "processed" / "labeled_restaurants.csv").exists()


@pytest.mark.parametrize("place_id", ["star", "bib", "ordinary-1", "ordinary-2"])
@pytest.mark.parametrize("stored,raw_range,expected_status", [
    ("200-400", {"startPrice": {"currencyCode": "TWD", "units": "200"}, "endPrice": {"currencyCode": "TWD", "units": "400"}}, "passed"),
    ("2000+", {"startPrice": {"currencyCode": "TWD", "units": "2000"}}, "passed"),
    (None, None, "passed"),
    (None, {"startPrice": {"currencyCode": "TWD", "units": "2000"}}, "failed"),
    ("2000+", None, "failed"),
    ("200-400", {"startPrice": {"currencyCode": "TWD", "units": "2000"}}, "failed"),
])
def test_raw_google_price_range_parity_in_guide_and_ordinary_identities(
    tmp_path: Path, place_id: str, stored: str | None,
    raw_range: dict[str, dict[str, str]] | None, expected_status: str,
) -> None:
    database, _, _ = export_fixture(tmp_path)
    with duckdb.connect(str(database)) as connection:
        payload = json.loads(connection.execute(
            "SELECT response_json FROM place_raw_responses WHERE place_id=?", [place_id],
        ).fetchone()[0])
        if raw_range is not None:
            payload["priceRange"] = raw_range
        connection.execute("UPDATE restaurants SET price_range=? WHERE place_id=?", [stored, place_id])
        connection.execute("UPDATE place_raw_responses SET response_json=? WHERE place_id=?", [json.dumps(payload), place_id])
        result = validation.check_google_provenance(connection)
    assert result.status == expected_status


def test_malformed_raw_google_price_range_fails_explicitly(tmp_path: Path) -> None:
    database, _, _ = export_fixture(tmp_path)
    with duckdb.connect(str(database)) as connection:
        payload = json.loads(connection.execute(
            "SELECT response_json FROM place_raw_responses WHERE place_id='star'",
        ).fetchone()[0])
        payload["priceRange"] = {"startPrice": {"currencyCode": "TWD", "units": ["2000"]}}
        connection.execute("UPDATE place_raw_responses SET response_json=? WHERE place_id='star'", [json.dumps(payload)])
        with pytest.raises(ValidationError):
            validation.check_google_provenance(connection)

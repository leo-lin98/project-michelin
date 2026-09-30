from pathlib import Path

import duckdb
import pytest

from michelin.data import snapshot
from test_database_export import export_fixture


def test_frozen_snapshot_survives_live_changes_and_refuses_overwrite(tmp_path: Path) -> None:
    database, guide, mappings = export_fixture(tmp_path)
    destination = tmp_path / "snapshot"
    manifest = snapshot.freeze_snapshot(
        database, guide, mappings, Path("config/pipeline.yaml"), Path("config/features.yaml"), destination,
    )
    assert manifest.table_rows["restaurants"] == 6
    assert manifest.collection_start.startswith("2026-08-06")
    assert manifest.published_reconciliation == "deferred"
    with duckdb.connect(str(database)) as connection:
        connection.execute("UPDATE restaurants SET rating=1 WHERE place_id='star'")
    guide.write_text("changed live source")
    with duckdb.connect(str(destination / "restaurants.duckdb"), read_only=True) as connection:
        assert connection.execute("SELECT rating FROM restaurants WHERE place_id='star'").fetchone() == (4.5,)
    assert "Guide star" in (destination / "guide.csv").read_text()
    assert snapshot.verify_snapshot(destination) == manifest
    with pytest.raises(FileExistsError):
        snapshot.freeze_snapshot(database, guide, mappings, Path("config/pipeline.yaml"), Path("config/features.yaml"), destination)


def test_snapshot_verification_detects_changed_mapping(tmp_path: Path) -> None:
    database, guide, mappings = export_fixture(tmp_path)
    destination = tmp_path / "snapshot"
    snapshot.freeze_snapshot(database, guide, mappings, Path("config/pipeline.yaml"), Path("config/features.yaml"), destination)
    (destination / "place_ids.csv").write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch.*place_ids.csv"):
        snapshot.verify_snapshot(destination)

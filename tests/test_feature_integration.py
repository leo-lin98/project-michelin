"""Independent density and local feature-build boundary checks."""

from math import degrees
from pathlib import Path
import sys

import duckdb
import pandas as pd
import pytest
import yaml

from michelin.config import load_config
from michelin.data.database import export_database_panel
from michelin.features.database import PanelManifest, eligible_reference_batches, verify_manifest
from michelin.features.temporal import neighborhood_density
from michelin.features.diagnostics import FeatureReport
from scripts import build_features
from test_database_export import export_fixture


@pytest.mark.parametrize("batch_size", [1, 2, 20])
def test_density_counts_frozen_neighbors_excludes_self_and_includes_boundary(batch_size: int) -> None:
    """Reference includes duplicate coordinates and restaurants outside the queried panel."""
    distances = [0.0, 0.0, 499.999, 500.0, 500.001]
    reference = pd.DataFrame({
        "restaurant_id": [f"google_places:p{i}" for i in range(5)],
        "source_id": [f"p{i}" for i in range(5)],
        "latitude": [0.0] * 5,
        "longitude": [degrees(distance / 6371008.8) for distance in distances],
    })
    query = reference.iloc[[0]].copy(deep=True)
    before = query.copy(deep=True)
    reference_before = reference.copy(deep=True)
    batches = (reference.iloc[start:start + batch_size] for start in range(0, 5, batch_size))
    result = neighborhood_density(query, batches, 500.0)
    assert result.to_dict() == {"google_places:p0": 3}
    pd.testing.assert_frame_equal(query, before)
    pd.testing.assert_frame_equal(reference, reference_before)


@pytest.mark.parametrize("scenario", ["missing_self", "duplicate_self", "changed_coordinates", "invalid_coordinates"])
def test_density_rejects_invalid_frozen_reference_identity(scenario: str) -> None:
    query = pd.DataFrame({"restaurant_id": ["google_places:p"], "source_id": ["p"], "latitude": [25.0], "longitude": [121.5]})
    if scenario == "missing_self":
        batches = [query.assign(source_id="other")]
    elif scenario == "duplicate_self":
        batches = [query, query.copy(deep=True)]
    elif scenario == "changed_coordinates":
        batches = [query.assign(latitude=25.001)]
    else:
        batches = [query.assign(latitude=91.0)]
    with pytest.raises(ValueError):
        neighborhood_density(query, iter(batches), 500.0)


@pytest.fixture
def frozen_panel(tmp_path: Path) -> tuple[PanelManifest, pd.DataFrame, Path]:
    database, guide, mapping = export_fixture(tmp_path)
    config = load_config(Path("config/pipeline.yaml"), Path("config/features.yaml")).pipeline
    config = config.model_copy(update={"pools": config.pools.model_copy(update={"ordinary_in_sample_limit": 1, "ordinary_out_of_sample_limit": 1})})
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    output = tmp_path / "phase1"
    export_database_panel(database, guide, mapping, output, config)
    manifest = PanelManifest.model_validate_json((output / "processed" / "panel_manifest.json").read_text(encoding="utf-8"))
    panel = pd.read_csv(output / "processed" / "labeled_restaurants.csv", float_precision="round_trip")
    return manifest, panel, pipeline_path


def test_reference_pool_contains_unsampled_candidates_and_all_allowed_statuses(
    frozen_panel: tuple[PanelManifest, pd.DataFrame, Path],
) -> None:
    manifest, panel, pipeline_path = frozen_panel
    config = load_config(pipeline_path, Path("config/features.yaml")).pipeline
    with duckdb.connect(str(manifest.database)) as connection:
        connection.execute("UPDATE restaurants SET business_status='CLOSED_PERMANENTLY' WHERE place_id='ordinary-1'")
        connection.execute("UPDATE restaurants SET business_status=NULL WHERE place_id='ordinary-2'")
        connection.execute("UPDATE restaurants SET business_status='FUTURE_OPENING' WHERE place_id='ordinary-3'")
    # Eligibility is independently exercised here; hash integrity is tested separately.
    before = manifest.database.read_bytes()
    batches = list(eligible_reference_batches(manifest, config, 2, panel))
    assert all(len(batch) <= 2 for batch in batches)
    references = pd.concat(batches, ignore_index=True)
    assert set(references["source_id"]) == {"star", "bib", "ordinary-1", "ordinary-2", "ordinary-3"}
    assert len(panel) == 4
    assert manifest.database.read_bytes() == before
    result = neighborhood_density(panel, iter(batches), 500.0)
    assert (result == 4).all()


@pytest.mark.parametrize("source", ["database", "guide_snapshot", "place_id_mapping"])
def test_manifest_detects_each_frozen_source_tamper(
    frozen_panel: tuple[PanelManifest, pd.DataFrame, Path], source: str,
) -> None:
    manifest, _, pipeline_path = frozen_panel
    config = load_config(pipeline_path, Path("config/features.yaml")).pipeline
    verify_manifest(manifest, config)
    path = getattr(manifest, source)
    with path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_manifest(manifest, config)


def test_script_success_writes_only_local_raw_features_and_diagnostics(
    frozen_panel: tuple[PanelManifest, pd.DataFrame, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, panel, pipeline_path = frozen_panel
    phase1_path = pipeline_path.parent / "phase1" / "processed"
    inputs = [phase1_path / "labeled_restaurants.csv", phase1_path / "panel_manifest.json", manifest.database, manifest.guide_snapshot, manifest.place_id_mapping]
    before = {path: path.read_bytes() for path in inputs}
    destination = tmp_path / "features"
    monkeypatch.setattr(sys, "argv", ["build_features", "--panel", str(inputs[0]), "--manifest", str(inputs[1]), "--pipeline-config", str(pipeline_path), "--output-dir", str(destination)])
    build_features.main()
    assert {path.name for path in destination.iterdir()} == {"shared_features.csv", "feature_report.json"}
    features = pd.read_csv(destination / "shared_features.csv", index_col="restaurant_id")
    assert tuple(features.columns) == ("average_rating", "review_count", "neighborhood_density", "price_range", "cuisine", "city")
    assert set(features.index) == set(panel["restaurant_id"])
    assert (features["neighborhood_density"] == 4).all()
    assert all(path.read_bytes() == content for path, content in before.items())


def test_script_blocked_gate_retains_diagnostics_without_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Failure injection at the report boundary verifies publication behavior independently
    # of the statistical test, whose exact values are covered in test_leakage.py.
    config = load_config(Path("config/pipeline.yaml"), Path("config/features.yaml"))
    panel = pd.DataFrame([{
        "restaurant_id": "google_places:p", "source_id": "p", "city": "Taipei",
        "latitude": 25.0, "longitude": 121.5, "price_range": "2000+", "price_level": 2.0, "average_rating": 4.0,
        "review_count": 100.0, "cuisine": "restaurant", "feature_provider": "google_places",
        "feature_snapshot_date": "2026-09-01T00:00:00",
    }])
    source = tmp_path / "source.csv"
    panel.to_csv(source, index=False)
    source_before = source.read_bytes()
    manifest = PanelManifest(
        database=source, guide_snapshot=source, place_id_mapping=source,
        cities=config.pipeline.project.geography, seed=config.pipeline.seed,
        feature_provider="google_places", snapshot_policy="per_record_fetched_at",
        input_sha256={"database": "dummy", "guide": "dummy", "mapping": "dummy"},
        eligibility=config.pipeline.eligibility,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    blocked = FeatureReport(validation_status="blocked_missingness_leakage", feature_names=("price_range",), density_radius_m=500, missingness_alpha=.05, missingness_tests=(), slices=(), caveats=("diagnostic fixture",))
    monkeypatch.setattr(build_features, "verify_manifest", lambda _manifest, _config: None)
    monkeypatch.setattr(build_features, "eligible_reference_batches", lambda _manifest, _config, _batch_size, _panel: iter([panel]))
    monkeypatch.setattr(build_features, "build_report", lambda _panel, _features, _config: blocked)
    destination = tmp_path / "blocked"
    monkeypatch.setattr(sys, "argv", ["build_features", "--panel", str(source), "--manifest", str(manifest_path), "--output-dir", str(destination)])
    with pytest.raises(ValueError, match="Missingness leakage gate"):
        build_features.main()
    assert not (destination / "shared_features.csv").exists()
    report = FeatureReport.model_validate_json((destination / "feature_report.json").read_text(encoding="utf-8"))
    assert report.validation_status == "blocked_missingness_leakage"
    assert source.read_bytes() == source_before


def test_reference_eligibility_rejects_invalid_features_and_unclassified_municipality(
    frozen_panel: tuple[PanelManifest, pd.DataFrame, Path],
) -> None:
    manifest, panel, pipeline_path = frozen_panel
    config = load_config(pipeline_path, Path("config/features.yaml")).pipeline
    with duckdb.connect(str(manifest.database)) as connection:
        connection.execute("UPDATE restaurants SET rating=7.0 WHERE place_id='ordinary-1'")
        connection.execute("UPDATE restaurants SET address='Taipei restaurant, Taiwan' WHERE place_id='ordinary-2'")
    references = pd.concat(eligible_reference_batches(manifest, config, 2, panel), ignore_index=True)
    assert set(references["source_id"]) == {"star", "bib", "ordinary-3"}

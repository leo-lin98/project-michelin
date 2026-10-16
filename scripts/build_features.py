"""Build local Phase 2 raw features and diagnostics without fitting model transforms."""

import argparse
from pathlib import Path

import pandas as pd

from michelin.config import load_config
from michelin.features.build import build_shared_features, validate_panel
from michelin.features.database import PanelManifest, eligible_reference_batches, verify_manifest
from michelin.features.diagnostics import build_report
from michelin.features.temporal import neighborhood_density


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/processed/labeled_restaurants.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/panel_manifest.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pipeline-config", type=Path, default=Path("config/pipeline.yaml"))
    parser.add_argument("--features-config", type=Path, default=Path("config/features.yaml"))
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    config = load_config(args.pipeline_config, args.features_config)
    config.features.require_phase_2_schema()
    if config.features.feature_schema.optional_text:
        raise ValueError("Optional review text features have no shared source and must be disabled")
    manifest = PanelManifest.model_validate_json(args.manifest.read_text(encoding="utf-8"))
    verify_manifest(manifest, config.pipeline)
    panel = pd.read_csv(args.panel, dtype={"restaurant_id": "str", "source_id": "str", "price_range": "str", "class": "Int64"}, float_precision="round_trip")
    validate_panel(panel, config)
    density = neighborhood_density(panel, eligible_reference_batches(manifest, config.pipeline, 1000, panel), config.features.density_radius_m)
    features = build_shared_features(panel, density, config)
    report = build_report(panel, features, config)
    protected = {args.panel.resolve(), args.manifest.resolve(), manifest.database.resolve(), manifest.guide_snapshot.resolve(), manifest.place_id_mapping.resolve()}
    matrix_path, report_path = destination / "shared_features.csv", destination / "feature_report.json"
    if matrix_path in protected or report_path in protected:
        raise ValueError("Feature outputs would overwrite Phase 1 inputs")
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if report.validation_status != "passed":
        raise ValueError(f"Missingness leakage gate blocked feature readiness; see {report_path}")
    features.to_csv(matrix_path, index=True)
    print(f"Phase 2 feature validation passed: {len(features)} rows; report: {report_path}")


if __name__ == "__main__":
    main()

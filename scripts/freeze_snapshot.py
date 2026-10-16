"""Freeze local restaurant data and configuration for an auditable panel build."""

import argparse
from pathlib import Path

from michelin.data.snapshot import freeze_snapshot
from restaurant_ingestion.storage.duckdb_store import DuckDbStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("database", "guide-csv", "place-ids-csv", "pipeline-config", "features-config", "output-dir"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    args = parser.parse_args()
    store = DuckDbStore(args.database)
    try:
        store.create_schema()
    finally:
        store.close()
    manifest = freeze_snapshot(args.database, args.guide_csv, args.place_ids_csv, args.pipeline_config, args.features_config, args.output_dir)
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

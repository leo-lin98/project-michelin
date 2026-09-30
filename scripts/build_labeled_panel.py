"""Build the local modeling panel from DuckDB without network calls or database writes."""

import argparse
import json
from pathlib import Path

from michelin.config import load_config
from michelin.data.database import export_database_panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--guide-csv", type=Path, required=True)
    parser.add_argument("--place-ids-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pipeline-config", type=Path, default=Path("config/pipeline.yaml"))
    parser.add_argument("--features-config", type=Path, default=Path("config/features.yaml"))
    args = parser.parse_args()
    config = load_config(args.pipeline_config, args.features_config)
    result = export_database_panel(args.database, args.guide_csv, args.place_ids_csv, args.output_dir, config.pipeline)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

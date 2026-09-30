"""Validate frozen inputs and compare two independent panel builds."""

import argparse
from pathlib import Path

from michelin.data.validation import validate_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = validate_snapshot(args.snapshot, args.output_dir)
    print(report.model_dump_json(indent=2))
    raise SystemExit(0 if report.phase_1_ready else 1)


if __name__ == "__main__":
    main()

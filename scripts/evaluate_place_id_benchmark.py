"""Run the place_id benchmark gate offline and report TUNE vs HELD-OUT side by side.

Refuses to run while any label is PENDING or any fixture is missing — it never fabricates a
pass. Held-out labels are read only here, at evaluation time, after tuning is frozen.
"""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from restaurant_ingestion.benchmark.evaluate import (
    evaluate,
    gate_report,
    load_fixture_queries,  # noqa: F401  (kept importable for ad-hoc use)
    load_labels,
    load_manifest,
    pending_or_missing,
    slice_metrics,
)

BENCHMARK_DIR = ROOT / "tests" / "benchmark"
MANIFEST_CSV = BENCHMARK_DIR / "place_id_benchmark.csv"
TUNE_LABELS_CSV = BENCHMARK_DIR / "labels.tune.csv"
HELDOUT_LABELS_CSV = BENCHMARK_DIR / "labels.heldout.csv"
FIXTURES_DIR = BENCHMARK_DIR / "fixtures"


def main() -> int:
    manifest_rows = load_manifest(MANIFEST_CSV)
    labels = {**load_labels(TUNE_LABELS_CSV), **load_labels(HELDOUT_LABELS_CSV)}

    problems = pending_or_missing(manifest_rows, labels, FIXTURES_DIR)
    if problems:
        print("BLOCKED: benchmark is not ready to evaluate (labels PENDING or fixtures missing).")
        print("Resolve these before a verdict means anything:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    evaluations = evaluate(manifest_rows, labels, FIXTURES_DIR)
    tune = tuple(row for row in evaluations if row.slice == "tune")
    heldout = tuple(row for row in evaluations if row.slice == "heldout")

    report = {
        "tune": {"metrics": slice_metrics(tune), "gate": gate_report(tune)},
        "heldout": {"metrics": slice_metrics(heldout), "gate": gate_report(heldout)},
        "rows": [
            {"row_id": row.row_id, "slice": row.slice, "category": row.category, "verdict": row.verdict,
             "expected": row.expected_outcome, "predicted": row.predicted_outcome}
            for row in evaluations
        ],
    }
    print(json.dumps(report, indent=2))
    heldout_gate_passes = all(report["heldout"]["gate"].values())
    return 0 if heldout_gate_passes else 1


if __name__ == "__main__":
    sys.exit(main())

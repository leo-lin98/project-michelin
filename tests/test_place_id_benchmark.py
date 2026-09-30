"""Tests for the benchmark evaluation *machinery* (not a proof of matching accuracy).

These verify that classify/metrics/gate and the deterministic wiring behave correctly given
known inputs. The actual place_id proof requires fetched fixtures + sealed labels and is run
via scripts/evaluate_place_id_benchmark.py once those exist.
"""

from pathlib import Path
import csv
import json

from restaurant_ingestion.benchmark.evaluate import (
    RowEvaluation,
    classify_row,
    deterministic_decision_for_row,
    evaluate,
    gate_report,
    pending_or_missing,
    slice_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_CSV = ROOT / "tests" / "benchmark" / "place_id_benchmark.csv"


def test_classify_row_covers_every_outcome() -> None:
    assert classify_row("place_id", "p1", "place_id", "p1") == "correct_resolve"
    assert classify_row("place_id", "p1", "place_id", "p2") == "wrong_id_resolve"
    assert classify_row("place_id", "p1", "manual_review", "") == "missed_resolve"
    assert classify_row("manual_review", "", "place_id", "p2") == "over_match"
    assert classify_row("manual_review", "", "manual_review", "") == "correct_defer"
    assert classify_row("no_safe_match", "", "manual_review", "") == "correct_defer"
    assert classify_row("no_safe_match", "", "place_id", "p9") == "over_match"


def test_slice_metrics_counts_precision_and_recall() -> None:
    evaluations = (
        _evaluation("a", "place_id", verdict="correct_resolve", predicted="place_id"),
        _evaluation("b", "place_id", verdict="missed_resolve", predicted="manual_review"),
        _evaluation("c", "no_safe_match", verdict="over_match", predicted="place_id"),
        _evaluation("d", "manual_review", verdict="correct_defer", predicted="manual_review"),
    )

    metrics = slice_metrics(evaluations)

    assert metrics["false_positives"] == 1  # the over_match
    assert metrics["false_negatives"] == 1  # the missed_resolve
    assert metrics["precision"] == 0.5  # 1 correct / 2 predicted resolves
    assert metrics["recall"] == 0.5  # 1 correct / 2 expected resolves


def test_gate_flags_same_building_collision_and_regression() -> None:
    collide = (
        _evaluation("x", "place_id", category="same_building", verdict="correct_resolve", predicted="place_id", predicted_id="shared"),
        _evaluation("y", "place_id", category="same_building", verdict="wrong_id_resolve", predicted="place_id", predicted_id="shared"),
    )
    gate = gate_report(collide)
    assert gate["same_building_distinct"] is False
    assert gate["no_duplicate_place_id"] is False

    clean = (
        _evaluation("x", "place_id", category="same_building", verdict="correct_resolve", predicted="place_id", predicted_id="a"),
        _evaluation("y", "place_id", category="same_building", verdict="correct_resolve", predicted="place_id", predicted_id="b"),
    )
    assert gate_report(clean)["same_building_distinct"] is True


def test_deterministic_decision_wiring_resolves_strong_match() -> None:
    manifest_row = {
        "name": "Taben",
        "address": "28 Jilin Road, Zhongshan District, Taipei, Taiwan",
        "latitude": "25.0",
        "longitude": "121.0",
        "phone_number": "",
        "website_url": "https://taben.example",
        "award": "1 Star",
        "michelin_url": "",
    }
    fixture_queries = [
        {
            "response": {
                "places": [
                    {
                        "id": "place-taben",
                        "displayName": {"text": "Taben"},
                        "formattedAddress": "28 Jilin Road, Zhongshan District, Taipei, Taiwan",
                        "location": {"latitude": 25.0, "longitude": 121.0},
                        "websiteUri": "https://taben.example",
                        "primaryType": "restaurant",
                    }
                ]
            }
        }
    ]

    outcome, place_id = deterministic_decision_for_row(manifest_row, fixture_queries)

    assert outcome == "place_id"
    assert place_id == "place-taben"


def test_pending_or_missing_blocks_until_ready(tmp_path: Path) -> None:
    manifest_rows = [{"row_id": "01-x", "name": "X", "category": "resolved_clean", "slice": "tune"}]
    labels_pending = {"01-x": {"row_id": "01-x", "expected_outcome": "PENDING_FIXTURE", "expected_place_id": ""}}
    assert pending_or_missing(manifest_rows, labels_pending, tmp_path)  # PENDING + no fixture

    labels_ready = {"01-x": {"row_id": "01-x", "expected_outcome": "manual_review", "expected_place_id": ""}}
    (tmp_path / "01-x.json").write_text(json.dumps({"row_id": "01-x", "queries": []}), encoding="utf-8")
    assert pending_or_missing(manifest_rows, labels_ready, tmp_path) == []


def test_evaluate_end_to_end_against_synthetic_fixture(tmp_path: Path) -> None:
    manifest_rows = [
        {
            "row_id": "01-taben",
            "name": "Taben",
            "category": "resolved_clean",
            "slice": "tune",
            "address": "28 Jilin Road, Zhongshan District, Taipei, Taiwan",
            "latitude": "25.0",
            "longitude": "121.0",
            "phone_number": "",
            "website_url": "https://taben.example",
            "award": "1 Star",
            "michelin_url": "",
        }
    ]
    labels = {"01-taben": {"row_id": "01-taben", "expected_outcome": "place_id", "expected_place_id": "place-taben"}}
    (tmp_path / "01-taben.json").write_text(
        json.dumps(
            {
                "row_id": "01-taben",
                "queries": [
                    {
                        "response": {
                            "places": [
                                {
                                    "id": "place-taben",
                                    "displayName": {"text": "Taben"},
                                    "formattedAddress": "28 Jilin Road, Zhongshan District, Taipei, Taiwan",
                                    "location": {"latitude": 25.0, "longitude": 121.0},
                                    "websiteUri": "https://taben.example",
                                    "primaryType": "restaurant",
                                }
                            ]
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    evaluations = evaluate(manifest_rows, labels, tmp_path)

    assert len(evaluations) == 1
    assert evaluations[0].verdict == "correct_resolve"
    assert gate_report(evaluations)["no_resolved_clean_regression"] is True


def test_manifest_artifact_is_well_formed() -> None:
    with MANIFEST_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 20
    assert sum(1 for row in rows if row["slice"] == "tune") == 10
    assert sum(1 for row in rows if row["slice"] == "heldout") == 10
    assert len({row["row_id"] for row in rows}) == 20


def _evaluation(
    row_id: str,
    expected_outcome: str,
    verdict: str,
    predicted: str,
    category: str = "base_unresolved",
    predicted_id: str = "",
) -> RowEvaluation:
    return RowEvaluation(
        row_id=row_id,
        category=category,
        slice="tune",
        expected_outcome=expected_outcome,
        expected_place_id="p1" if expected_outcome == "place_id" else "",
        predicted_outcome=predicted,
        predicted_place_id=predicted_id,
        verdict=verdict,
    )

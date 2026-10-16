"""Replay frozen benchmark fixtures through the deterministic matcher and score outcomes.

This proves the SCORING layer only. The LLM tier is live and non-deterministic, so it is out
of scope here: the deterministic scorer's correct behaviour for both ``manual_review`` and
``no_safe_match`` labels is to defer (return ``manual_review``), and that is what the gate
checks. ``classify_row`` / ``slice_metrics`` / ``gate_report`` are pure so they can be unit
tested without any fixtures or network.
"""

from dataclasses import dataclass
import csv
import json
from pathlib import Path

DEFER_OUTCOMES = ("manual_review", "no_safe_match")
PENDING_LABEL = "PENDING_FIXTURE"
RESOLVED_CLEAN_CATEGORY = "resolved_clean"
SAME_BUILDING_CATEGORY = "same_building"


@dataclass(frozen=True)
class RowEvaluation:
    row_id: str
    category: str
    slice: str
    expected_outcome: str
    expected_place_id: str
    predicted_outcome: str
    predicted_place_id: str
    verdict: str


def scoring_row_from_manifest(manifest_row: dict[str, str]) -> dict[str, object]:
    return {
        "name": manifest_row["name"],
        "address": manifest_row["address"],
        "latitude": manifest_row["latitude"],
        "longitude": manifest_row["longitude"],
        "phone_number": manifest_row.get("phone_number", ""),
        "website_url": manifest_row.get("website_url", ""),
        "michelin_category": manifest_row.get("award", ""),
        "michelin_url": manifest_row.get("michelin_url", ""),
    }


def deterministic_decision_for_row(manifest_row: dict[str, str], fixture_queries: list[dict[str, object]]) -> tuple[str, str]:
    from restaurant_ingestion.ingestion.place_id_scoring import decide_place_id_match

    decision = decide_place_id_match(scoring_row_from_manifest(manifest_row), {"queries": fixture_queries})
    return decision.outcome, decision.place_id


def classify_row(
    expected_outcome: str,
    expected_place_id: str,
    predicted_outcome: str,
    predicted_place_id: str,
) -> str:
    predicted_resolves = predicted_outcome == "place_id"
    if expected_outcome == "place_id":
        if not predicted_resolves:
            return "missed_resolve"
        if predicted_place_id == expected_place_id:
            return "correct_resolve"
        return "wrong_id_resolve"
    if expected_outcome not in DEFER_OUTCOMES:
        raise ValueError(f"Unsupported expected_outcome: {expected_outcome}")
    if predicted_resolves:
        return "over_match"
    return "correct_defer"


def slice_metrics(evaluations: tuple[RowEvaluation, ...]) -> dict[str, object]:
    verdicts = [row.verdict for row in evaluations]
    correct_resolve = verdicts.count("correct_resolve")
    wrong_id_resolve = verdicts.count("wrong_id_resolve")
    over_match = verdicts.count("over_match")
    missed_resolve = verdicts.count("missed_resolve")
    correct_defer = verdicts.count("correct_defer")
    predicted_resolves = correct_resolve + wrong_id_resolve + over_match
    expected_resolves = correct_resolve + wrong_id_resolve + missed_resolve
    return {
        "rows": len(evaluations),
        "correct_resolve": correct_resolve,
        "wrong_id_resolve": wrong_id_resolve,
        "over_match": over_match,
        "missed_resolve": missed_resolve,
        "correct_defer": correct_defer,
        "false_positives": wrong_id_resolve + over_match,
        "false_negatives": wrong_id_resolve + missed_resolve,
        "precision": correct_resolve / predicted_resolves if predicted_resolves else None,
        "recall": correct_resolve / expected_resolves if expected_resolves else None,
    }


def gate_report(evaluations: tuple[RowEvaluation, ...]) -> dict[str, bool]:
    resolved_place_ids = [row.predicted_place_id for row in evaluations if row.predicted_outcome == "place_id"]
    expected_resolve = [row for row in evaluations if row.expected_outcome == "place_id"]
    expected_defer = [row for row in evaluations if row.expected_outcome in DEFER_OUTCOMES]
    resolved_clean = [row for row in evaluations if row.category == RESOLVED_CLEAN_CATEGORY]
    same_building_ids = [
        row.predicted_place_id
        for row in evaluations
        if row.category == SAME_BUILDING_CATEGORY and row.predicted_outcome == "place_id"
    ]
    return {
        "expected_resolves_all_correct": all(row.verdict == "correct_resolve" for row in expected_resolve),
        "expected_defers_all_held": all(row.verdict == "correct_defer" for row in expected_defer),
        "no_resolved_clean_regression": all(row.verdict == "correct_resolve" for row in resolved_clean),
        "no_duplicate_place_id": len(resolved_place_ids) == len(set(resolved_place_ids)),
        "same_building_distinct": len(same_building_ids) == len(set(same_building_ids)),
    }


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_labels(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["row_id"]: row for row in csv.DictReader(handle)}


def load_fixture_queries(path: Path) -> list[dict[str, object]]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    queries = fixture["queries"]
    if not isinstance(queries, list):
        raise ValueError(f"Fixture queries must be a list: path={path}")
    return queries


def pending_or_missing(manifest_rows: list[dict[str, str]], labels: dict[str, dict[str, str]], fixtures_dir: Path) -> list[str]:
    problems: list[str] = []
    for row in manifest_rows:
        row_id = row["row_id"]
        label = labels.get(row_id)
        if label is None:
            problems.append(f"{row_id}: no label")
            continue
        if label["expected_outcome"] == PENDING_LABEL or label["expected_outcome"] == "":
            problems.append(f"{row_id}: label is {PENDING_LABEL}")
        if not (fixtures_dir / f"{row_id}.json").exists():
            problems.append(f"{row_id}: fixture missing")
    return problems


def evaluate(manifest_rows: list[dict[str, str]], labels: dict[str, dict[str, str]], fixtures_dir: Path) -> tuple[RowEvaluation, ...]:
    evaluations: list[RowEvaluation] = []
    for row in manifest_rows:
        label = labels[row["row_id"]]
        fixture_queries = load_fixture_queries(fixtures_dir / f"{row['row_id']}.json")
        predicted_outcome, predicted_place_id = deterministic_decision_for_row(row, fixture_queries)
        verdict = classify_row(
            label["expected_outcome"],
            label["expected_place_id"],
            predicted_outcome,
            predicted_place_id,
        )
        evaluations.append(
            RowEvaluation(
                row_id=row["row_id"],
                category=row["category"],
                slice=row["slice"],
                expected_outcome=label["expected_outcome"],
                expected_place_id=label["expected_place_id"],
                predicted_outcome=predicted_outcome,
                predicted_place_id=predicted_place_id,
                verdict=verdict,
            )
        )
    return tuple(evaluations)

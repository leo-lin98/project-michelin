"""Evaluate fixed LR on saved nested-GBM outer folds without tuning or refits."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Annotated

import pandas as pd
from pydantic import BaseModel, ConfigDict, StrictStr, StringConstraints, TypeAdapter, ValidationError

from michelin.config import AppConfig, load_config
from michelin.model.dataset import ClassificationDataset, DatasetSplit, assemble_classification_dataset, stratified_folds
from michelin.model.evaluate import CalibrationReport, EvaluationReport, evaluate_calibration
from michelin.model.settings import load_modeling_settings
from michelin.model.train import evaluate_dataset_rows, fit_explanation, predict_probabilities


ROOT = Path(__file__).resolve().parents[1]
PANEL_KEY = "data/processed/labeled_restaurants.csv"
FEATURES_KEY = "data/processed/phase2-price-range-2026-10-13/shared_features.csv"
HISTORICAL_SOURCE_KEYS = (
    "config/modeling.yaml", "src/michelin/model/train.py", "pyproject.toml", "uv.lock",
)
CURRENT_SOURCE_KEYS = (
    "src/michelin/__init__.py", "src/michelin/model/__init__.py", "src/michelin/features/__init__.py",
    "src/michelin/config.py", "src/michelin/model/dataset.py", "src/michelin/model/train.py",
    "src/michelin/model/settings.py", "src/michelin/model/evaluate.py",
    "src/michelin/model/thresholds.py", "src/michelin/features/transforms.py",
    "config/modeling.yaml", "pyproject.toml", "uv.lock", "scripts/evaluate_logistic_cv.py",
)
LIMITATIONS = (
    "Fixed LR outer-fold OOF CV: no inner selection, tuning, class weighting, or added calibration.",
    "Saved GBM uses nested Platt calibration; its saved scores are evaluated again without refitting.",
    "All-population OOF is not a new untouched holdout; the earlier holdout has already been inspected.",
    "Few positive examples and Taipei/New Taipei geography limit precision and geographic generalization.",
    "The bounded sampled restaurant population does not establish population-wide prevalence or performance.",
    "Display rows are excluded from fitting, scoring, and evaluation.",
    "Additional current source hashes are current-run provenance, not proof of historical GBM source versions.",
)


@dataclass(frozen=True)
class LogisticCVResult:
    probabilities: pd.Series
    folds: tuple[DatasetSplit, ...]
    evaluation: EvaluationReport
    calibration: CalibrationReport
    fold_evaluations: tuple[EvaluationReport, ...]


class ReferenceFold(BaseModel):
    """Validate the external saved nested-CV partition at the JSON boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    validation_ids: list[StrictStr]
    train_ids: list[StrictStr]
    calibration_ids: list[StrictStr]


def run_logistic_cv(dataset: ClassificationDataset, config: AppConfig) -> LogisticCVResult:
    """Fit unchanged LR independently on each shared outer training partition."""
    if config.pipeline.split.cv_folds != 5:
        raise ValueError("Logistic comparison requires exactly five outer folds")
    cutoff = config.pipeline.threshold.similar_to_starred_cutoff
    if cutoff != 0.2:  #experimentation: fixed cutoff 20% (previously 50%).
        raise ValueError("Logistic comparison requires fixed inclusive cutoff 0.2")  #experimentation: fixed cutoff 20% (previously 50%).
    identities = tuple(sorted(dataset.labels.index))
    folds = stratified_folds(dataset, identities, config.pipeline)
    scores = tuple(
        predict_probabilities(fit_explanation(dataset, fold.train_ids, config), dataset.features.loc[list(fold.test_ids)])
        for fold in folds
    )
    probabilities = pd.concat(scores).sort_index()
    if not probabilities.index.is_unique or tuple(probabilities.index) != identities:
        raise ValueError("Outer-fold probabilities must cover every modeling identity exactly once")
    settings = load_modeling_settings(ROOT / "config/modeling.yaml")
    return LogisticCVResult(
        probabilities=probabilities,
        folds=folds,
        evaluation=evaluate_dataset_rows(dataset, probabilities, cutoff),
        calibration=evaluate_calibration(dataset.labels.loc[probabilities.index], probabilities, settings.calibration_bins),
        fold_evaluations=tuple(evaluate_dataset_rows(dataset, score, cutoff) for score in scores),
    )


def read_bounded_csv(path: Path, row_limit: int) -> pd.DataFrame:
    """Read only the accepted bound plus one overflow sentinel row."""
    frame = pd.read_csv(path, nrows=row_limit + 1, float_precision="round_trip")
    if len(frame) > row_limit:
        raise ValueError(f"CSV at {path} exceeds row_limit={row_limit}")
    return frame


def file_hash(path: Path) -> str:
    """Hash a known file in a stream without loading it into memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_reference_hashes(reference: Path, inputs: dict[str, Path]) -> dict[str, str]:
    """Check only explicit known keys; never open paths supplied by a manifest."""
    digest_type = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]
    try:
        manifest = TypeAdapter(dict[StrictStr, digest_type]).validate_json((reference / "hashes.json").read_bytes())
    except ValidationError as exc:
        raise ValueError("Invalid saved GBM hash manifest JSON") from exc
    known = {**inputs, **{key: ROOT / key for key in HISTORICAL_SOURCE_KEYS}}
    current = {key: file_hash(path) for key, path in known.items()}
    for key, digest in current.items():
        if key not in manifest:
            raise ValueError(f"Saved GBM hash manifest is missing required hash: {key}")
        if manifest[key] != digest:
            raise ValueError(f"Saved GBM hash mismatch for {key}: expected={manifest[key]}, current={digest}")
    return current


def verify_reference_folds(reference: Path, folds: tuple[DatasetSplit, ...]) -> None:
    """Require ordered saved training and validation identities to match exactly."""
    try:
        saved = TypeAdapter(list[ReferenceFold]).validate_json((reference / "run1/oof_provenance.json").read_bytes())
    except ValidationError as exc:
        raise ValueError("Invalid saved GBM fold provenance JSON") from exc
    if len(saved) != len(folds):
        raise ValueError("Saved GBM outer fold count does not match current folds")
    for number, (expected, actual) in enumerate(zip(folds, saved, strict=True), start=1):
        if tuple(actual.train_ids) != expected.train_ids or tuple(actual.validation_ids) != expected.test_ids:
            raise ValueError(f"Saved GBM ordered train/validation identities differ in outer fold {number}")
        if actual.calibration_ids != actual.train_ids:
            raise ValueError(f"Saved GBM calibration identities differ from training in outer fold {number}")


def read_reference_scores(reference: Path, dataset: ClassificationDataset, row_limit: int) -> pd.Series:
    """Require a finite probability for precisely each modeling identity."""
    frame = read_bounded_csv(reference / "run1/oof_probabilities.csv", row_limit)
    if list(frame.columns) != ["restaurant_id", "probability"]:
        raise ValueError("Saved GBM score columns must be restaurant_id,probability")
    scores = frame.set_index("restaurant_id")["probability"]
    if set(scores.index) != set(dataset.labels.index):
        raise ValueError("Saved GBM score identities must exactly cover the modeling population")
    try:
        evaluate_dataset_rows(dataset, scores, 0.2)  #experimentation: fixed cutoff 20% (previously 50%).
    except ValueError as exc:
        raise ValueError(f"Invalid saved GBM scores at {reference / 'run1/oof_probabilities.csv'}: {exc}") from exc
    return scores.loc[dataset.labels.index].copy(deep=True)


def write_json(path: Path, payload: object) -> None:
    """Serialize the bounded report, rejecting nonfinite numbers explicitly."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str]) -> None:
    """Validate bounded local evidence before fitting and write a fresh comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "features", "pipeline-config", "features-config", "gbm-evidence-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--row-limit", type=int, required=True)
    args = parser.parse_args(argv)
    if args.row_limit < 1:
        raise ValueError("row_limit must be a positive integer")
    if args.output_dir.exists():
        raise FileExistsError(f"Output evidence directory already exists: {args.output_dir}")
    panel = read_bounded_csv(args.panel, args.row_limit)
    features = read_bounded_csv(args.features, args.row_limit).set_index("restaurant_id")
    config = load_config(args.pipeline_config, args.features_config)
    dataset = assemble_classification_dataset(panel, features, config)
    inputs = {
        PANEL_KEY: args.panel, FEATURES_KEY: args.features,
        "config/pipeline.yaml": args.pipeline_config, "config/features.yaml": args.features_config,
    }
    historical_hashes = verify_reference_hashes(args.gbm_evidence_dir, inputs)
    folds = stratified_folds(dataset, tuple(dataset.labels.index), config.pipeline)
    verify_reference_folds(args.gbm_evidence_dir, folds)
    gbm_scores = read_reference_scores(args.gbm_evidence_dir, dataset, args.row_limit)
    settings = load_modeling_settings(ROOT / "config/modeling.yaml")
    gbm_evaluation = evaluate_dataset_rows(dataset, gbm_scores, 0.2)  #experimentation: fixed cutoff 20% (previously 50%).
    gbm_calibration = evaluate_calibration(dataset.labels, gbm_scores, settings.calibration_bins)
    result = run_logistic_cv(dataset, config)
    current_hashes = {
        **{key: file_hash(path) for key, path in inputs.items()},
        **{key: file_hash(ROOT / key) for key in CURRENT_SOURCE_KEYS},
    }
    reference_hashes = {
        name: file_hash(args.gbm_evidence_dir / name)
        for name in ("hashes.json", "run1/oof_provenance.json", "run1/oof_probabilities.csv")
    }
    rows = {"modeling": len(dataset.labels), "positive": int(dataset.labels.sum()), "display_excluded": len(dataset.display_features)}
    provenance = {
        "seed": config.pipeline.seed, "cutoff": 0.2, "outer_folds": 5,  #experimentation: fixed cutoff 20% (previously 50%).
        "model_settings": settings.model_dump(), "pipeline_config": config.pipeline.model_dump(mode="json"),
        "features_config": config.features.model_dump(mode="json"), "inner_selection": False,
        "added_calibration": False, "display_excluded": True, "rows": rows,
        "historical_hashes_verified": historical_hashes, "current_hashes": current_hashes,
        "reference_hashes": reference_hashes, "limitations": LIMITATIONS,
        "folds": [{"train_ids": fold.train_ids, "validation_ids": fold.test_ids} for fold in result.folds],
    }
    validation = {
        "cutoff": 0.2, "precision_k": int(dataset.labels.sum()), "rows": rows,  #experimentation: fixed cutoff 20% (previously 50%).
        "evaluation": asdict(result.evaluation), "calibration": asdict(result.calibration),
        "folds": [
            {
                "fold": number, "rows": len(fold.test_ids), "precision_k": report.known_positive_count,
                "evaluation": asdict(report),
                "calibration": asdict(evaluate_calibration(dataset.labels.loc[list(fold.test_ids)], result.probabilities.loc[list(fold.test_ids)], settings.calibration_bins)),
            }
            for number, (fold, report) in enumerate(zip(result.folds, result.fold_evaluations, strict=True), start=1)
        ],
    }
    comparison = {
        "cutoff": 0.2, "precision_k": int(dataset.labels.sum()), "rows": rows,  #experimentation: fixed cutoff 20% (previously 50%).
        "logistic_regression": {"evaluation": asdict(result.evaluation), "brier_score": result.calibration.brier_score},
        "saved_nested_gbm": {"evaluation": asdict(gbm_evaluation), "brier_score": gbm_calibration.brier_score},
        "limitations": LIMITATIONS,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result.probabilities.rename_axis("restaurant_id").to_csv(args.output_dir / "probabilities.csv")
    write_json(args.output_dir / "provenance.json", provenance)
    write_json(args.output_dir / "validation.json", validation)
    write_json(args.output_dir / "comparison.json", comparison)


if __name__ == "__main__":
    main(sys.argv[1:])

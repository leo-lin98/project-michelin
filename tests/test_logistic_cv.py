"""CLI safety checks derived from the comparison evidence contract."""

import json
from pathlib import Path
import shutil

import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss

from michelin.model.dataset import ClassificationDataset, DatasetSplit
from michelin.config import AppConfig
from scripts import evaluate_logistic_cv as candidate
from test_model_acceptance import config, modeling_case


PANEL = Path("data/processed/labeled_restaurants.csv")
FEATURES = Path("data/processed/phase2-price-range-2026-10-13/shared_features.csv")
REFERENCE = Path("data/processed/modeling-phase6/source-integration")


def arguments(reference: Path, output: Path, row_limit: int) -> list[str]:
    return [
        "--panel", str(PANEL), "--features", str(FEATURES),
        "--pipeline-config", "config/pipeline.yaml", "--features-config", "config/features.yaml",
        "--gbm-evidence-dir", str(reference), "--output-dir", str(output), "--row-limit", str(row_limit),
    ]


@pytest.fixture
def reference(tmp_path: Path) -> Path:
    if not PANEL.is_file() or not FEATURES.is_file() or not REFERENCE.is_dir():
        pytest.skip("Local-only modeling datasets or GBM reference evidence are unavailable")
    destination = tmp_path / "reference"
    shutil.copytree(REFERENCE, destination)
    #experimentation start: reference fixture for the fixed 20% protocol
    manifest_path = destination / "hashes.json"
    manifest = candidate.TypeAdapter(dict[str, str]).validate_json(manifest_path.read_bytes())
    manifest_path.write_text(json.dumps({
        **manifest,
        "config/pipeline.yaml": candidate.file_hash(Path("config/pipeline.yaml")),
        "src/michelin/model/train.py": candidate.file_hash(Path("src/michelin/model/train.py")),
    }), encoding="utf-8")
    #experimentation end: reference fixture for the fixed 20% protocol
    return destination


@pytest.fixture
def forbid_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_fit(*args: object) -> None:
        pytest.fail("Invalid CLI inputs reached model fitting")

    monkeypatch.setattr(candidate, "fit_explanation", unexpected_fit)


@pytest.mark.skipif(not PANEL.is_file(), reason="Local-only labeled restaurant dataset is unavailable")
def test_bounded_reads_reject_overflow_before_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forbid_fit: None,
) -> None:
    original = pd.read_csv
    calls: list[tuple[int, str]] = []

    def capture_read(path: Path, *, nrows: int, float_precision: str) -> pd.DataFrame:
        calls.append((nrows, float_precision))
        return original(path, nrows=nrows, float_precision=float_precision)

    monkeypatch.setattr(candidate.pd, "read_csv", capture_read)
    with pytest.raises(ValueError, match="row.limit"):
        candidate.main(arguments(REFERENCE, tmp_path / "output", 2))
    assert calls and all(call == (3, "round_trip") for call in calls)
    assert not (tmp_path / "output").exists()


def test_existing_output_is_protected_before_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forbid_fit: None,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("original", encoding="utf-8")

    def unexpected_read(*args: object, **kwargs: object) -> None:
        pytest.fail("Existing evidence should fail before input reads")

    monkeypatch.setattr(candidate.pd, "read_csv", unexpected_read)
    with pytest.raises(FileExistsError, match="output|evidence"):
        candidate.main(arguments(REFERENCE, output, 3000))
    assert sentinel.read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize("row_limit", [0, -1])
def test_invalid_row_limit_is_explicit(tmp_path: Path, row_limit: int, forbid_fit: None) -> None:
    with pytest.raises(ValueError, match="row.limit"):
        candidate.main(arguments(REFERENCE, tmp_path / "output", row_limit))


@pytest.mark.parametrize("damage", ["invalid_json", "missing_hash", "mismatch", "wrong_type"])
def test_manifest_errors_fail_before_fit(
    tmp_path: Path, reference: Path, damage: str, forbid_fit: None,
) -> None:
    path = reference / "hashes.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if damage == "missing_hash":
        del manifest[str(PANEL)]
    elif damage == "mismatch":
        manifest[str(PANEL)] = "0" * 64
    elif damage == "wrong_type":
        manifest[str(PANEL)] = 3
    path.write_text("{" if damage == "invalid_json" else json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest|hash|JSON"):
        candidate.main(arguments(reference, tmp_path / "output", 3000))
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("damage", ["reordered_train", "reordered_validation", "missing_fold", "invalid_type"])
def test_reference_folds_must_match_ordered_partition(
    tmp_path: Path, reference: Path, damage: str, forbid_fit: None,
) -> None:
    path = reference / "run1/oof_provenance.json"
    folds = json.loads(path.read_text(encoding="utf-8"))
    if damage == "reordered_train":
        folds[0]["train_ids"].reverse()
    elif damage == "reordered_validation":
        folds[0]["validation_ids"].reverse()
    elif damage == "missing_fold":
        folds.pop()
    else:
        folds[0]["validation_ids"][0] = 7
    path.write_text(json.dumps(folds), encoding="utf-8")
    with pytest.raises(ValueError, match="fold|provenance|validation"):
        candidate.main(arguments(reference, tmp_path / "output", 3000))
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("damage", ["duplicate", "missing", "display", "nan", "infinity", "negative", "over_one", "text", "wrong_columns"])
def test_reference_scores_require_exact_finite_modeling_population(
    tmp_path: Path, reference: Path, damage: str, forbid_fit: None,
) -> None:
    path = reference / "run1/oof_probabilities.csv"
    scores = pd.read_csv(path, nrows=3001, float_precision="round_trip")
    if damage == "duplicate":
        scores.loc[1, "restaurant_id"] = scores.loc[0, "restaurant_id"]
    elif damage == "missing":
        scores = scores.iloc[1:]
    elif damage == "display":
        scores.loc[0, "restaurant_id"] = "not-a-modeling-id"
    elif damage == "wrong_columns":
        scores = scores.rename(columns={"probability": "unexpected"})
    else:
        values = {"nan": "NaN", "infinity": "inf", "negative": "-0.1", "over_one": "1.1", "text": "bad"}
        scores = scores.assign(probability=scores["probability"].astype(str))
        scores.loc[0, "probability"] = values[damage]
    scores.to_csv(path, index=False)
    with pytest.raises(ValueError, match="score|probabilit|identit"):
        candidate.main(arguments(reference, tmp_path / "output", 3000))
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("setting,value", [("cv_folds", 4), ("similar_to_starred_cutoff", 0.4)])
def test_fixed_protocol_rejects_changed_settings_before_fit(
    modeling_case: tuple[ClassificationDataset, DatasetSplit], config: AppConfig,
    setting: str, value: int | float, forbid_fit: None,
) -> None:
    pipeline = config.pipeline.model_dump()
    section = "split" if setting == "cv_folds" else "threshold"
    pipeline[section][setting] = value
    changed = AppConfig.model_validate({"pipeline": pipeline, "features": config.features.model_dump()})
    with pytest.raises(ValueError, match="five outer folds|cutoff 0.2"):  #experimentation: fixed cutoff 20%.
        candidate.run_logistic_cv(modeling_case[0], changed)


def test_cli_writes_recomputed_evidence_and_ignores_external_manifest_paths(
    tmp_path: Path, reference: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden_path = tmp_path / "unapproved-input.txt"
    manifest_path = reference / "hashes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[str(forbidden_path)] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (reference / "run1/validation.json").write_text('{"oof":{"average_precision":0.0}}', encoding="utf-8")
    original_open = Path.open

    def approved_open(path: Path, *args: object, **kwargs: object):
        assert path != forbidden_path, "Runner opened an arbitrary manifest key"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", approved_open)
    output = tmp_path / "output"
    candidate.main(arguments(reference, output, 3000))
    assert {path.name for path in output.iterdir()} == {"probabilities.csv", "provenance.json", "validation.json", "comparison.json"}
    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    panel = pd.read_csv(PANEL, nrows=3001, float_precision="round_trip").set_index("restaurant_id")
    gbm = pd.read_csv(reference / "run1/oof_probabilities.csv", nrows=3001, float_precision="round_trip").set_index("restaurant_id")["probability"]
    labels = panel.loc[gbm.index, "class"]
    lr = pd.read_csv(output / "probabilities.csv", nrows=3001, float_precision="round_trip").set_index("restaurant_id")["probability"]
    assert len(lr) == len(gbm) == comparison["rows"]["modeling"] == 1192
    assert comparison["precision_k"] == 43
    assert comparison["cutoff"] == 0.2  #experimentation: fixed cutoff 20%.
    assert comparison["saved_nested_gbm"]["evaluation"]["metrics"]["average_precision"] == pytest.approx(average_precision_score(labels, gbm))
    assert comparison["saved_nested_gbm"]["brier_score"] == pytest.approx(brier_score_loss(labels, gbm))
    assert validation["evaluation"] == comparison["logistic_regression"]["evaluation"]
    assert len(validation["folds"]) == len(provenance["folds"]) == 5
    assert sum(fold["rows"] for fold in validation["folds"]) == len(lr)
    assert all("calibration" in fold and "evaluation" in fold for fold in validation["folds"])
    assert provenance["inner_selection"] is provenance["added_calibration"] is False
    assert provenance["display_excluded"] is True
    assert set(candidate.CURRENT_SOURCE_KEYS).issubset(provenance["current_hashes"])
    assert str(forbidden_path) not in provenance["current_hashes"]

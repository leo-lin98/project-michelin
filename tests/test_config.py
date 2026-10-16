from pathlib import Path

import pandas as pd
import pytest
import yaml

from michelin.config import Award, load_config
from michelin.features.build import build_shared_features
from michelin.features.diagnostics import missingness_tests
from michelin.features.transforms import fit_transforms


def test_load_config_validates_pipeline_and_feature_yaml() -> None:
    config = load_config(Path("config/pipeline.yaml"), Path("config/features.yaml"))

    assert config.pipeline.project.geography == ("Taipei", "New Taipei")
    assert config.pipeline.labels.starred_awards == (
        Award.ONE_STAR,
        Award.TWO_STARS,
        Award.THREE_STARS,
    )
    assert config.features.leakage_controls.require_feature_provenance_parity is True
    assert config.features.feature_schema.numeric == ("average_rating", "review_count", "neighborhood_density")
    assert config.features.feature_schema.categorical == ("price_range", "cuisine", "city")


def test_load_config_raises_on_missing_required_key(tmp_path: Path) -> None:
    pipeline_path = tmp_path / "pipeline.yaml"
    features_path = tmp_path / "features.yaml"
    pipeline_path.write_text("seed: 1\n", encoding="utf-8")
    features_path.write_text(Path("config/features.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid Michelin pipeline configuration"):
        load_config(pipeline_path, features_path)


def test_historical_numeric_config_parses_but_cannot_execute_phase_two(tmp_path: Path) -> None:
    document = yaml.safe_load(Path("config/features.yaml").read_text(encoding="utf-8"))
    document["feature_schema"]["numeric"] = ["price_level", "average_rating", "review_count", "neighborhood_density"]
    document["feature_schema"]["categorical"] = ["cuisine", "city"]
    path = tmp_path / "historical_features.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    config = load_config(Path("config/pipeline.yaml"), path)
    assert config.features.feature_schema.numeric[0] == "price_level"
    with pytest.raises(ValueError, match="categorical price_range"):
        fit_transforms(pd.DataFrame(), config.features)
    with pytest.raises(ValueError, match="categorical price_range"):
        build_shared_features(pd.DataFrame(), pd.Series(dtype=float), config)
    with pytest.raises(ValueError, match="categorical price_range"):
        missingness_tests(pd.DataFrame(), config)

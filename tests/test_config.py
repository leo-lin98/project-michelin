from pathlib import Path

import pytest

from michelin.config import Award, load_config


def test_load_config_validates_pipeline_and_feature_yaml() -> None:
    config = load_config(Path("config/pipeline.yaml"), Path("config/features.yaml"))

    assert config.pipeline.project.geography == "Taipei"
    assert config.pipeline.labels.starred_awards == (
        Award.ONE_STAR,
        Award.TWO_STARS,
        Award.THREE_STARS,
    )
    assert config.features.leakage_controls.require_feature_provenance_parity is True


def test_load_config_raises_on_missing_required_key(tmp_path: Path) -> None:
    pipeline_path = tmp_path / "pipeline.yaml"
    features_path = tmp_path / "features.yaml"
    pipeline_path.write_text("seed: 1\n", encoding="utf-8")
    features_path.write_text(Path("config/features.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid Michelin pipeline configuration"):
        load_config(pipeline_path, features_path)

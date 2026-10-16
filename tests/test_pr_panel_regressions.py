from pathlib import Path

import pytest

from michelin.config import load_config
from michelin.data import database
from michelin.data.panel import PanelBuildConfig, PanelBuildResult
from test_database_export import export_fixture


@pytest.mark.parametrize("failure", ["missing_mapping", "invalid_input", "partial_publication"])
def test_failed_rebuild_removes_current_outputs_and_preserves_accepted_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    db_path, guide_path, mapping_path = export_fixture(tmp_path)
    config = load_config(Path("config/pipeline.yaml"), Path("config/features.yaml")).pipeline
    config = config.model_copy(update={"pools": config.pools.model_copy(update={
        "ordinary_in_sample_limit": 2, "ordinary_out_of_sample_limit": 1,
    })})
    output = tmp_path / "output"
    database.export_database_panel(db_path, guide_path, mapping_path, output, config)
    canonical = (
        output / "processed" / "labeled_restaurants.csv",
        output / "processed" / "panel_summary.json",
        output / "processed" / "panel_manifest.json",
    )
    accepted = tuple((path.name, path.read_bytes()) for path in canonical)

    if failure == "missing_mapping":
        mapping_path.write_text("place_id,michelin_url\nstar,https://guide.example/star\n")
        expected_error = ValueError
        expected_message = "Guide coverage incomplete"
    elif failure == "invalid_input":
        guide_path.write_text("invalid_column\ninvalid_value\n")
        expected_error = ValueError
        expected_message = None
    else:
        write_outputs = database.write_panel_outputs

        def interrupted_publication(result: PanelBuildResult, panel_config: PanelBuildConfig) -> None:
            write_outputs(result, panel_config)
            raise OSError("simulated publication failure")

        monkeypatch.setattr(database, "write_panel_outputs", interrupted_publication)
        expected_error = OSError
        expected_message = "simulated publication failure"

    with pytest.raises(expected_error, match=expected_message):
        database.export_database_panel(db_path, guide_path, mapping_path, output, config)

    assert all(not path.exists() for path in canonical)
    for name, original_bytes in accepted:
        recovered = tuple(path for path in output.rglob(name) if path not in canonical)
        assert recovered, f"Previously accepted {name} was lost"
        assert any(path.read_bytes() == original_bytes for path in recovered)

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("script_name", ["hydrate_michelin_prices", "backfill_price_fields"])
def test_price_cli_help_from_another_directory_without_pythonpath(tmp_path: Path, script_name: str) -> None:
    """File entry points resolve shared helpers independently of the working directory."""
    repository_root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [sys.executable, str(repository_root / "scripts" / f"{script_name}.py"), "--help"],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert process.returncode == 0, process.stderr
    assert "--dry-run" in process.stdout
    assert "price_range" in process.stdout

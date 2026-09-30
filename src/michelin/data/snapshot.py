"""Immutable local inputs and integrity manifests for reproducible panel builds."""

from datetime import UTC, datetime
from hashlib import file_digest
from pathlib import Path
import shutil
from typing import Literal

import duckdb
from pydantic import BaseModel, ConfigDict


SNAPSHOT_TABLE_COLUMNS = {
    "restaurants": "place_id, name, address, latitude, longitude, rating, review_count, price_level, price_range, website, phone, google_maps_url, business_status, primary_category, is_michelin, last_updated",
    "restaurant_michelin": "place_id, michelin_name, michelin_category, michelin_stars, michelin_bib_gourmand, guide_year, last_updated",
    "place_raw_responses": "place_id, response_json, response_hash, fetched_at",
    "place_raw_response_history": "place_id, response_json, response_hash, fetched_at",
}
SNAPSHOT_FILES = frozenset({"restaurants.duckdb", "guide.csv", "place_ids.csv", "pipeline.yaml", "features.yaml"})


class SnapshotManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    captured_at: datetime
    source_database: str
    file_sha256: dict[str, str]
    table_rows: dict[str, int]
    collection_start: str
    collection_end: str
    published_reconciliation: Literal["deferred"]
    history_limit: str


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def freeze_snapshot(
    database_path: Path, guide_path: Path, mapping_path: Path,
    pipeline_path: Path, features_path: Path, destination: Path,
) -> SnapshotManifest:
    """Copy only restaurant inputs in one database transaction; never replace a snapshot."""
    destination.mkdir(parents=True, exist_ok=False)
    captured_at = datetime.now(UTC)
    escaped_source = str(database_path.resolve()).replace("'", "''")
    with duckdb.connect(str(destination / "restaurants.duckdb")) as connection:
        connection.execute(f"ATTACH '{escaped_source}' AS live (READ_ONLY)")
        connection.execute("BEGIN TRANSACTION")
        try:
            for table, columns in SNAPSHOT_TABLE_COLUMNS.items():
                connection.execute(f"CREATE TABLE {table} AS SELECT {columns} FROM live.{table}")
            table_rows = {table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in SNAPSHOT_TABLE_COLUMNS}
            first, last = connection.execute("SELECT min(fetched_at), max(fetched_at) FROM place_raw_responses").fetchone()
            if first is None or last is None:
                raise ValueError("Cannot freeze a snapshot with no collected restaurant responses")
            connection.execute("COMMIT")
        except (duckdb.Error, ValueError):
            connection.execute("ROLLBACK")
            raise
    for source, name in ((guide_path, "guide.csv"), (mapping_path, "place_ids.csv"), (pipeline_path, "pipeline.yaml"), (features_path, "features.yaml")):
        shutil.copyfile(source, destination / name)
    manifest = SnapshotManifest(
        captured_at=captured_at, source_database=str(database_path),
        file_sha256={name: sha256_file(destination / name) for name in sorted(SNAPSHOT_FILES)},
        table_rows=table_rows, collection_start=first.isoformat(), collection_end=last.isoformat(),
        published_reconciliation="deferred",
        history_limit="History starts with responses surviving at migration; previously overwritten responses are unrecoverable.",
    )
    (destination / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_snapshot(directory: Path) -> SnapshotManifest:
    """Reject modified or incomplete frozen inputs before any panel build."""
    manifest = SnapshotManifest.model_validate_json((directory / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest.file_sha256) != SNAPSHOT_FILES:
        raise ValueError("Snapshot manifest does not contain the required input files")
    for name, expected in manifest.file_sha256.items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"Snapshot hash mismatch: {name}")
    return manifest

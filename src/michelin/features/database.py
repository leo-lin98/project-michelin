"""Read-only, bounded access to the frozen eligible Phase 1 reference pool."""

from collections.abc import Iterator
import logging
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError

from michelin.config import EligibilityConfig, PipelineConfig
from michelin.data.database import DatabaseRestaurant, city_from_address, provider_feature_row, read_guide_place_ids, read_scoped_guide
from michelin.data.enrichment import FEATURE_COLUMNS
from michelin.data.panel import split_invalid_rows
from michelin.data.snapshot import sha256_file


class InputHashes(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    database: str
    guide: str
    mapping: str


class PanelManifest(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    database: Path
    guide_snapshot: Path
    place_id_mapping: Path
    cities: tuple[str, ...]
    seed: int
    feature_provider: Literal["google_places"]
    snapshot_policy: Literal["per_record_fetched_at"]
    input_sha256: InputHashes
    eligibility: EligibilityConfig


def verify_manifest(manifest: PanelManifest, config: PipelineConfig) -> None:
    if manifest.cities != config.project.geography or manifest.seed != config.seed or manifest.feature_provider != config.data_sources.enrichment_provider:
        raise ValueError("Panel manifest geography/seed/provider differs from pipeline configuration")
    if manifest.eligibility != config.eligibility:
        raise ValueError("Panel manifest eligibility differs from pipeline configuration")
    for path, expected in ((manifest.database, manifest.input_sha256.database), (manifest.guide_snapshot, manifest.input_sha256.guide), (manifest.place_id_mapping, manifest.input_sha256.mapping)):
        if sha256_file(path) != expected:
            raise ValueError(f"Frozen Phase 1 input hash mismatch: {path}")


def eligible_reference_batches(manifest: PanelManifest, config: PipelineConfig, batch_size: int, panel: pd.DataFrame) -> Iterator[pd.DataFrame]:
    """Match Phase 1 schema, municipality, status, and Guide identity eligibility."""
    if batch_size <= 0:
        raise ValueError("Reference batch_size must be positive")
    guide = read_scoped_guide(manifest.guide_snapshot, config.project.geography)
    mappings = read_guide_place_ids(manifest.place_id_mapping, guide)
    if set(config.eligibility.unresolved_guide_exclusions).intersection(mappings):
        raise ValueError("An excluded Guide entry now has a mapping; review eligibility")
    guide_ids = frozenset(mappings.values())
    panel_by_source = panel.set_index("source_id")
    with duckdb.connect(str(manifest.database), read_only=True) as connection:
        if connection.execute("SELECT count(*) - count(DISTINCT place_id) FROM restaurants").fetchone()[0]:
            raise ValueError("Frozen density database contains duplicate identities")
        cursor = connection.execute("""SELECT r.place_id AS source_id, r.name, r.address, r.latitude, r.longitude,
                   r.rating AS average_rating, r.review_count, r.price_level, r.price_range, r.primary_category,
                   raw.fetched_at, r.is_michelin, r.business_status
            FROM restaurants r LEFT JOIN place_raw_responses raw USING (place_id)
            ORDER BY r.place_id""")
        columns = [column[0] for column in cursor.description]
        while batch := cursor.fetchmany(batch_size):
            retained: list[dict[str, str | int | float | None]] = []
            for values in batch:
                try:
                    row = DatabaseRestaurant.model_validate(dict(zip(columns, values, strict=True)))
                except ValidationError as exc:
                    logging.getLogger(__name__).warning(
                        "Excluded provider row under Phase 1 schema eligibility",
                        extra={"source_id": values[0], "validation_errors": [
                            {"field": ".".join(map(str, error["loc"])), "error_type": error["type"]}
                            for error in exc.errors(include_input=False)
                        ]},
                    )
                    continue
                city = city_from_address(row.address)
                if city not in config.project.geography or row.business_status not in config.eligibility.business_statuses:
                    continue
                if row.is_michelin and row.source_id not in guide_ids:
                    raise ValueError(f"Database Guide Place ID {row.source_id} absent from frozen Guide mapping")
                retained.append(provider_feature_row(row, city))
            eligible, rejected = split_invalid_rows(pd.DataFrame(retained, columns=["source_id", *FEATURE_COLUMNS]))
            for source_id in rejected["source_id"]:
                logging.getLogger(__name__).warning(
                    "Excluded provider row under Phase 1 identity/snapshot eligibility",
                    extra={"source_id": source_id, "reason": "invalid_identity_or_snapshot"},
                )
            verify_panel_source_features(panel_by_source, eligible)
            yield eligible


def verify_panel_source_features(panel_by_source: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Reject changed shared inputs by comparing identities against the frozen provider."""
    fields = ("price_range", "average_rating", "review_count", "cuisine", "city",
              "latitude", "longitude", "feature_provider", "feature_snapshot_date")
    overlapping = reference.loc[reference["source_id"].isin(panel_by_source.index)].set_index("source_id")
    for source_id, provider in overlapping.iterrows():
        panel_row = panel_by_source.loc[source_id]
        for field in fields:
            actual, expected = panel_row[field], provider[field]
            actual_missing, expected_missing = bool(pd.isna(actual)), bool(pd.isna(expected))
            if actual_missing != expected_missing or (not actual_missing and actual != expected):
                raise ValueError(f"Panel feature differs from frozen provider: source_id={source_id}, field={field}")

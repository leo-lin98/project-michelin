"""Resolve Taipei and New Taipei Michelin rows to Google Places IDs."""

import math
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Callable
from typing import Protocol

import pandas as pd

from restaurant_ingestion.clients.google_places import ensure_request_budget
from restaurant_ingestion.models import CandidatePlace, PlaceSearchResponse

if TYPE_CHECKING:
    from restaurant_ingestion.ingestion.place_id_scoring import MatchDecision

TAIPEI = "Taipei"
TAIWAN = "Taiwan"
SOURCE_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "Name": ("Name", "name"),
    "Address": ("Address", "address"),
    "Location": ("Location", "location"),
    "Award": ("Award", "michelin_category"),
    "Latitude": ("Latitude", "latitude"),
    "Longitude": ("Longitude", "longitude"),
    "Url": ("Url", "michelin_url"),
    "PhoneNumber": ("PhoneNumber", "phone_number"),
    "WebsiteUrl": ("WebsiteUrl", "website_url"),
}
REQUIRED_SOURCE_COLUMNS = ("Name", "Address", "Location", "Award", "Latitude", "Longitude", "Url")
OUTPUT_COLUMNS = (
    "place_id",
    "name",
    "michelin_category",
    "address",
    "location",
    "latitude",
    "longitude",
    "michelin_url",
)
UNRESOLVED_COLUMNS = (*OUTPUT_COLUMNS[1:], "phone_number", "website_url", "text_query")
VALIDATION_REPORT_COLUMNS = (
    "name",
    "michelin_category",
    "address",
    "location",
    "latitude",
    "longitude",
    "michelin_url",
    "text_query",
    "reason",
    "candidate_count",
    "valid_candidate_count",
    "candidate_place_ids",
    "valid_candidate_place_ids",
    "candidate_addresses",
    "candidate_distances_meters",
    "candidate_failed_checks",
    "predicted_outcome",
    "predicted_place_id",
    "llm_recommended_outcome",
    "llm_recommended_place_id",
    "llm_confidence",
    "llm_evidence",
    "llm_conflicts",
)
OPTIONAL_VALIDATION_REPORT_COLUMNS = frozenset(
    (
        "predicted_outcome",
        "predicted_place_id",
        "llm_recommended_outcome",
        "llm_recommended_place_id",
        "llm_confidence",
        "llm_evidence",
        "llm_conflicts",
    )
)
QUERY_MODE_STRICT = "strict"
QUERY_MODE_SIMPLE = "simple"
QUERY_MODES = (QUERY_MODE_STRICT, QUERY_MODE_SIMPLE)
EARTH_RADIUS_METERS = 6_371_000.0
MAX_PLACE_MATCH_DISTANCE_METERS = 50.0
MAX_STRONG_ADDRESS_MATCH_DISTANCE_METERS = 1_000.0
RETRY_STEM_PREFIX = "retry-"


class PlaceIdSearchClient(Protocol):
    def search_text_place_candidates(
        self,
        text_query: str,
        request_budget_remaining: int,
    ) -> tuple[PlaceSearchResponse, tuple[dict[str, int], ...]]:
        pass


class PlaceIdMatchCandidateClient(Protocol):
    def search_text_place_id_match_candidates(
        self,
        text_query: str,
        request_budget_remaining: int,
    ) -> tuple[dict[str, Any], tuple[dict[str, int], ...]]:
        pass


type PlaceIdLlmJudge = Callable[[dict[str, object], tuple[dict[str, object], ...]], dict[str, str | tuple[str, ...]]]


def plan_michelin_place_id_lookup(source_csv_path: Path) -> dict[str, int]:
    rows = read_source_rows(source_csv_path)
    taipei_rows = taipei_source_rows(rows)
    return {
        "source_rows": len(rows),
        "taipei_rows": len(taipei_rows),
        "non_taipei_rows": len(rows) - len(taipei_rows),
        "text_search_requests": len(taipei_rows),
    }


def resolve_michelin_place_ids(
    source_csv_path: Path,
    output_csv_path: Path,
    client: PlaceIdSearchClient,
    request_budget: int,
    query_mode: str,
) -> dict[str, int]:
    return resolve_michelin_place_ids_to_paths(
        source_csv_path,
        output_csv_path,
        unresolved_csv_path(output_csv_path),
        client,
        request_budget,
        query_mode,
    )


def resolve_michelin_place_ids_to_paths(
    source_csv_path: Path,
    output_csv_path: Path,
    unresolved_output_csv_path: Path,
    client: PlaceIdSearchClient,
    request_budget: int,
    query_mode: str,
) -> dict[str, int]:
    validate_query_mode(query_mode)
    source_rows = taipei_source_rows(read_source_rows(source_csv_path))
    existing_rows = read_existing_output_rows(output_csv_path)
    existing_unresolved_rows = read_unresolved_rows(unresolved_output_csv_path)
    existing_keys = tuple(row_key(row) for row in existing_rows)
    existing_unresolved_keys = tuple(row_key(row) for row in existing_unresolved_rows)
    existing_validation_report_rows = read_validation_report_rows(validation_report_csv_path(unresolved_output_csv_path))
    resolved_rows: list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []
    validation_report_rows: list[dict[str, object]] = []
    used_requests = 0
    skipped_existing_rows = 0
    skipped_existing_unresolved_rows = 0

    for row in source_rows:
        key = row_key(output_row(row, "already-resolved"))
        if key in existing_keys:
            skipped_existing_rows += 1
            continue
        if key in existing_unresolved_keys:
            skipped_existing_unresolved_rows += 1
            continue
        ensure_request_budget(used_requests, request_budget, 1)
        text_query = text_query_for_row(row, query_mode)
        response, attempts = client.search_text_place_candidates(
            text_query,
            request_budget - used_requests,
        )
        used_requests += len(attempts)
        report_row = validation_report_row(row, text_query, response.places)
        valid_place_ids = tuple(place_id for place_id in row_text(report_row, "valid_candidate_place_ids").split("|") if place_id)
        if len(valid_place_ids) != 1:
            unresolved_rows.append(unresolved_row(row, text_query))
            validation_report_rows.append(report_row)
            continue
        resolved_rows.append(output_row(row, valid_place_ids[0]))

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame((*existing_rows, *resolved_rows), columns=OUTPUT_COLUMNS).to_csv(output_csv_path, index=False)
    unresolved_output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame((*existing_unresolved_rows, *unresolved_rows), columns=UNRESOLVED_COLUMNS).to_csv(
        unresolved_output_csv_path,
        index=False,
    )
    validation_report_output_path = validation_report_csv_path(unresolved_output_csv_path)
    validation_report_output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        (*existing_validation_report_rows, *validation_report_rows),
        columns=VALIDATION_REPORT_COLUMNS,
    ).to_csv(
        validation_report_output_path,
        index=False,
    )
    return {
        "taipei_rows": len(source_rows),
        "resolved_rows": len(resolved_rows),
        "unresolved_rows": len(unresolved_rows),
        "skipped_existing_rows": skipped_existing_rows,
        "skipped_existing_unresolved_rows": skipped_existing_unresolved_rows,
        "text_search_requests": used_requests,
    }


def resolve_michelin_place_ids_with_matcher(
    source_csv_path: Path,
    output_csv_path: Path,
    client: PlaceIdMatchCandidateClient,
    request_budget: int,
    query_mode: str,
    judge: PlaceIdLlmJudge | None,
) -> dict[str, int]:
    return resolve_michelin_place_ids_with_matcher_to_paths(
        source_csv_path,
        output_csv_path,
        unresolved_csv_path(output_csv_path),
        client,
        request_budget,
        query_mode,
        judge,
    )


def resolve_michelin_place_ids_with_matcher_to_paths(
    source_csv_path: Path,
    output_csv_path: Path,
    unresolved_output_csv_path: Path,
    client: PlaceIdMatchCandidateClient,
    request_budget: int,
    query_mode: str,
    judge: PlaceIdLlmJudge | None,
) -> dict[str, int]:
    validate_query_mode(query_mode)
    source_rows = taipei_source_rows(read_source_rows(source_csv_path))
    existing_rows = read_existing_output_rows(output_csv_path)
    existing_unresolved_rows = read_unresolved_rows(unresolved_output_csv_path)
    existing_keys = tuple(row_key(row) for row in existing_rows)
    existing_unresolved_keys = tuple(row_key(row) for row in existing_unresolved_rows)
    existing_validation_report_rows = read_validation_report_rows(validation_report_csv_path(unresolved_output_csv_path))
    resolved_rows: list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []
    validation_report_rows: list[dict[str, object]] = []
    used_requests = 0
    llm_judged_rows = 0
    skipped_existing_rows = 0
    skipped_existing_unresolved_rows = 0

    for row in source_rows:
        key = row_key(output_row(row, "already-resolved"))
        if key in existing_keys:
            skipped_existing_rows += 1
            continue
        if key in existing_unresolved_keys:
            skipped_existing_unresolved_rows += 1
            continue
        ensure_request_budget(used_requests, request_budget, 1)
        scoring_row = scoring_source_row(row)
        primary_query = text_query_for_row(row, query_mode)
        primary_response, attempts = client.search_text_place_id_match_candidates(
            primary_query,
            request_budget - used_requests,
        )
        used_requests += len(attempts)
        queries: list[dict[str, object]] = [{"text_query": primary_query, "response": primary_response}]
        decision = deterministic_place_id_decision(scoring_row, queries)

        # Tier 1 (recall): when the primary pool is uncertain and budget remains, pull in
        # candidates from the complementary query and re-score over the unioned pool.
        if decision.outcome != "place_id" and used_requests < request_budget:
            escalation_query = text_query_for_row(row, escalation_query_mode(query_mode))
            if escalation_query != primary_query:
                escalation_response, escalation_attempts = client.search_text_place_id_match_candidates(
                    escalation_query,
                    request_budget - used_requests,
                )
                used_requests += len(escalation_attempts)
                queries.append({"text_query": escalation_query, "response": escalation_response})
                decision = deterministic_place_id_decision(scoring_row, queries)

        # Tier 2 (precision): only if still uncertain after recall escalation, let the LLM
        # adjudicate over the unioned candidate pool.
        judgment = None
        if decision.outcome != "place_id" and judge is not None:
            decision, judgment = llm_place_id_decision(scoring_row, queries, judge)
            if judgment is not None:
                llm_judged_rows += 1

        report_query = pipe_join(tuple(str(entry["text_query"]) for entry in queries))
        validation_report_rows.append(
            matcher_validation_report_row(row, report_query, merged_candidate_response(queries), decision, judgment)
        )
        if decision.outcome == "place_id":
            resolved_rows.append(output_row(row, decision.place_id))
            continue
        unresolved_rows.append(unresolved_row(row, primary_query))

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame((*existing_rows, *resolved_rows), columns=OUTPUT_COLUMNS).to_csv(output_csv_path, index=False)
    unresolved_output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame((*existing_unresolved_rows, *unresolved_rows), columns=UNRESOLVED_COLUMNS).to_csv(
        unresolved_output_csv_path,
        index=False,
    )
    validation_report_output_path = validation_report_csv_path(unresolved_output_csv_path)
    validation_report_output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        (*existing_validation_report_rows, *validation_report_rows),
        columns=VALIDATION_REPORT_COLUMNS,
    ).to_csv(
        validation_report_output_path,
        index=False,
    )
    return {
        "taipei_rows": len(source_rows),
        "resolved_rows": len(resolved_rows),
        "unresolved_rows": len(unresolved_rows),
        "skipped_existing_rows": skipped_existing_rows,
        "skipped_existing_unresolved_rows": skipped_existing_unresolved_rows,
        "text_search_requests": used_requests,
        "llm_judged_rows": llm_judged_rows,
    }


def stack_resolved_place_id_datasets(input_csv_paths: tuple[Path, ...], output_csv_path: Path) -> dict[str, int]:
    if not input_csv_paths:
        raise ValueError("At least one resolved Michelin place_id CSV is required")

    stacked_rows = tuple(read_resolved_output_frame(path) for path in input_csv_paths)
    validate_stacked_resolved_rows(stacked_rows)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(stacked_rows, ignore_index=True).to_csv(output_csv_path, index=False)
    return {
        "input_files": len(input_csv_paths),
        "output_rows": sum(len(rows) for rows in stacked_rows),
    }


def build_processed_place_id_dataset(
    resolved_dir: Path,
    unresolved_dir: Path,
    output_csv_path: Path,
) -> dict[str, int | str]:
    input_csv_paths = processed_place_id_input_paths(resolved_dir, unresolved_dir)
    result = stack_resolved_place_id_datasets(input_csv_paths, output_csv_path)
    mode = "base-only" if len(input_csv_paths) == 1 else "stack-resolved"
    return {**result, "mode": mode}


def stack_resolved_place_id_directory(resolved_dir: Path, output_csv_path: Path) -> dict[str, int]:
    return stack_resolved_place_id_datasets(resolved_place_id_csv_paths(resolved_dir), output_csv_path)


def processed_place_id_input_paths(resolved_dir: Path, unresolved_dir: Path) -> tuple[Path, ...]:
    resolved_paths = resolved_place_id_csv_paths(resolved_dir)
    base_path = resolved_dir / "base.csv"
    if base_path not in resolved_paths:
        raise FileNotFoundError(f"Resolved Michelin place_id base CSV does not exist: path={base_path}")

    base_unresolved_path = unresolved_output_path_for_run_stem(unresolved_dir, "base")
    if not base_unresolved_path.exists():
        raise FileNotFoundError(f"Base unresolved Michelin place_id CSV does not exist: path={base_unresolved_path}")
    if len(read_unresolved_frame(base_unresolved_path)) == 0:
        return (base_path,)

    latest_resolved_stem = latest_resolved_retry_stem(resolved_paths)
    if latest_resolved_stem is None:
        raise ValueError(f"Unresolved Michelin place_id rows remain for the base run: unresolved_csv={base_unresolved_path}")
    latest_unresolved_path = unresolved_output_path_for_run_stem(unresolved_dir, latest_resolved_stem)
    if not latest_unresolved_path.exists():
        raise FileNotFoundError(f"Latest retry unresolved Michelin place_id CSV does not exist: path={latest_unresolved_path}")
    if len(read_unresolved_frame(latest_unresolved_path)) > 0:
        raise ValueError(f"Unresolved Michelin place_id rows remain for the latest run: unresolved_csv={latest_unresolved_path}")
    return resolved_paths


def resolved_place_id_csv_paths(resolved_dir: Path) -> tuple[Path, ...]:
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Resolved Michelin place_id directory does not exist: path={resolved_dir}")
    paths = tuple(sorted(path for path in resolved_dir.glob("*.csv") if path.is_file()))
    if not paths:
        raise ValueError(f"Resolved Michelin place_id directory contains no CSV files: path={resolved_dir}")
    return paths


def latest_unresolved_csv_path(unresolved_dir: Path) -> Path:
    path = latest_unresolved_csv_path_or_none(unresolved_dir)
    if path is None:
        raise ValueError(f"Unresolved Michelin place_id directory contains no non-empty CSV files: path={unresolved_dir}")
    return path


def latest_unresolved_csv_path_or_none(unresolved_dir: Path) -> Path | None:
    if not unresolved_dir.exists():
        raise FileNotFoundError(f"Unresolved Michelin place_id directory does not exist: path={unresolved_dir}")
    paths = unresolved_workflow_csv_paths(unresolved_dir)
    nonempty_paths = tuple(path for path in paths if len(read_unresolved_frame(path)) > 0)
    if not nonempty_paths:
        return None
    return nonempty_paths[-1]


def unresolved_workflow_csv_paths(unresolved_dir: Path) -> tuple[Path, ...]:
    paths = tuple(path for path in unresolved_dir.glob("*.unresolved.csv") if path.is_file())
    workflow_paths = tuple(path for path in paths if unresolved_workflow_sort_key(path) is not None)
    return tuple(sorted(workflow_paths, key=lambda path: unresolved_workflow_sort_key(path) or 0))


def unresolved_workflow_sort_key(path: Path) -> int | None:
    stem = path.name.removesuffix(".unresolved.csv")
    if stem == "base":
        return 0
    retry_number = retry_number_from_stem(stem)
    if retry_number is None:
        return None
    return retry_number


def latest_resolved_retry_stem(resolved_paths: tuple[Path, ...]) -> str | None:
    retry_paths = tuple(path for path in resolved_paths if retry_number_from_stem(path.stem) is not None)
    if not retry_paths:
        return None
    return max(retry_paths, key=lambda path: retry_number_from_stem(path.stem) or 0).stem


def next_retry_output_path(resolved_dir: Path) -> Path:
    resolved_dir.mkdir(parents=True, exist_ok=True)
    retry_numbers = tuple(retry_number_from_stem(path.stem) for path in resolved_dir.glob(f"{RETRY_STEM_PREFIX}*.csv"))
    valid_retry_numbers = tuple(number for number in retry_numbers if number is not None)
    next_number = max(valid_retry_numbers, default=0) + 1
    return resolved_dir / f"{RETRY_STEM_PREFIX}{next_number:03d}.csv"


def unresolved_output_path_for_resolved_output(resolved_output_csv_path: Path, unresolved_dir: Path) -> Path:
    return unresolved_output_path_for_run_stem(unresolved_dir, resolved_output_csv_path.stem)


def unresolved_output_path_for_run_stem(unresolved_dir: Path, run_stem: str) -> Path:
    return unresolved_dir / f"{run_stem}.unresolved.csv"


def run_output_path(resolved_dir: Path, run_stem: str) -> Path:
    validate_run_stem(run_stem)
    return resolved_dir / f"{run_stem}.csv"


def validate_run_stem(run_stem: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_stem):
        raise ValueError(f"Invalid run stem: value={run_stem}")


def retry_number_from_stem(stem: str) -> int | None:
    if not stem.startswith(RETRY_STEM_PREFIX):
        return None
    suffix = stem.removeprefix(RETRY_STEM_PREFIX)
    if not suffix.isdecimal():
        return None
    return int(suffix)


def read_source_rows(source_csv_path: Path) -> tuple[dict[str, object], ...]:
    rows = pd.read_csv(source_csv_path)
    rename_map = build_source_rename_map(rows.columns)
    missing_columns = set(REQUIRED_SOURCE_COLUMNS).difference(rename_map.values())
    if missing_columns:
        raise ValueError(f"Michelin source CSV missing required columns: {sorted(missing_columns)}")
    rows = rows.rename(columns=rename_map)
    for column in ("PhoneNumber", "WebsiteUrl"):
        if column not in rows.columns:
            rows[column] = ""
    return tuple(dict(row) for _, row in rows.iterrows())


def build_source_rename_map(columns: pd.Index) -> dict[str, str]:
    rename_map: dict[str, str] = {}
    for canonical_name, aliases in SOURCE_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in columns:
                rename_map[alias] = canonical_name
                break
    return rename_map


def read_existing_output_rows(output_csv_path: Path) -> tuple[dict[str, object], ...]:
    if not output_csv_path.exists():
        return ()
    rows = pd.read_csv(output_csv_path)
    missing_columns = set(OUTPUT_COLUMNS).difference(rows.columns)
    if missing_columns:
        raise ValueError(f"Existing Michelin place_id output missing required columns: {sorted(missing_columns)}")
    return tuple(dict(row) for _, row in rows.iterrows())


def read_resolved_output_frame(input_csv_path: Path) -> pd.DataFrame:
    rows = pd.read_csv(input_csv_path)
    actual_columns = set(rows.columns)
    expected_columns = set(OUTPUT_COLUMNS)
    missing_columns = expected_columns.difference(actual_columns)
    unexpected_columns = actual_columns.difference(expected_columns)
    if missing_columns or unexpected_columns:
        raise ValueError(
            "Resolved Michelin place_id CSV schema mismatch: "
            f"path={input_csv_path}, missing={sorted(missing_columns)}, unexpected={sorted(unexpected_columns)}"
        )
    blank_place_id_mask = rows["place_id"].fillna("").astype(str).str.strip().eq("")
    if blank_place_id_mask.any():
        raise ValueError(f"Resolved Michelin place_id CSV contains blank place_id values: path={input_csv_path}")
    return rows.loc[:, OUTPUT_COLUMNS].copy()


def validate_stacked_resolved_rows(stacked_rows: tuple[pd.DataFrame, ...]) -> None:
    if not stacked_rows:
        raise ValueError("At least one resolved Michelin place_id frame is required")
    rows = pd.concat(stacked_rows, ignore_index=True)
    row_keys_by_place_id: dict[str, set[tuple[str, str, str, str]]] = {}
    place_ids_by_row_key: dict[tuple[str, str, str, str], set[str]] = {}
    for _, row in rows.iterrows():
        row_dict = dict(row)
        key = row_key(row_dict)
        place_id = row_text(row_dict, "place_id")
        row_keys_by_place_id.setdefault(place_id, set()).add(key)
        place_ids_by_row_key.setdefault(key, set()).add(place_id)

    duplicated_place_ids = sorted(place_id for place_id, keys in row_keys_by_place_id.items() if len(keys) > 1)
    conflicting_row_keys = sorted(key for key, place_ids in place_ids_by_row_key.items() if len(place_ids) > 1)
    if duplicated_place_ids:
        raise ValueError(f"Resolved Michelin place_id CSVs assign one place_id to multiple Michelin rows: place_ids={duplicated_place_ids}")
    if conflicting_row_keys:
        raise ValueError(f"Resolved Michelin place_id CSVs assign conflicting place_ids to the same Michelin row: row_keys={conflicting_row_keys}")


def read_existing_unresolved_rows(output_csv_path: Path) -> tuple[dict[str, object], ...]:
    return read_unresolved_rows(unresolved_csv_path(output_csv_path))


def read_unresolved_rows(path: Path) -> tuple[dict[str, object], ...]:
    if not path.exists():
        return ()
    rows = read_unresolved_frame(path)
    return tuple(dict(row) for _, row in rows.iterrows())


def read_unresolved_frame(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    actual_columns = set(rows.columns)
    expected_columns = set(UNRESOLVED_COLUMNS)
    optional_columns = {"phone_number", "website_url"}
    missing_columns = expected_columns.difference(actual_columns).difference(optional_columns)
    unexpected_columns = actual_columns.difference(expected_columns)
    if missing_columns or unexpected_columns:
        raise ValueError(
            "Existing Michelin unresolved output schema mismatch: "
            f"path={path}, missing={sorted(missing_columns)}, unexpected={sorted(unexpected_columns)}"
        )
    for column in optional_columns:
        if column not in rows.columns:
            rows[column] = ""
    return rows.loc[:, UNRESOLVED_COLUMNS].copy()


def read_validation_report_rows(path: Path) -> tuple[dict[str, object], ...]:
    if not path.exists():
        return ()
    rows = pd.read_csv(path)
    actual_columns = set(rows.columns)
    expected_columns = set(VALIDATION_REPORT_COLUMNS)
    missing_columns = expected_columns.difference(actual_columns).difference(OPTIONAL_VALIDATION_REPORT_COLUMNS)
    unexpected_columns = actual_columns.difference(expected_columns)
    if missing_columns or unexpected_columns:
        raise ValueError(
            "Existing Michelin place_id validation report schema mismatch: "
            f"path={path}, missing={sorted(missing_columns)}, unexpected={sorted(unexpected_columns)}"
        )
    for column in OPTIONAL_VALIDATION_REPORT_COLUMNS:
        if column not in rows.columns:
            rows[column] = ""
    return tuple(dict(row) for _, row in rows.loc[:, VALIDATION_REPORT_COLUMNS].iterrows())


def taipei_source_rows(rows: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    return tuple(row for row in rows if is_taipei_taiwan_row(row))


def is_taipei_taiwan_row(row: dict[str, object]) -> bool:
    location_parts = tuple(part.strip().lower() for part in row_text(row, "Location").split(","))
    return location_parts in {(TAIPEI.lower(), TAIWAN.lower()), ("new taipei", TAIWAN.lower())}


def text_query_for_row(row: dict[str, object], query_mode: str) -> str:
    validate_query_mode(query_mode)
    name = row_text(row, "Name")
    address = row_text(row, "Address")
    if not name:
        raise ValueError("Taipei Michelin row is missing Name")
    if not address:
        raise ValueError(f"Taipei Michelin row is missing Address: name={name}")
    city = row_text(row, "Location").split(",")[0].strip()
    if not is_taipei_taiwan_row(row):
        raise ValueError(f"Guide row is outside Taipei/New Taipei: {city}")
    if query_mode == QUERY_MODE_SIMPLE:
        return f"{name}, {city}, Taiwan"
    return f"{name}, {address}, {city}, Taiwan"


def matching_candidate_places(
    row: dict[str, object],
    candidates: tuple[CandidatePlace, ...],
) -> tuple[CandidatePlace, ...]:
    return tuple(candidate for candidate in candidates if candidate_matches_row(row, candidate))


def validation_report_row(
    row: dict[str, object],
    text_query: str,
    candidates: tuple[CandidatePlace, ...],
) -> dict[str, object]:
    candidate_reports = tuple(candidate_validation_report(row, candidate) for candidate in candidates)
    valid_candidate_reports = tuple(report for report in candidate_reports if not report["failed_checks"])
    return {
        "name": row_text(row, "Name"),
        "michelin_category": row_text(row, "Award"),
        "address": row_text(row, "Address"),
        "location": row_text(row, "Location"),
        "latitude": row_text(row, "Latitude"),
        "longitude": row_text(row, "Longitude"),
        "michelin_url": row_text(row, "Url"),
        "text_query": text_query,
        "reason": unresolved_reason(candidate_reports, valid_candidate_reports),
        "candidate_count": len(candidate_reports),
        "valid_candidate_count": len(valid_candidate_reports),
        "candidate_place_ids": pipe_join(tuple(row_text(report, "place_id") for report in candidate_reports)),
        "valid_candidate_place_ids": pipe_join(tuple(row_text(report, "place_id") for report in valid_candidate_reports)),
        "candidate_addresses": pipe_join(tuple(row_text(report, "formatted_address") for report in candidate_reports)),
        "candidate_distances_meters": pipe_join(tuple(row_text(report, "distance_meters") for report in candidate_reports)),
        "candidate_failed_checks": pipe_join(tuple(row_text(report, "failed_checks") for report in candidate_reports)),
    }


def matcher_validation_report_row(
    row: dict[str, object],
    text_query: str,
    response_json: dict[str, Any],
    decision: "MatchDecision",
    judgment: dict[str, str | tuple[str, ...]] | None,
) -> dict[str, object]:
    candidate_place_ids = candidate_place_ids_from_response(response_json)
    llm_report = llm_validation_report_fields(judgment)
    return {
        "name": row_text(row, "Name"),
        "michelin_category": row_text(row, "Award"),
        "address": row_text(row, "Address"),
        "location": row_text(row, "Location"),
        "latitude": row_text(row, "Latitude"),
        "longitude": row_text(row, "Longitude"),
        "michelin_url": row_text(row, "Url"),
        "text_query": text_query,
        "reason": decision.reason,
        "candidate_count": len(candidate_place_ids),
        "valid_candidate_count": 1 if decision.outcome == "place_id" else 0,
        "candidate_place_ids": pipe_join(candidate_place_ids),
        "valid_candidate_place_ids": decision.place_id,
        "candidate_addresses": pipe_join(candidate_addresses_from_response(response_json)),
        "candidate_distances_meters": pipe_join(candidate_distances_from_decision(decision)),
        "candidate_failed_checks": pipe_join(candidate_signals_from_decision(decision)),
        "predicted_outcome": decision.outcome,
        "predicted_place_id": decision.place_id,
        **llm_report,
    }


def llm_validation_report_fields(judgment: dict[str, str | tuple[str, ...]] | None) -> dict[str, object]:
    if judgment is None:
        return {
            "llm_recommended_outcome": "",
            "llm_recommended_place_id": "",
            "llm_confidence": "",
            "llm_evidence": "",
            "llm_conflicts": "",
        }
    from restaurant_ingestion.ingestion.place_id_llm_judge import llm_judgment_to_report

    report = llm_judgment_to_report(judgment)
    return {
        "llm_recommended_outcome": row_text(report, "recommended_outcome"),
        "llm_recommended_place_id": row_text(report, "recommended_place_id"),
        "llm_confidence": row_text(report, "confidence"),
        "llm_evidence": pipe_join(tuple(str(value) for value in report["evidence"])),
        "llm_conflicts": pipe_join(tuple(str(value) for value in report["conflicts"])),
    }


def escalation_query_mode(query_mode: str) -> str:
    validate_query_mode(query_mode)
    if query_mode == QUERY_MODE_SIMPLE:
        return QUERY_MODE_STRICT
    return QUERY_MODE_SIMPLE


def deterministic_place_id_decision(scoring_row: dict[str, object], queries: list[dict[str, object]]) -> "MatchDecision":
    from restaurant_ingestion.ingestion.place_id_scoring import decide_place_id_match

    return decide_place_id_match(scoring_row, {"queries": queries})


def llm_place_id_decision(
    scoring_row: dict[str, object],
    queries: list[dict[str, object]],
    judge: PlaceIdLlmJudge,
) -> tuple["MatchDecision", dict[str, str | tuple[str, ...]] | None]:
    from restaurant_ingestion.ingestion.place_id_llm_judge import decide_place_id_match_with_llm

    return decide_place_id_match_with_llm(scoring_row, {"queries": queries}, judge)


def merged_candidate_response(queries: list[dict[str, object]]) -> dict[str, Any]:
    seen: set[str] = set()
    places: list[dict[str, object]] = []
    for entry in queries:
        response = entry["response"]
        if not isinstance(response, dict):
            raise ValueError("Google Places match candidate response must be an object")
        for place in response_places(response):
            place_id = row_text(place, "id")
            if place_id and place_id not in seen:
                seen.add(place_id)
                places.append(place)
    return {"places": places}


def scoring_source_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "name": row_text(row, "Name"),
        "address": row_text(row, "Address"),
        "latitude": row_text(row, "Latitude"),
        "longitude": row_text(row, "Longitude"),
        "phone_number": row_text(row, "PhoneNumber"),
        "website_url": row_text(row, "WebsiteUrl"),
        "michelin_category": row_text(row, "Award"),
        "michelin_url": row_text(row, "Url"),
    }


def candidate_place_ids_from_response(response_json: dict[str, Any]) -> tuple[str, ...]:
    return tuple(row_text(candidate, "id") for candidate in response_places(response_json))


def candidate_addresses_from_response(response_json: dict[str, Any]) -> tuple[str, ...]:
    return tuple(row_text(candidate, "formattedAddress") for candidate in response_places(response_json))


def response_places(response_json: dict[str, Any]) -> tuple[dict[str, object], ...]:
    places = response_json.get("places", [])
    if not isinstance(places, list):
        raise ValueError("Google Places match candidate response must contain a places list")
    result: list[dict[str, object]] = []
    for place in places:
        if not isinstance(place, dict):
            raise ValueError("Google Places match candidate must be an object")
        result.append(place)
    return tuple(result)


def candidate_distances_from_decision(decision: "MatchDecision") -> tuple[str, ...]:
    return tuple(f"{candidate.distance_meters:.1f}" for candidate in decision.candidates)


def candidate_signals_from_decision(decision: "MatchDecision") -> tuple[str, ...]:
    return tuple("+".join(candidate.signals) for candidate in decision.candidates)


def candidate_validation_report(row: dict[str, object], candidate: CandidatePlace) -> dict[str, object]:
    failed_checks: list[str] = []
    distance_meters = ""
    if not candidate.formattedAddress:
        failed_checks.append("missing_formatted_address")
    if candidate.location is None:
        failed_checks.append("missing_location")
    else:
        distance_value = candidate_distance_meters(row, candidate)
        distance_meters = f"{distance_value:.1f}"
        if distance_value > max_allowed_candidate_distance_meters(row, candidate):
            failed_checks.append("distance_too_far")

    if candidate.formattedAddress and not structured_addresses_match(
        row_text(row, "Address"),
        candidate.formattedAddress,
    ):
        failed_checks.append("address_mismatch")

    return {
        "place_id": candidate.id,
        "formatted_address": candidate.formattedAddress or "",
        "distance_meters": distance_meters,
        "failed_checks": "+".join(failed_checks),
    }


def unresolved_reason(
    candidate_reports: tuple[dict[str, object], ...],
    valid_candidate_reports: tuple[dict[str, object], ...],
) -> str:
    if not candidate_reports:
        return "no_candidates"
    if len(valid_candidate_reports) > 1:
        return "ambiguous_candidates"
    failed_checks = tuple(row_text(report, "failed_checks") for report in candidate_reports)
    if any("missing_location" in checks for checks in failed_checks):
        return "missing_location"
    if any("distance_too_far" in checks for checks in failed_checks):
        return "distance_too_far"
    if any("address_mismatch" in checks for checks in failed_checks):
        return "address_mismatch"
    return "ambiguous_candidates"


def candidate_matches_row(row: dict[str, object], candidate: CandidatePlace) -> bool:
    if not candidate.formattedAddress or candidate.location is None:
        return False

    if candidate_distance_meters(row, candidate) > max_allowed_candidate_distance_meters(row, candidate):
        return False

    return structured_addresses_match(row_text(row, "Address"), candidate.formattedAddress)


def max_allowed_candidate_distance_meters(row: dict[str, object], candidate: CandidatePlace) -> float:
    if candidate.formattedAddress and structured_addresses_match(row_text(row, "Address"), candidate.formattedAddress):
        return MAX_STRONG_ADDRESS_MATCH_DISTANCE_METERS
    return MAX_PLACE_MATCH_DISTANCE_METERS


def candidate_distance_meters(row: dict[str, object], candidate: CandidatePlace) -> float:
    if candidate.location is None:
        raise ValueError("Cannot compute candidate distance without location")
    return distance_between_coordinates_meters(
        row_coordinate(row, "Latitude"),
        row_coordinate(row, "Longitude"),
        candidate.location.latitude,
        candidate.location.longitude,
    )


def pipe_join(values: tuple[str, ...]) -> str:
    return "|".join(value.replace("|", "/") for value in values)


def structured_addresses_match(source_address: str, candidate_address: str) -> bool:
    source_parts = parse_structured_address(source_address)
    candidate_parts = parse_structured_address(candidate_address)
    source_road = row_text(source_parts, "road")
    candidate_road = row_text(candidate_parts, "road")
    if not source_road or not candidate_road or source_road != candidate_road:
        return False
    return optional_parts_match(source_parts, candidate_parts, ("section", "lane", "alley")) and house_numbers_match(
        tuple(source_parts["house_numbers"]),
        tuple(candidate_parts["house_numbers"]),
    )


def parse_structured_address(address: str) -> dict[str, object]:
    normalized = normalize_address_text(address)
    return {
        "road": extract_road_key(normalized),
        "section": first_match(normalized, (r"section\s+(\d+)",)),
        "lane": first_match(normalized, (r"lane\s+(\d+)",)),
        "alley": first_match(normalized, (r"alley\s+(\d+)",)),
        "house_numbers": extract_house_numbers(normalized),
    }


def normalize_address_text(address: str) -> str:
    normalized = address.casefold().replace("號", " no ")
    normalized = re.sub(r"[.,;:/()\-]", " ", normalized)
    token_replacements = (
        (r"\bn\b", "north"),
        (r"\bs\b", "south"),
        (r"\be\b", "east"),
        (r"\bw\b", "west"),
        (r"\brd\b", "road"),
        (r"\bst\b", "street"),
        (r"\bave\b", "avenue"),
        (r"\bsec\b", "section"),
        (r"\bln\b", "lane"),
    )
    for pattern, replacement in token_replacements:
        normalized = re.sub(pattern, replacement, normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def extract_road_key(normalized_address: str) -> str:
    tokens = address_tokens(normalized_address)
    suffix_index = first_road_suffix_index(tokens)
    if suffix_index is None:
        return ""
    road_words: list[str] = []
    for token in reversed(tokens[:suffix_index]):
        if token.isdecimal():
            continue
        if token in {"section", "lane", "alley", "no"}:
            if road_words:
                break
            continue
        road_words.append(token)
        if len(road_words) == 3:
            break
    if not road_words:
        return ""
    return " ".join((*reversed(road_words), tokens[suffix_index]))


def canonical_road_suffix(suffix: str) -> str:
    if suffix in {"rd", "road"}:
        return "road"
    if suffix in {"st", "street"}:
        return "street"
    if suffix in {"ave", "avenue"}:
        return "avenue"
    return suffix


def first_match(text: str, patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match is not None:
            return match.group(1)
    return None


def extract_house_numbers(normalized_address: str) -> tuple[str, ...]:
    numbers: list[str] = []
    for pattern in (r"\bno\s+(\d+)", r"\b(\d+)\s+no\b", r"\b(\d+)\s*&\s*(\d+)\b"):
        for match in re.finditer(pattern, normalized_address):
            numbers.extend(group for group in match.groups() if group is not None)
    numbers.extend(house_numbers_before_road(normalized_address))
    return tuple(dict.fromkeys(numbers))


def address_tokens(normalized_address: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z]+|\d+|&", normalized_address))


def first_road_suffix_index(tokens: tuple[str, ...]) -> int | None:
    for index, token in enumerate(tokens):
        if token in {"road", "street", "avenue"}:
            return index
    return None


def house_numbers_before_road(normalized_address: str) -> tuple[str, ...]:
    tokens = address_tokens(normalized_address)
    suffix_index = first_road_suffix_index(tokens)
    if suffix_index is None:
        return ()
    for index in range(suffix_index - 1, -1, -1):
        token = tokens[index]
        if not token.isdecimal() or is_address_component_number(tokens, index):
            continue
        if index >= 2 and tokens[index - 1] == "&" and tokens[index - 2].isdecimal():
            return (tokens[index - 2], token)
        if index + 2 < suffix_index and tokens[index + 1] == "&" and tokens[index + 2].isdecimal():
            return (token, tokens[index + 2])
        return (token,)
    return ()


def is_address_component_number(tokens: tuple[str, ...], index: int) -> bool:
    previous_token = tokens[index - 1] if index > 0 else ""
    next_token = tokens[index + 1] if index + 1 < len(tokens) else ""
    return previous_token in {"section", "lane", "alley"} or next_token in {"f", "floor"}


def optional_parts_match(
    source_parts: dict[str, object],
    candidate_parts: dict[str, object],
    keys: tuple[str, ...],
) -> bool:
    for key in keys:
        source_value = source_parts[key]
        candidate_value = candidate_parts[key]
        if source_value is not None and candidate_value is not None and source_value != candidate_value:
            return False
    return True


def house_numbers_match(source_numbers: tuple[str, ...], candidate_numbers: tuple[str, ...]) -> bool:
    if not source_numbers or not candidate_numbers:
        return False
    return bool(set(source_numbers).intersection(candidate_numbers))


def row_coordinate(row: dict[str, object], column: str) -> float:
    try:
        return float(row_text(row, column))
    except ValueError as exc:
        raise ValueError(f"Taipei Michelin row has invalid {column}: value={row.get(column)}") from exc


def distance_between_coordinates_meters(
    first_latitude: float,
    first_longitude: float,
    second_latitude: float,
    second_longitude: float,
) -> float:
    first_latitude_radians = math.radians(first_latitude)
    second_latitude_radians = math.radians(second_latitude)
    latitude_delta = math.radians(second_latitude - first_latitude)
    longitude_delta = math.radians(second_longitude - first_longitude)
    haversine_value = (
        math.sin(latitude_delta / 2.0) ** 2
        + math.cos(first_latitude_radians)
        * math.cos(second_latitude_radians)
        * math.sin(longitude_delta / 2.0) ** 2
    )
    return EARTH_RADIUS_METERS * 2.0 * math.atan2(math.sqrt(haversine_value), math.sqrt(1.0 - haversine_value))


def primary_address_number(address: str) -> str | None:
    for token in re.findall(r"\d+[A-Za-z]*", address):
        if token.casefold().endswith("f"):
            continue
        return numeric_token_digits(token)
    return None


def address_numbers(address: str) -> set[str]:
    return {numeric_token_digits(token) for token in re.findall(r"\d+[A-Za-z]*", address)}


def numeric_token_digits(token: str) -> str:
    return "".join(character for character in token if character.isdigit())


def validate_query_mode(query_mode: str) -> None:
    if query_mode not in QUERY_MODES:
        raise ValueError(f"Unsupported query_mode={query_mode}; expected one of {QUERY_MODES}")


def output_row(row: dict[str, object], place_id: str) -> dict[str, object]:
    if not place_id.strip():
        raise ValueError("Google Places returned a blank place_id")
    return {
        "place_id": place_id,
        "name": row_text(row, "Name"),
        "michelin_category": row_text(row, "Award"),
        "address": row_text(row, "Address"),
        "location": row_text(row, "Location"),
        "latitude": row_text(row, "Latitude"),
        "longitude": row_text(row, "Longitude"),
        "michelin_url": row_text(row, "Url"),
    }


def unresolved_row(row: dict[str, object], text_query: str) -> dict[str, object]:
    return {
        "name": row_text(row, "Name"),
        "michelin_category": row_text(row, "Award"),
        "address": row_text(row, "Address"),
        "location": row_text(row, "Location"),
        "latitude": row_text(row, "Latitude"),
        "longitude": row_text(row, "Longitude"),
        "michelin_url": row_text(row, "Url"),
        "phone_number": row_text(row, "PhoneNumber"),
        "website_url": row_text(row, "WebsiteUrl"),
        "text_query": text_query,
    }


def unresolved_csv_path(output_csv_path: Path) -> Path:
    return output_csv_path.with_name(f"{output_csv_path.stem}.unresolved{output_csv_path.suffix}")


def validation_report_csv_path(unresolved_output_csv_path: Path) -> Path:
    return unresolved_output_csv_path.with_name(
        unresolved_output_csv_path.name.replace(".unresolved.csv", ".validation-report.csv")
    )


def row_key(row: dict[str, object]) -> tuple[str, str, str, str]:
    return (
        row_text(row, "name"),
        row_text(row, "address"),
        row_text(row, "location"),
        row_text(row, "michelin_url"),
    )


def row_text(row: dict[str, object], column: str) -> str:
    value = row.get(column, "")
    if pd.isna(value):
        return ""
    return str(value).strip()

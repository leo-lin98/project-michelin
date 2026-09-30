from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

from restaurant_ingestion.clients.google_places import GooglePlacesClient
from restaurant_ingestion.clients.place_id_llm_judge import VertexPlaceIdLlmJudgeClient
from restaurant_ingestion.config import (
    GOOGLE_API_KEY_ENV_VAR,
    GOOGLE_MAX_RETRIES,
    GOOGLE_REGION_CODE,
    GOOGLE_REQUEST_TIMEOUT_SECONDS,
    GOOGLE_RETRY_INITIAL_BACKOFF_SECONDS,
    GOOGLE_RETRY_MAX_BACKOFF_SECONDS,
    MICHELIN_PLACE_ID_REQUEST_BUDGET,
    MICHELIN_PLACE_ID_LANGUAGE_CODE,
    MICHELIN_PLACE_ID_DATASET_CSV,
    MICHELIN_PLACE_ID_RESOLVED_DIR,
    MICHELIN_PLACE_ID_UNRESOLVED_DIR,
    PLACE_ID_LLM_MAX_RETRIES,
    PLACE_ID_LLM_MODEL,
    PLACE_ID_LLM_PROJECT_ENV_VAR,
    PLACE_ID_LLM_REGION_ENV_VAR,
    PLACE_ID_LLM_RETRY_BACKOFF_SECONDS,
    PLACE_ID_LLM_USE_VERTEX_ENV_VAR,
)
from restaurant_ingestion.ingestion.michelin_place_ids import (
    QUERY_MODE_SIMPLE,
    QUERY_MODES,
    build_processed_place_id_dataset,
    latest_unresolved_csv_path,
    next_retry_output_path,
    plan_michelin_place_id_lookup,
    processed_place_id_input_paths,
    resolve_michelin_place_ids_with_matcher,
    resolve_michelin_place_ids_with_matcher_to_paths,
    run_output_path,
    unresolved_output_path_for_resolved_output,
)

DEFAULT_QUERY_MODE = QUERY_MODE_SIMPLE


def main() -> None:
    parser = argparse.ArgumentParser(description="Add Google Places place_id values to Taipei and New Taipei Michelin rows.")
    parser.add_argument("--source-csv")
    parser.add_argument("--output-csv")
    parser.add_argument("--run-stem")
    parser.add_argument("--retry-latest-unresolved", action="store_true")
    parser.add_argument("--stack-resolved", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--confirm-live-api-cost", action="store_true")
    parser.add_argument(
        "--query-mode",
        choices=QUERY_MODES,
        default=DEFAULT_QUERY_MODE,
        help="Primary search query. If a row stays manual_review, the complementary query is fired and candidates are unioned before re-adjudicating.",
    )
    parser.add_argument("--llm-judge", action="store_true")
    args = parser.parse_args()

    if args.stack_resolved:
        if args.dry_run:
            print(plan_stack_resolved())
            return
        print(
            build_processed_place_id_dataset(
                MICHELIN_PLACE_ID_RESOLVED_DIR,
                MICHELIN_PLACE_ID_UNRESOLVED_DIR,
                MICHELIN_PLACE_ID_DATASET_CSV,
            )
        )
        return

    source_csv_path, output_csv_path, unresolved_output_csv_path = resolve_command_paths(
        args.source_csv,
        args.output_csv,
        args.run_stem,
        args.retry_latest_unresolved,
    )

    if args.dry_run:
        plan = plan_michelin_place_id_lookup(source_csv_path)
        print(
            {
                **plan,
                "source_csv": str(source_csv_path),
                "output_csv": str(output_csv_path),
                "unresolved_csv": str(unresolved_output_csv_path),
                "llm_judge": args.llm_judge,
            }
        )
        return

    request_budget = validate_live_run_args(args.max_requests, args.confirm_live_api_cost)
    require_doppler_runtime()
    client = build_client()
    judge = build_llm_judge().judge_place_id if args.llm_judge else None
    try:
        if args.output_csv is not None and args.run_stem is None and not args.retry_latest_unresolved:
            print(
                resolve_michelin_place_ids_with_matcher(
                    source_csv_path,
                    output_csv_path,
                    client,
                    request_budget,
                    args.query_mode,
                    judge,
                )
            )
            return
        print(
            resolve_michelin_place_ids_with_matcher_to_paths(
                source_csv_path,
                output_csv_path,
                unresolved_output_csv_path,
                client,
                request_budget,
                args.query_mode,
                judge,
            )
        )
    finally:
        client.close()


def resolve_command_paths(
    source_csv: str | None,
    output_csv: str | None,
    run_stem: str | None,
    retry_latest_unresolved: bool,
) -> tuple[Path, Path, Path]:
    if retry_latest_unresolved:
        validate_retry_mode_args(source_csv, output_csv, run_stem)
        source_csv_path = latest_unresolved_csv_path(MICHELIN_PLACE_ID_UNRESOLVED_DIR)
        output_csv_path = next_retry_output_path(MICHELIN_PLACE_ID_RESOLVED_DIR)
        unresolved_output_csv_path = unresolved_output_path_for_resolved_output(
            output_csv_path,
            MICHELIN_PLACE_ID_UNRESOLVED_DIR,
        )
        return source_csv_path, output_csv_path, unresolved_output_csv_path

    if run_stem is not None:
        validate_run_stem_mode_args(source_csv, output_csv)
        output_csv_path = run_output_path(MICHELIN_PLACE_ID_RESOLVED_DIR, run_stem)
        unresolved_output_csv_path = unresolved_output_path_for_resolved_output(
            output_csv_path,
            MICHELIN_PLACE_ID_UNRESOLVED_DIR,
        )
        return Path(source_csv), output_csv_path, unresolved_output_csv_path

    if source_csv is None or output_csv is None:
        raise RuntimeError("Set --source-csv and --output-csv, or use --run-stem, --retry-latest-unresolved, or --stack-resolved")

    output_csv_path = Path(output_csv)
    return Path(source_csv), output_csv_path, output_csv_path.with_name(f"{output_csv_path.stem}.unresolved{output_csv_path.suffix}")


def validate_retry_mode_args(source_csv: str | None, output_csv: str | None, run_stem: str | None) -> None:
    if source_csv is not None or output_csv is not None or run_stem is not None:
        raise RuntimeError("--retry-latest-unresolved cannot be combined with --source-csv, --output-csv, or --run-stem")


def validate_run_stem_mode_args(source_csv: str | None, output_csv: str | None) -> None:
    if source_csv is None:
        raise RuntimeError("--run-stem requires --source-csv")
    if output_csv is not None:
        raise RuntimeError("--run-stem cannot be combined with --output-csv")


def plan_stack_resolved() -> dict[str, str | int | bool]:
    try:
        input_paths = processed_place_id_input_paths(MICHELIN_PLACE_ID_RESOLVED_DIR, MICHELIN_PLACE_ID_UNRESOLVED_DIR)
    except (FileNotFoundError, ValueError) as exc:
        return {
            "resolved_dir": str(MICHELIN_PLACE_ID_RESOLVED_DIR),
            "unresolved_dir": str(MICHELIN_PLACE_ID_UNRESOLVED_DIR),
            "dataset_csv": str(MICHELIN_PLACE_ID_DATASET_CSV),
            "input_files": 0,
            "ready": False,
            "reason": str(exc),
        }
    return {
        "resolved_dir": str(MICHELIN_PLACE_ID_RESOLVED_DIR),
        "unresolved_dir": str(MICHELIN_PLACE_ID_UNRESOLVED_DIR),
        "dataset_csv": str(MICHELIN_PLACE_ID_DATASET_CSV),
        "input_files": len(input_paths),
        "mode": "base-only" if len(input_paths) == 1 else "stack-resolved",
        "ready": True,
    }


def validate_live_run_args(max_requests: int | None, confirm_live_api_cost: bool) -> int:
    if not confirm_live_api_cost:
        raise RuntimeError("Live Google Places calls are disabled without --confirm-live-api-cost")
    if max_requests is None:
        raise RuntimeError("Set --max-requests for this paid Google Places run")
    if max_requests < 1:
        raise RuntimeError("--max-requests must be at least 1")
    if max_requests > MICHELIN_PLACE_ID_REQUEST_BUDGET:
        raise RuntimeError(f"--max-requests must be <= {MICHELIN_PLACE_ID_REQUEST_BUDGET}")
    return max_requests


def require_doppler_runtime() -> None:
    if not os.environ.get("DOPPLER_PROJECT") or not os.environ.get("DOPPLER_CONFIG"):
        raise RuntimeError("Run this command through Doppler: doppler run -- python3 scripts/add_michelin_place_ids.py")


def build_client() -> GooglePlacesClient:
    return GooglePlacesClient.from_environment(
        GOOGLE_API_KEY_ENV_VAR,
        MICHELIN_PLACE_ID_LANGUAGE_CODE,
        GOOGLE_REGION_CODE,
        GOOGLE_REQUEST_TIMEOUT_SECONDS,
        GOOGLE_MAX_RETRIES,
        GOOGLE_RETRY_INITIAL_BACKOFF_SECONDS,
        GOOGLE_RETRY_MAX_BACKOFF_SECONDS,
    )


def build_llm_judge() -> VertexPlaceIdLlmJudgeClient:
    return VertexPlaceIdLlmJudgeClient.from_environment(
        PLACE_ID_LLM_PROJECT_ENV_VAR,
        PLACE_ID_LLM_REGION_ENV_VAR,
        PLACE_ID_LLM_USE_VERTEX_ENV_VAR,
        PLACE_ID_LLM_MODEL,
        PLACE_ID_LLM_MAX_RETRIES,
        PLACE_ID_LLM_RETRY_BACKOFF_SECONDS,
    )


if __name__ == "__main__":
    main()

"""Request progress reporting helpers."""

import json
import sys


def request_progress_report(stage: str, completed: int, total: int, elapsed_seconds: float) -> dict[str, object]:
    """Summarise how far a run has got, and how fast, so a stall is distinguishable from slow work.

    A throughput and an ETA are what separate the two: a run that is merely slow keeps
    emitting reports with a steady rate, while a stalled one stops emitting entirely.
    """
    calls_per_second = completed / elapsed_seconds if elapsed_seconds > 0 else 0.0
    remaining = max(total - completed, 0)
    return {
        "stage": stage,
        "completed": completed,
        "total": total,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "calls_per_second": round(calls_per_second, 2),
        "eta_seconds": round(remaining / calls_per_second) if calls_per_second > 0 else None,
    }


def print_request_progress(stage: str, completed: int, total: int, elapsed_seconds: float) -> None:
    """Write a progress report to stderr, keeping stdout a clean JSON result document."""
    print(json.dumps(request_progress_report(stage, completed, total, elapsed_seconds)), file=sys.stderr, flush=True)


def crossed_progress_marks(previous_count: int, current_count: int, interval: int) -> tuple[int, ...]:
    if interval <= 0:
        raise ValueError("progress interval must be positive")
    first_mark = ((previous_count // interval) + 1) * interval
    marks: list[int] = []
    current_mark = first_mark
    while current_mark <= current_count:
        marks.append(current_mark)
        current_mark += interval
    return tuple(marks)


def print_crossed_request_progress(
    stage: str,
    previous_count: int,
    current_count: int,
    request_budget: int,
    interval: int,
) -> None:
    for mark in crossed_progress_marks(previous_count, current_count, interval):
        print({"stage": stage, "requests_used": mark, "request_budget": request_budget}, flush=True)

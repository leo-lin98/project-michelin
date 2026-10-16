"""LLM adjudication boundary for unresolved Michelin place_id matches."""

from collections.abc import Callable
import json
from typing import Literal, cast

from restaurant_ingestion.ingestion.place_id_scoring import MatchDecision, decide_place_id_match, eligible_candidates

LLM_ELIGIBLE_OUTCOMES = frozenset(("manual_review", "uncertain"))
LLM_JUDGMENT_KEYS = frozenset(("recommended_outcome", "recommended_place_id", "confidence", "evidence", "conflicts"))
PLACE_ID_OUTCOMES = frozenset(("place_id", "manual_review", "no_safe_match"))
CONFIDENCE_VALUES = frozenset(("low", "medium", "high"))
type LlmRecommendedOutcome = Literal["place_id", "manual_review", "no_safe_match"]
type LlmConfidence = Literal["low", "medium", "high"]
type LlmPlaceIdJudgment = dict[str, str | tuple[str, ...]]
type PlaceIdLlmJudge = Callable[[dict[str, object], tuple[dict[str, object], ...]], LlmPlaceIdJudgment]


def decide_place_id_match_with_llm(
    row: dict[str, object],
    fixture_row: dict[str, object],
    judge: PlaceIdLlmJudge,
) -> tuple[MatchDecision, LlmPlaceIdJudgment | None]:
    deterministic_decision = decide_place_id_match(row, fixture_row)
    if not outcome_eligible_for_llm(deterministic_decision.outcome):
        return deterministic_decision, None

    frozen_google_candidates = frozen_google_candidate_payloads(fixture_row)
    judgment = judge(row, frozen_google_candidates)
    adjudicated_decision = apply_llm_judgment(deterministic_decision, judgment)
    return adjudicated_decision, judgment


def apply_llm_judgment(
    deterministic_decision: MatchDecision,
    judgment: LlmPlaceIdJudgment | None,
) -> MatchDecision:
    if judgment is None or not outcome_eligible_for_llm(deterministic_decision.outcome):
        return deterministic_decision

    high_confidence = llm_judgment_confidence(judgment) == "high"
    recommended_outcome = llm_judgment_outcome(judgment)

    # A non-match recommendation (manual_review / no_safe_match) is terminal only when the
    # judge is highly confident; otherwise leave the row in manual_review rather than letting a
    # weakly held opinion force the outcome.
    if recommended_outcome != "place_id":
        if not high_confidence:
            return decision_with_reason(deterministic_decision, f"{deterministic_decision.reason};llm_not_high_confidence")
        return MatchDecision(
            recommended_outcome,
            "",
            f"{deterministic_decision.reason};llm_adjudicated",
            deterministic_decision.candidates,
        )

    recommended_place_id = llm_judgment_place_id(judgment)
    candidate_place_ids = frozenset(candidate.place_id for candidate in deterministic_decision.candidates)
    if recommended_place_id not in candidate_place_ids:
        return decision_with_reason(deterministic_decision, f"{deterministic_decision.reason};llm_place_id_not_in_candidates")

    # Bind a place_id on corroboration, not on a single confidence token: accept when the judge
    # is highly confident, or when its pick agrees with the deterministic top-ranked candidate.
    if not high_confidence and recommended_place_id != top_ranked_place_id(deterministic_decision):
        return decision_with_reason(deterministic_decision, f"{deterministic_decision.reason};llm_not_high_confidence")

    return MatchDecision(
        "place_id",
        recommended_place_id,
        f"{deterministic_decision.reason};llm_adjudicated",
        deterministic_decision.candidates,
    )


def build_place_id_judge_prompt(
    source_row: dict[str, object],
    frozen_google_candidates: tuple[dict[str, object], ...],
) -> str:
    payload = {
        "source_row": source_row,
        "frozen_google_candidates": frozen_google_candidates,
    }
    return (
        "You are adjudicating one Michelin Guide restaurant row against frozen Google Places candidates.\n"
        "Use only the supplied source row and Google candidate payloads. Do not invent facts.\n"
        "Return exactly one JSON object with this schema:\n"
        "{"
        "\"recommended_outcome\":\"place_id\"|\"manual_review\"|\"no_safe_match\","
        "\"recommended_place_id\":string,"
        "\"confidence\":\"low\"|\"medium\"|\"high\","
        "\"evidence\":[string],"
        "\"conflicts\":[string]"
        "}\n"
        "Set recommended_place_id to an empty string unless recommended_outcome is place_id.\n"
        "Case:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def parse_place_id_llm_judgment(response_text: str) -> LlmPlaceIdJudgment:
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM place_id judge returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("LLM place_id judge response must be a JSON object")
    return validate_place_id_llm_judgment(parsed)


def validate_place_id_llm_judgment(payload: dict[str, object]) -> LlmPlaceIdJudgment:
    payload_keys = frozenset(payload)
    if payload_keys != LLM_JUDGMENT_KEYS:
        raise ValueError(
            "LLM place_id judge response schema mismatch: "
            f"missing={sorted(LLM_JUDGMENT_KEYS.difference(payload_keys))}, "
            f"unexpected={sorted(payload_keys.difference(LLM_JUDGMENT_KEYS))}"
        )

    recommended_outcome = validate_literal_text(payload["recommended_outcome"], PLACE_ID_OUTCOMES, "recommended_outcome")
    recommended_place_id = validate_text(payload["recommended_place_id"], "recommended_place_id")
    confidence = validate_literal_text(payload["confidence"], CONFIDENCE_VALUES, "confidence")
    evidence = validate_text_tuple(payload["evidence"], "evidence")
    conflicts = validate_text_tuple(payload["conflicts"], "conflicts")

    if recommended_outcome == "place_id" and not recommended_place_id:
        raise ValueError("recommended_place_id is required when recommended_outcome is place_id")
    if recommended_outcome != "place_id" and recommended_place_id:
        raise ValueError("recommended_place_id must be blank unless recommended_outcome is place_id")

    return {
        "recommended_outcome": recommended_outcome,
        "recommended_place_id": recommended_place_id,
        "confidence": confidence,
        "evidence": evidence,
        "conflicts": conflicts,
    }


def llm_judgment_to_report(judgment: LlmPlaceIdJudgment) -> dict[str, object]:
    return {
        "recommended_outcome": llm_judgment_outcome(judgment),
        "recommended_place_id": llm_judgment_place_id(judgment),
        "confidence": llm_judgment_confidence(judgment),
        "evidence": list(llm_judgment_text_tuple(judgment, "evidence")),
        "conflicts": list(llm_judgment_text_tuple(judgment, "conflicts")),
    }


def llm_judgment_outcome(judgment: LlmPlaceIdJudgment) -> LlmRecommendedOutcome:
    value = llm_judgment_text(judgment, "recommended_outcome")
    if value not in PLACE_ID_OUTCOMES:
        raise ValueError(f"Invalid LLM recommended_outcome: value={value}")
    return cast(LlmRecommendedOutcome, value)


def llm_judgment_confidence(judgment: LlmPlaceIdJudgment) -> LlmConfidence:
    value = llm_judgment_text(judgment, "confidence")
    if value not in CONFIDENCE_VALUES:
        raise ValueError(f"Invalid LLM confidence: value={value}")
    return cast(LlmConfidence, value)


def llm_judgment_place_id(judgment: LlmPlaceIdJudgment) -> str:
    return llm_judgment_text(judgment, "recommended_place_id")


def llm_judgment_text_tuple(judgment: LlmPlaceIdJudgment, key: str) -> tuple[str, ...]:
    value = judgment[key]
    if not isinstance(value, tuple):
        raise ValueError(f"LLM judgment field must be a tuple: key={key}")
    return value


def llm_judgment_text(judgment: LlmPlaceIdJudgment, key: str) -> str:
    value = judgment[key]
    if not isinstance(value, str):
        raise ValueError(f"LLM judgment field must be text: key={key}")
    return value


def validate_literal_text(value: object, allowed_values: frozenset[str], field_name: str) -> str:
    text = validate_text(value, field_name)
    if text not in allowed_values:
        raise ValueError(f"Invalid LLM place_id judge field: field={field_name}, value={text}")
    return text


def validate_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"LLM place_id judge field must be text: field={field_name}")
    return value.strip()


def validate_text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"LLM place_id judge field must be a list: field={field_name}")
    return tuple(validate_text(item, field_name) for item in value)


def frozen_google_candidate_payloads(fixture_row: dict[str, object]) -> tuple[dict[str, object], ...]:
    return tuple(dict(candidate) for candidate in eligible_candidates(fixture_row))


def top_ranked_place_id(decision: MatchDecision) -> str:
    if not decision.candidates:
        return ""
    return decision.candidates[0].place_id


def outcome_eligible_for_llm(outcome: str) -> bool:
    return outcome in LLM_ELIGIBLE_OUTCOMES


def decision_with_reason(decision: MatchDecision, reason: str) -> MatchDecision:
    return MatchDecision(decision.outcome, decision.place_id, reason, decision.candidates)

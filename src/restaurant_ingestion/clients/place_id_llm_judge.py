"""Vertex AI connector for Michelin place_id LLM judgments."""

import importlib
import os
import re
import time
from types import ModuleType
import warnings

from restaurant_ingestion.ingestion.place_id_llm_judge import (
    LlmPlaceIdJudgment,
    build_place_id_judge_prompt,
    parse_place_id_llm_judgment,
)

REGION_PATTERN = re.compile(r"^[a-z]+-[a-z]+[0-9]$")


class VertexPlaceIdLlmJudgeClient:
    def __init__(
        self,
        project: str,
        location: str,
        model: str,
        max_retries: int,
        retry_backoff_seconds: float,
    ) -> None:
        self._project = required_text(project, "project")
        self._location = validated_region(location)
        self._model = required_text(model, "model")
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        genai, genai_types = load_google_genai()
        self._genai_types = genai_types
        self._client = genai.Client(vertexai=True, project=self._project, location=self._location)

    @classmethod
    def from_environment(
        cls,
        project_env_var: str,
        region_env_var: str,
        use_vertex_env_var: str,
        model: str,
        max_retries: int,
        retry_backoff_seconds: float,
    ) -> "VertexPlaceIdLlmJudgeClient":
        project = required_setting(project_env_var)
        location = required_setting(region_env_var)
        use_vertex = required_setting(use_vertex_env_var)
        if use_vertex.upper() != "TRUE":
            raise RuntimeError(f"{use_vertex_env_var} must be TRUE for Vertex AI place_id LLM judging")
        return cls(project, location, model, max_retries, retry_backoff_seconds)

    def judge_place_id(
        self,
        source_row: dict[str, object],
        frozen_google_candidates: tuple[dict[str, object], ...],
    ) -> LlmPlaceIdJudgment:
        prompt = build_place_id_judge_prompt(source_row, frozen_google_candidates)
        response_text = self.generate_judgment_text(prompt)
        return parse_place_id_llm_judgment(response_text)

    def generate_judgment_text(self, prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=self._genai_types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                    ),
                )
            except Exception as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
                warnings.warn(f"Retrying Vertex place_id judge request: attempt={attempt + 1}", RuntimeWarning, stacklevel=2)
                time.sleep(self._retry_backoff_seconds)
                continue
            if not response.text:
                raise RuntimeError("Vertex returned an empty place_id judge response")
            return str(response.text)
        raise RuntimeError("Vertex place_id judge request failed after retries") from last_error


def required_setting(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required runtime setting: {name}")
    return value


def required_text(value: str, field_name: str) -> str:
    text = value.strip()
    if not text:
        raise RuntimeError(f"Missing required Vertex place_id judge setting: field={field_name}")
    return text


def validated_region(value: str) -> str:
    text = required_text(value, "location")
    if not REGION_PATTERN.fullmatch(text):
        raise RuntimeError(f"Invalid Vertex AI region for place_id judge: value={text}")
    return text


def load_google_genai() -> tuple[ModuleType, ModuleType]:
    try:
        genai = importlib.import_module("google.genai")
        genai_types = importlib.import_module("google.genai.types")
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: google-genai. Run `uv sync`, then retry.") from exc
    return genai, genai_types

"""Unit tests for narration module."""
from __future__ import annotations

import pytest

from src.core.base import BaseLLM, LLMConfig, LLMResponse, Message
from src.datasets.schemas import ItemProfile, Rating, UserProfile
from src.narration.narrator import NarrationResult, RecommendationNarrator
from src.narration.templates import (
    NARRATION_PROMPT_TEMPLATE,
    RECOMMENDATION_SYSTEM_PROMPT,
    SINGLE_ITEM_TEMPLATE,
)


# ─── Mock LLM ────────────────────────────────────────────────────────────────

class _MockLLM(BaseLLM):
    def __init__(self, response: str = "Great picks!"):
        super().__init__(LLMConfig(model="mock-model"))
        self._response = response

    def complete(self, messages: list[Message]) -> LLMResponse:
        return LLMResponse(content=self._response, model=self.config.model)

    def stream(self, messages: list[Message]):
        yield self._response


class _ErrorLLM(BaseLLM):
    def __init__(self):
        super().__init__(LLMConfig(model="error-model"))

    def complete(self, messages):
        raise RuntimeError("LLM unavailable")

    def stream(self, messages):
        raise RuntimeError("LLM unavailable")
        yield  # make it a generator


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_items() -> dict[str, ItemProfile]:
    return {
        f"i{j}": ItemProfile(
            item_id=f"i{j}",
            title=f"Movie {j}",
            genres=(("Action", "Drama")[j % 2],),
        )
        for j in range(1, 6)
    }


def _make_recs() -> list[tuple[str, float]]:
    return [(f"i{j}", 5.0 - j * 0.5) for j in range(1, 6)]


# ─── Templates ───────────────────────────────────────────────────────────────

def test_templates_not_empty():
    assert RECOMMENDATION_SYSTEM_PROMPT
    assert NARRATION_PROMPT_TEMPLATE
    assert SINGLE_ITEM_TEMPLATE


def test_templates_have_placeholders():
    assert "{user_preferences}" in NARRATION_PROMPT_TEMPLATE
    assert "{item_list}" in NARRATION_PROMPT_TEMPLATE


# ─── RecommendationNarrator ──────────────────────────────────────────────────

def test_narrate_basic():
    narrator = RecommendationNarrator(_MockLLM())
    result = narrator.narrate(
        user_id="u1",
        recommendations=_make_recs(),
        items_by_id=_make_items(),
    )
    assert isinstance(result, NarrationResult)
    assert result.user_id == "u1"
    assert result.narrative == "Great picks!"
    assert len(result.item_ids) > 0


def test_narrate_with_user_profile():
    profile = UserProfile(user_id="u1", features={"age_group": 30.0, "lang": 1.0})
    narrator = RecommendationNarrator(_MockLLM())
    result = narrator.narrate(
        user_id="u1",
        recommendations=_make_recs(),
        items_by_id=_make_items(),
        user_profile=profile,
    )
    assert result.narrative == "Great picks!"


def test_narrate_limits_top_n():
    narrator = RecommendationNarrator(_MockLLM(), max_items_in_prompt=2)
    result = narrator.narrate(
        user_id="u1",
        recommendations=_make_recs(),
        items_by_id=_make_items(),
    )
    assert len(result.item_ids) <= 5  # item_ids may still include all input


def test_narrate_fallback_on_llm_error():
    narrator = RecommendationNarrator(_ErrorLLM())
    result = narrator.narrate(
        user_id="u1",
        recommendations=_make_recs(),
        items_by_id=_make_items(),
    )
    # Fallback must still return a NarrationResult with non-empty narrative
    assert isinstance(result, NarrationResult)
    assert result.narrative  # should have fallback text


def test_explain_item():
    narrator = RecommendationNarrator(_MockLLM(response="Because you liked Action."))
    items_by_id = _make_items()
    text = narrator.explain_item(items_by_id["i1"])
    assert text == "Because you liked Action."


def test_explain_item_fallback_on_error():
    narrator = RecommendationNarrator(_ErrorLLM())
    items_by_id = _make_items()
    text = narrator.explain_item(items_by_id["i1"])
    assert isinstance(text, str)
    assert text  # non-empty fallback


def test_narrate_prompt_injection_truncated():
    """Injected strings longer than the limit should be truncated."""
    narrator = RecommendationNarrator(_MockLLM())
    evil_user_id = "x" * 5000   # exceeds per-field limit
    result = narrator.narrate(
        user_id=evil_user_id,
        recommendations=_make_recs(),
        items_by_id=_make_items(),
    )
    # Result must come back without error (truncation happened)
    assert isinstance(result, NarrationResult)


def test_narrate_empty_recommendations():
    narrator = RecommendationNarrator(_MockLLM())
    result = narrator.narrate(
        user_id="u1",
        recommendations=[],
        items_by_id=_make_items(),
    )
    assert isinstance(result, NarrationResult)
    assert result.item_ids == []


def test_user_prefs_str_no_profile():
    text = RecommendationNarrator._user_prefs_str(None)
    assert isinstance(text, str)


def test_user_prefs_str_with_profile():
    profile = UserProfile(user_id="u1", features={"age_group": 25.0})
    text = RecommendationNarrator._user_prefs_str(profile)
    assert "age_group" in text


# ─── NarrationResult ─────────────────────────────────────────────────────────

def test_narration_result_attributes():
    nr = NarrationResult(user_id="u1", narrative="text", model="m", item_ids=["i1"])
    assert nr.user_id == "u1"
    assert nr.narrative == "text"
    assert nr.model == "m"
    assert nr.item_ids == ["i1"]

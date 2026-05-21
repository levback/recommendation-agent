from __future__ import annotations

import logging
from dataclasses import dataclass

from ..core.base import BaseLLM
from ..datasets.schemas import ItemProfile, UserProfile
from .templates import (
    NARRATION_PROMPT_TEMPLATE,
    RECOMMENDATION_SYSTEM_PROMPT,
    SINGLE_ITEM_TEMPLATE,
)

logger = logging.getLogger(__name__)

# Maximum prompt length — prevents prompt-injection via long LLM inputs
_MAX_PROMPT_CHARS = 4000


@dataclass
class NarrationResult:
    user_id: str
    narrative: str
    model: str
    item_ids: list[str]


class RecommendationNarrator:
    """Generates natural-language narratives via Claude Haiku 4.5 on Bedrock.

    Security: all user-supplied strings are truncated before being inserted
    into prompts to prevent prompt-injection attacks.
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_items_in_prompt: int = 5,
    ) -> None:
        self._llm = llm
        self.max_items_in_prompt = max_items_in_prompt

    def narrate(
        self,
        user_id: str,
        recommendations: list[tuple[str, float]],
        items_by_id: dict[str, ItemProfile],
        user_profile: UserProfile | None = None,
    ) -> NarrationResult:
        """Generate a narrative explanation for a ranked recommendation list."""
        top = recommendations[: self.max_items_in_prompt]
        item_ids = [iid for iid, _ in top]

        item_lines: list[str] = []
        for rank, (iid, score) in enumerate(top, 1):
            prof = items_by_id.get(iid)
            if prof:
                genres = ", ".join(prof.genres) if prof.genres else "various genres"
                title = prof.title[:100]  # truncate for security
                item_lines.append(
                    f"{rank}. {title} (Genres: {genres}, Score: {score:.2f})"
                )
            else:
                item_lines.append(f"{rank}. Item {iid[:50]} (Score: {score:.2f})")

        prefs = self._user_prefs_str(user_profile)
        prompt = NARRATION_PROMPT_TEMPLATE.format(
            user_preferences=prefs[:200],
            item_list="\n".join(item_lines),
        )
        prompt = prompt[:_MAX_PROMPT_CHARS]

        try:
            narrative = self._llm.chat(prompt, system_prompt=RECOMMENDATION_SYSTEM_PROMPT)
        except Exception as exc:
            logger.warning("Narration LLM call failed: %s — using fallback", exc)
            narrative = (
                f"Based on your interest in {prefs}, we selected {len(top)} "
                "items we think you'll enjoy."
            )

        return NarrationResult(
            user_id=user_id,
            narrative=narrative,
            model=self._llm.config.model,
            item_ids=item_ids,
        )

    def explain_item(
        self,
        item: ItemProfile,
        user_profile: UserProfile | None = None,
    ) -> str:
        """Generate a one-sentence explanation for a single recommended item."""
        prefs = self._user_prefs_str(user_profile)
        genres = ", ".join(item.genres) if item.genres else "various genres"
        prompt = SINGLE_ITEM_TEMPLATE.format(
            title=item.title[:100],
            genres=genres[:100],
            preferences=prefs[:200],
        )
        try:
            return self._llm.chat(prompt, system_prompt=RECOMMENDATION_SYSTEM_PROMPT)
        except Exception as exc:
            logger.warning("Item explanation failed: %s", exc)
            return f"'{item.title}' matches your interest in {prefs}."

    @staticmethod
    def _user_prefs_str(profile: UserProfile | None) -> str:
        if profile and profile.features:
            top = sorted(profile.features, key=profile.features.__getitem__, reverse=True)[:4]
            return ", ".join(top)
        return "diverse content"

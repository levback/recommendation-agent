"""Example 04: Bedrock narration (requires AWS credentials).

Run with:
    AWS_PROFILE=your_profile python examples/04_bedrock_narration.py

Falls back to a placeholder if credentials are unavailable.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.loader import DatasetLoader
from src.datasets.preprocessor import DatasetPreprocessor
from src.recommender.hybrid import HybridConfig, HybridRecommender
from src.narration.narrator import RecommendationNarrator


def main() -> None:
    loader = DatasetLoader()
    ratings, items, users = loader.generate_synthetic(
        n_users=30, n_items=60, n_ratings=600, seed=99
    )
    pp = DatasetPreprocessor(test_ratio=0.2, seed=99)
    ds = pp.split(ratings)
    items_by_id = {it.item_id: it for it in items}

    hr = HybridRecommender(HybridConfig(n_recommendations=10))
    hr.fit(ds.train)
    uid = sorted({r.user_id for r in ds.train})[0]
    result = hr.recommend(uid, n=5)

    # Try real Bedrock; fall back to mock
    try:
        from src.core.factory import create_bedrock_llm
        llm = create_bedrock_llm()
        narrator = RecommendationNarrator(llm)
        print("Using real AWS Bedrock (Claude Haiku 4.5)…")
    except Exception as exc:
        print(f"Bedrock unavailable ({exc}); using mock LLM.")
        from src.core.base import BaseLLM, LLMConfig, LLMResponse, Message

        class _MockLLM(BaseLLM):
            def complete(self, messages):
                return LLMResponse(
                    content="Based on your viewing history, these films match your taste for compelling narratives and strong character development.",
                    model=self.config.model,
                )

            def stream(self, messages):
                yield self.complete(messages).content

        narrator = RecommendationNarrator(_MockLLM(LLMConfig(model="mock")))

    nr = narrator.narrate(
        user_id=uid,
        recommendations=result.recommendations,
        items_by_id=items_by_id,
    )

    output = {
        "user_id": uid,
        "recommendations": [{"item_id": iid, "score": round(s, 4)} for iid, s in result.recommendations],
        "narrative": nr.narrative,
        "model": nr.model,
    }
    print(json.dumps(output, indent=2))

    os.makedirs(os.path.join(os.path.dirname(__file__), "output"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "output", "04_bedrock_narration.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()

"""Example 06: Semantic (content-based) recommendations with FAISS.

Compares:
  - CF-only recommendations (matrix factorisation)
  - CF + RL recommendations
  - CF + RL + Semantic (content-based FAISS) recommendations

Uses real MovieLens data from HuggingFace and shows how genre-aware
embedding enriches ranking and narrator output.

Run:
    cd /Users/levent.ozparlak/Projects/recommendation_agent
    source .venv/bin/activate
    python examples/06_semantic_recommendations.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import create_bedrock_llm
from src.datasets.movielens import load_movielens_hf
from src.narration.narrator import RecommendationNarrator
from src.rag.item_retriever import ItemRetriever
from src.recommender.hybrid import CFMethod, HybridConfig, HybridRecommender

DEMO_USER = "1"
N_RECS = 10


def _print_recs(label: str, recs: list[tuple[str, float]], items_by_id: dict) -> None:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    for rank, (iid, score) in enumerate(recs[:5], 1):
        title = items_by_id[iid].title if iid in items_by_id else iid
        genres = ", ".join(items_by_id[iid].genres[:3]) if iid in items_by_id else ""
        print(f"  {rank}. {title[:45]:<45} [{genres}]  {score:.3f}")


def main() -> None:
    print(f"\n{'='*70}")
    print("  Semantic Recommendations — CF vs CF+RL vs CF+RL+Semantic")
    print(f"{'='*70}\n")

    print("Loading MovieLens data from HuggingFace…")
    dataset = load_movielens_hf()
    ratings = dataset.ratings
    items = dataset.items
    items_by_id = {item.item_id: item for item in items}

    print(f"  {len(ratings):,} ratings  |  {len(items):,} items  |  user={DEMO_USER}")

    # ── 1. CF only ────────────────────────────────────────────────────────────
    cfg_cf = HybridConfig(
        cf_method=CFMethod.MATRIX_FACTORIZATION,
        use_rl=False,
        use_semantic=False,
        n_recommendations=N_RECS,
    )
    rec_cf = HybridRecommender(cfg_cf).fit(ratings)
    result_cf = rec_cf.recommend(DEMO_USER, n=N_RECS)
    _print_recs("CF-only (MF)", result_cf.recommendations, items_by_id)

    # ── 2. CF + RL ────────────────────────────────────────────────────────────
    cfg_rl = HybridConfig(
        cf_method=CFMethod.MATRIX_FACTORIZATION,
        use_rl=True,
        use_semantic=False,
        n_recommendations=N_RECS,
    )
    rec_rl = HybridRecommender(cfg_rl).fit(ratings)
    result_rl = rec_rl.recommend(DEMO_USER, n=N_RECS)
    _print_recs("CF + RL (Thompson sampling)", result_rl.recommendations, items_by_id)

    # ── 3. CF + RL + Semantic ─────────────────────────────────────────────────
    print("\nBuilding FAISS semantic index on item profiles…")
    retriever = ItemRetriever()
    retriever.fit(items)
    print(f"  {retriever.item_count} items indexed")

    cfg_sem = HybridConfig(
        cf_method=CFMethod.MATRIX_FACTORIZATION,
        use_rl=True,
        use_semantic=True,
        semantic_weight=0.2,
        n_recommendations=N_RECS,
    )
    rec_sem = HybridRecommender(cfg_sem).fit(ratings, items=items)
    result_sem = rec_sem.recommend(DEMO_USER, n=N_RECS)
    _print_recs("CF + RL + Semantic (FAISS)", result_sem.recommendations, items_by_id)

    # ── 4. Show semantically similar items to top recommendation ──────────────
    top_id = result_sem.recommendations[0][0]
    top_title = items_by_id.get(top_id, type("o", (), {"title": top_id})()).title
    print(f"\n{'─'*60}")
    print(f"  Items semantically similar to top pick: '{top_title}'")
    print(f"{'─'*60}")
    similar = retriever.find_similar(top_id, n=5)
    for sid, score in similar:
        title = items_by_id[sid].title if sid in items_by_id else sid
        genres = ", ".join(items_by_id[sid].genres[:3]) if sid in items_by_id else ""
        print(f"  {title[:50]:<50} [{genres}]  sim={score:.3f}")

    # ── 5. Narrator with semantic context ────────────────────────────────────
    print("\nGenerating narration with semantic context…")
    llm = create_bedrock_llm()
    narrator = RecommendationNarrator(llm, item_retriever=retriever)
    narration = narrator.narrate(
        DEMO_USER,
        result_sem.recommendations,
        items_by_id,
    )
    print(f"\n{'─'*60}")
    print("  Narrator output:")
    print(f"{'─'*60}")
    print(narration.narrative[:800])

    print("\nDone.")


if __name__ == "__main__":
    main()

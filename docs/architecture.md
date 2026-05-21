# Architecture

Cross-references: [Overview](user-guide/overview.md) · [Main Components](user-guide/main-components.md) · [API Reference](reference/index.md)

---

## High-Level Design

The system is a three-layer hybrid recommender:

```
┌────────────────────────────────────────────────────────────────┐
│                     RecommendationPipeline                     │
│                                                                │
│  ┌──────────────┐   ┌───────────────────┐   ┌──────────────┐  │
│  │ Collaborative│   │    RL Bandits      │   │  Narration   │  │
│  │  Filtering   │──▶│  (online learning) │──▶│  (Bedrock)   │  │
│  └──────────────┘   └───────────────────┘   └──────────────┘  │
│         │                    │                                  │
│         └─────────┬──────────┘                                  │
│                   ▼                                             │
│           HybridRecommender                                     │
│       CF × 0.6  +  RL × 0.4  (configurable)                   │
└────────────────────────────────────────────────────────────────┘
```

---

## Component Diagram

```
src/
├── core/
│   ├── base.py              ← Message, LLMResponse, BaseLLM (ABC)
│   ├── bedrock_client.py    ← BedrockClient (Converse API)
│   └── factory.py           ← create_bedrock_llm()
│
├── datasets/
│   ├── schemas.py           ← Rating, UserProfile, ItemProfile, DatasetSplit
│   ├── loader.py            ← HuggingFace + synthetic fallback
│   └── preprocessor.py      ← train/test split, normalise, relevance sets
│
├── collaborative/
│   ├── base.py              ← BaseCollaborativeFilter (ABC)
│   ├── matrix_factorization.py ← biased SGD MF
│   ├── user_cf.py           ← cosine similarity neighbourhood
│   └── item_cf.py           ← adjusted cosine similarity
│
├── rl/
│   ├── bandit.py            ← ε-Greedy, UCB1, Thompson Sampling
│   ├── contextual_bandit.py ← LinUCB (per-arm ridge regression)
│   └── agent.py             ← RLRecommendationAgent (strategy pattern)
│
├── narration/
│   ├── templates.py         ← system prompt + narration templates
│   └── narrator.py          ← RecommendationNarrator
│
└── recommender/
    ├── evaluator.py         ← RMSE, MAE, Precision@K, Recall@K, NDCG@K
    ├── hybrid.py            ← HybridRecommender (CF + RL blend)
    └── pipeline.py          ← RecommendationPipeline (end-to-end)
```

---

## Data Flow

### Training phase

```
DatasetLoader.load_huggingface()
    │  (or generate_synthetic() on failure)
    ▼
DatasetPreprocessor.split()        → DatasetSplit(train, test)
    │
    ├──▶ CollaborativeFilter.fit(train)   ← learns user/item embeddings
    │
    └──▶ RLRecommendationAgent            ← initialised with item list
```

### Inference phase

```
HybridRecommender.recommend(user_id, n)
    │
    ├── CF.predict(user_id, item_id)   → raw rating score
    │       normalised to [0, 1]
    │
    ├── RLAgent.recommend(user_id, n)  → bandit arm selections
    │       normalised to [0, 1]
    │
    └── blended_score = cf × 0.6 + rl × 0.4
            → top-N items by blended score
            → RecommendationResult
```

### Online learning loop

```
user clicks item
    │
    ▼
HybridRecommender.update_feedback(user_id, item_id, reward=4.0)
    │
    └── RLAgent.update(item_id, reward_normalised)
            ← reward ÷ 5 → [0, 1] for bandit update
```

---

## Design Decisions

### Why biased SGD matrix factorization?

Standard latent factor models struggle with popularity bias and cold-start.
Biased SGD (user bias + item bias + global mean) captures systematic rating
differences without complex neural machinery, making it interpretable and fast
to train on CPU.

### Why bandits for online learning?

Bandits provide a principled exploration-exploitation trade-off for real-time
feedback without requiring a full supervised re-train cycle. Thompson Sampling
is used as the default strategy because it naturally handles uncertainty through
Bayesian posterior sampling.

### Why the Converse API (not InvokeModel)?

The Bedrock Converse API provides a unified interface across all Anthropic
models, handles message format validation, and natively supports multi-turn
conversations and streaming — all without model-specific JSON wrangling.

### Why frozen dataclasses for domain objects?

`Rating`, `UserProfile`, `ItemProfile`, and `LLMResponse` are all
`@dataclass(frozen=True)`. Immutable value objects prevent accidental mutation
in the recommendation pipeline where the same rating list is referenced by
multiple components.

---

## Security Architecture

| Concern | Mitigation |
|---------|------------|
| AWS credentials | boto3 chain (env → profile → IAM role). Never hardcoded. |
| Prompt injection | Narration input truncated to 4000 chars; system prompt enforces safe role |
| Dependency vulnerabilities | `safety check` in CI (`scripts/security_check.sh`) |
| Static analysis | `bandit -r src/` in CI; OWASP + NIST test suites |
| Secret leakage | `.env` in `.gitignore`; `.env.example` has no real values |

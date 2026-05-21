# Overview

> **Recommendation Agent** is a hybrid recommendation system combining
> collaborative filtering, reinforcement learning bandits, and Claude Haiku 4.5
> narration — all on a single, consistent Python abstraction.

Cross-references: [Get Started](get-started.md) · [Examples](../examples.md) · [API Reference](../reference/index.md)

---

## What does it do?

Given a dataset of user-item ratings, the system:

1. **Learns user preferences** via matrix factorization or neighbourhood-based CF
2. **Explores new items** via RL bandits that improve in real-time as users give feedback
3. **Blends both signals** into a single ranked list
4. **Narrates the recommendations** in plain English using Claude Haiku 4.5 on Bedrock

---

## The Three Layers

### Layer 1 — Collaborative Filtering

Learns from historical ratings to predict how much a user will like an unseen item.

| Algorithm | Approach | Best for |
|-----------|---------|---------|
| **Matrix Factorization** | Biased SGD; user/item latent factors | Large dense datasets |
| **User-CF** | Cosine similarity between user rating vectors | Finding taste twins |
| **Item-CF** | Adjusted cosine similarity between item vectors | Item-to-item recommendations |

### Layer 2 — RL Bandits (online learning)

Adapts in real-time to click / rating feedback without a full retrain.

| Strategy | Exploration | Best for |
|----------|------------|---------|
| **ε-Greedy** | Random with probability ε | Simple baseline |
| **UCB1** | Upper confidence bound | Deterministic exploration |
| **Thompson Sampling** | Beta posterior sampling | Bayesian uncertainty |
| **LinUCB** | Per-arm ridge regression on context | Context-aware recommendations |

### Layer 3 — Narration

`RecommendationNarrator` calls Claude Haiku 4.5 via the Bedrock Converse API
to produce a 2–3 sentence explanation of why each set of recommendations was
selected. Falls back to a templated explanation if Bedrock is unavailable.

---

## Repository layout

```
recommendation_agent/
├── src/
│   ├── core/              # LLM abstractions + Bedrock client
│   ├── datasets/          # Loader, preprocessor, schemas
│   ├── collaborative/     # MF, User-CF, Item-CF
│   ├── rl/                # Bandits + RLRecommendationAgent
│   ├── narration/         # Prompt templates + narrator
│   └── recommender/       # Evaluator, HybridRecommender, Pipeline
├── examples/              # 5 runnable scripts + pre-computed outputs
├── tests/                 # 200 tests (unit + integration + security)
├── docs/                  # This documentation
└── config/                # YAML configuration files
```

---

## Design principles

**Single `BaseLLM` abstraction**
All narration calls go through `BaseLLM.complete()`. Swapping Bedrock for any
other LLM is a one-line change.

**Frozen dataclasses everywhere**
`Rating`, `UserProfile`, `ItemProfile`, `LLMResponse` are all immutable.
Bugs from accidental mutation are impossible.

**Graceful degradation**
If the HuggingFace dataset is unavailable, synthetic data is generated.
If Bedrock is unavailable, a fallback narrative is returned.
Nothing hard-crashes in production.

# Documentation Index

This directory contains all technical documentation for the Recommendation Agent project.

---

## Documents

| Document | Description |
|---|---|
| [Installation](installation.md) | Environment setup, dependencies, AWS credentials |
| [Architecture](architecture.md) | System design, component diagram, data-flow, design decisions |
| [Configuration](configuration.md) | `model_config.yaml` reference, environment variables |
| [Examples](examples.md) | Walkthrough of all 5 runnable examples with sample outputs |
| [Development Guide](development.md) | Dev setup, testing, adding new algorithms, conventions |

For the quick-start and project overview, see the [root README](../README.md).

---

## System Overview

```
Dataset (HuggingFace MovieLens / Synthetic)
   │
   ▼
[Step 1] DatasetLoader + DatasetPreprocessor
         ← train/test split · normalize ratings · relevance sets
   │
   ▼
[Step 2] Collaborative Filtering
         ← MatrixFactorization (biased SGD) | UserBasedCF | ItemBasedCF
   │
   ▼
[Step 3] RL Bandit Layer
         ← ε-Greedy | UCB1 | Thompson Sampling | LinUCB
   │
   ▼
[Step 4] HybridRecommender
         ← CF score × cf_weight  +  RL score × (1 - cf_weight)
   │
   ▼
[Step 5] RecommendationNarrator  (optional)
         ← Claude Haiku 4.5 on Amazon Bedrock Converse API
   │
   ▼
RecommendationResult  →  JSON output · stdout report
```

---

## Project Structure

```
recommendation_agent/
├── config/
│   ├── model_config.yaml          # Bedrock model ID, recommender defaults
│   └── logging_config.yaml        # JSON + console logging
├── data/                          # Runtime data (gitignored)
├── examples/
│   ├── 01_collaborative_filtering.py
│   ├── 02_rl_bandit.py
│   ├── 03_hybrid_recommender.py
│   ├── 04_bedrock_narration.py
│   ├── 05_movielens_scenario.py
│   └── output/                    # Pre-computed JSON results
├── src/
│   ├── core/                      # BaseLLM, BedrockClient, factory
│   ├── datasets/                  # Loader, preprocessor, schemas
│   ├── collaborative/             # MF, User-CF, Item-CF
│   ├── rl/                        # Bandits + RLRecommendationAgent
│   ├── narration/                 # Prompt templates + narrator
│   └── recommender/               # Evaluator, HybridRecommender, Pipeline
├── tests/
│   ├── unit/                      # Unit tests (200 tests, 97%+ coverage)
│   ├── integration/               # End-to-end pipeline tests
│   └── security/                  # OWASP + NIST test suites
├── scripts/                       # Shell helpers
└── docs/                          # This documentation
    ├── user-guide/
    └── reference/
```

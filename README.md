# Recommendation Agent

A personalized recommendation system combining **Collaborative Filtering**, **Reinforcement Learning (bandits)**, and **AWS Bedrock (Claude Haiku 4.5) LLM narration**. Applied to the MovieLens dataset from HuggingFace Hub.

## Architecture

```
src/
├── core/             # BaseLLM, BedrockClient, factory
├── datasets/         # Schemas, loader (HuggingFace + synthetic), preprocessor
├── collaborative/    # Matrix Factorization, User-CF, Item-CF
├── rl/               # ε-Greedy, UCB, Thompson Sampling, LinUCB bandits + RLAgent
├── narration/        # Prompt templates + RecommendationNarrator (Bedrock)
└── recommender/      # HybridRecommender, RecommendationPipeline, Evaluator
```

### Key components

| Component | Description |
|---|---|
| `MatrixFactorization` | Biased SGD matrix factorization (20 factors, 20 epochs) |
| `UserBasedCF` / `ItemBasedCF` | Cosine / adjusted-cosine neighbourhood CF |
| `HybridRecommender` | CF (60%) + RL bandit (40%) blended scores |
| `RLRecommendationAgent` | Wraps any bandit; supports ε-greedy, UCB, Thompson, LinUCB |
| `RecommendationNarrator` | Claude Haiku 4.5 via Bedrock Converse API; prompt-injection safe |
| `RecommendationPipeline` | End-to-end: fit → recommend → narrate → evaluate |

## Quick Start

```bash
# 1. Set up environment
./scripts/setup_env.sh
source .venv/bin/activate

# 2. Configure AWS (for Bedrock narration)
cp .env.example .env
# edit .env: set AWS_PROFILE or AWS_ACCESS_KEY_ID etc.

# 3. Run the MovieLens scenario
python examples/05_movielens_scenario.py

# 4. Run CLI
python main.py --dataset movielens --cf-method mf --n-recs 10

# With narration (requires Bedrock)
python main.py --dataset movielens --narrate
```

## Installation

```bash
pip install -r requirements.txt
```

**Python 3.12+ required.**

## Running Tests

```bash
./scripts/run_tests.sh
# or directly:
pytest tests/ --cov=src --cov-fail-under=95
```

Test coverage target: **≥ 95%**

Test layout:
- `tests/unit/` — unit tests for all modules
- `tests/integration/` — end-to-end pipeline tests with synthetic data
- `tests/security/` — OWASP Top 10 + NIST SP 800-53 controls

## Examples

```bash
./scripts/run_examples.sh
```

| Script | Description |
|---|---|
| `01_collaborative_filtering.py` | MF, User-CF, Item-CF comparison |
| `02_rl_bandit.py` | Pure bandit simulation + RL agent |
| `03_hybrid_recommender.py` | Hybrid CF+RL with feedback loop |
| `04_bedrock_narration.py` | Claude Haiku 4.5 narrative generation |
| `05_movielens_scenario.py` | Full MovieLens end-to-end scenario |

Outputs are written to `examples/output/`.

## Security

```bash
./scripts/security_check.sh
```

Covers:
- **Bandit** static analysis (OWASP A03 injection, A05 misconfiguration)
- **Safety** dependency audit
- **OWASP tests**: A01 access control, A03 prompt injection, A04 input validation, A08 data integrity
- **NIST tests**: AC access control, AU audit logging, CM configuration, IA authentication, SC system protection, SI integrity

## Docker

```bash
docker compose up
```

## Configuration

`config/model_config.yaml` — model IDs, recommender hyperparameters  
`config/logging_config.yaml` — logging handlers  
`.env.example` — environment variable reference

## Dataset

Primary: `nateraw/movie-lens-latest-small` from HuggingFace Hub (100k ratings, 9k movies, 600 users).  
Automatic fallback to 2000-rating synthetic dataset if HuggingFace is unavailable.

## AWS Bedrock Setup

Model: `us.anthropic.claude-haiku-4-5-20250714-v1:0` (cross-region inference)

Credentials are **never hardcoded**. Provide via:
- `AWS_PROFILE` env var
- `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`
- IAM role (EC2/ECS/Lambda instance profile)

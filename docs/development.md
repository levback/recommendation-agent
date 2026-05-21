# Development Guide

Cross-references: [Installation](installation.md) · [Architecture](architecture.md) · [API Reference](reference/index.md)

---

## Dev Setup

```bash
git clone git@github.com:levback/recommendation-agent.git
cd recommendation_agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

---

## Running Tests

```bash
# All tests with coverage report
bash scripts/run_tests.sh

# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Security tests (OWASP + NIST)
pytest tests/security/ -v

# Coverage threshold (fails below 95%)
pytest --cov=src --cov-fail-under=95
```

Current status: **200 tests, 97.48% coverage**.

---

## Security Scanning

```bash
bash scripts/security_check.sh
# Runs bandit (static analysis) + safety (dependency audit)
```

---

## Adding a New Collaborative Filtering Algorithm

1. Create `src/collaborative/my_algo.py`
2. Subclass `BaseCollaborativeFilter` and implement `fit()`, `predict()`, `recommend()`
3. Add a `MyAlgoConfig` dataclass for hyperparameters
4. Add to `CFMethod` enum in `src/recommender/hybrid.py`
5. Handle the new case in `HybridRecommender._build_cf()`
6. Add unit tests in `tests/unit/test_collaborative.py`

---

## Adding a New Bandit Strategy

1. Subclass `BaseBandit` (or add to `src/rl/bandit.py`) and implement `select_arm()` and `update()`
2. Add to `BanditStrategy` enum in `src/rl/agent.py`
3. Handle the new case in `RLRecommendationAgent._build_bandit()`
4. Add unit tests in `tests/unit/test_rl.py`

---

## Swapping the LLM Backend

The narration layer accepts any `BaseLLM` subclass:

```python
from src.core.base import BaseLLM, LLMConfig, LLMResponse, Message

class MyLLM(BaseLLM):
    def __init__(self):
        super().__init__(LLMConfig(model="my-model"))

    def complete(self, messages: list[Message]) -> LLMResponse:
        # call your LLM here
        return LLMResponse(content="...", model=self.config.model)

    def stream(self, messages):
        yield self.complete(messages).content

narrator = RecommendationNarrator(MyLLM())
```

---

## Code Conventions

- **Python 3.12** — `from __future__ import annotations` in every file
- **Immutability** — domain objects (`Rating`, `LLMResponse`) are `frozen=True` dataclasses
- **No mutation** — always return new objects; never modify in-place
- **Type hints** — all public functions are fully annotated
- **File size** — keep files under 400 lines; split at natural boundaries
- **Error handling** — catch specific exceptions; log with context; never swallow silently

---

## Commit Convention

```
feat:     new feature
fix:      bug fix
test:     test additions or changes
docs:     documentation only
refactor: code restructuring
chore:    build, CI, deps
perf:     performance improvement
```

Example: `fix: wrap Bedrock message content as list of content blocks`

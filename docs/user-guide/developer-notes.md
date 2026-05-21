# Developer Notes

Cross-references: [Development Guide](../development.md) · [Architecture](../architecture.md)

---

## Testing Strategy

The test suite has three layers:

### Unit tests (`tests/unit/`)

| File | What it covers |
|------|---------------|
| `test_core.py` | `BaseLLM`, `BedrockClient`, `LLMConfig`, `Message` |
| `test_datasets.py` | `Rating` validation, `DatasetSplit`, `Preprocessor`, `Loader` |
| `test_collaborative.py` | MF convergence, User-CF, Item-CF edge cases |
| `test_rl.py` | Bandit arm selection, update logic, LinUCB regression |
| `test_narration.py` | Template rendering, prompt truncation, fallback |
| `test_recommender.py` | RMSE/MAE/ranking metrics, hybrid blending, pipeline |

### Integration tests (`tests/integration/`)

| File | What it covers |
|------|---------------|
| `test_pipeline.py` | Fit → recommend → evaluate → feedback loop |
| `test_end_to_end.py` | Full system scenario with mock LLM |

### Security tests (`tests/security/`)

| File | Standard |
|------|---------|
| `test_owasp.py` | OWASP Top 10 (injection, secrets, input validation) |
| `test_nist.py` | NIST SP 800-53 (auth, audit, config management) |

---

## Writing Mock LLMs in Tests

`BaseLLM.__init__` requires a `LLMConfig` argument. Always call `super().__init__()`:

```python
from src.core.base import BaseLLM, LLMConfig, LLMResponse, Message

class MockLLM(BaseLLM):
    def __init__(self):
        super().__init__(LLMConfig(model="mock"))

    def complete(self, messages: list[Message]) -> LLMResponse:
        return LLMResponse(content="mock response", model=self.config.model)

    def stream(self, messages: list[Message]):
        yield "mock response"
```

---

## Common Pitfalls

### `ds.users` is empty

`DatasetSplit.users` is a `list[UserProfile]` that is **only populated if you
pass `users=` to the preprocessor**. To get user IDs from training data, always use:

```python
user_ids = sorted({r.user_id for r in ds.train})
uid = user_ids[0]
```

### Bedrock message content must be a list

The Bedrock Converse API requires `messages[n].content` to be a list of content
blocks, not a plain string:

```python
# CORRECT
{"role": "user", "content": [{"text": "Hello"}]}

# WRONG — raises ValidationException
{"role": "user", "content": "Hello"}
```

`BedrockClient._split_system()` handles this automatically.

### Bandit `update()` must be called between `select_arm()` calls

UCB and other count-based bandits only update their internal counts in
`update()`, not in `select_arm()`. Always call `update()` after each selection
to ensure arms get their counts incremented.

---

## Security Notes

### AWS credentials

- Use IAM roles with least-privilege Bedrock permissions (`bedrock:InvokeModel`)
- For local dev, use `AWS_PROFILE` with a named profile
- Never pass credentials as function arguments or log them

### Prompt injection

- All user-supplied content passed to the narrator is truncated to `_MAX_PROMPT_CHARS = 4000`
- The system prompt instructs Claude to only produce recommendation narratives
- Item titles and genre strings from the dataset are treated as untrusted input

### Dependency auditing

```bash
safety check -r requirements.txt
bandit -r src/ -ll
```

Both are run in `scripts/security_check.sh` and in `tests/security/`.

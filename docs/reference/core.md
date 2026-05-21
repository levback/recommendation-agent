# Reference: `src/core`

---

## `Message`

```python
@dataclass(frozen=True)
class Message:
    role: str       # "system" | "user" | "assistant"
    content: str
```

---

## `LLMResponse`

```python
@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    input_tokens: int  = 0
    output_tokens: int = 0
```

---

## `LLMConfig`

```python
@dataclass
class LLMConfig:
    model: str
    temperature: float = 0.7
    max_tokens: int    = 1024
```

---

## `BaseLLM`

Abstract base. Subclass and implement `complete()` and `stream()`.

```python
class BaseLLM(ABC):
    config: LLMConfig

    def __init__(self, config: LLMConfig) -> None

    @abstractmethod
    def complete(self, messages: list[Message]) -> LLMResponse: ...

    @abstractmethod
    def stream(self, messages: list[Message]) -> Iterator[str]: ...

    def chat(self, prompt: str, system: str = "") -> str:
        """Single-turn convenience wrapper. Returns response text."""
```

---

## `BedrockClient`

```python
class BedrockClient(BaseLLM):
    def __init__(
        self,
        config: LLMConfig | None = None,
        region: str = "eu-central-1",
        profile: str | None = None,
        role_arn: str | None = None,
    ) -> None
```

Default model is derived from the region via `haiku_model_id(region)`.
Uses the Bedrock **Converse API** (`converse()` and `converse_stream()`).
Message content is automatically wrapped as `[{"text": "..."}]` lists.

**Raises:** `BotoCoreError`, `ClientError` on AWS failures (caught internally,
logged, then re-raised so callers can handle them).

---

## `haiku_model_id()`

```python
def haiku_model_id(region: str) -> str
```

Returns the correct cross-region inference profile ID for the given region.
The geographic prefix (`us`, `eu`, `ap`) is derived from the region name.

```python
haiku_model_id("eu-central-1")   # "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
haiku_model_id("us-east-1")      # "us.anthropic.claude-haiku-4-5-20251001-v1:0"
haiku_model_id("ap-northeast-1") # "ap.anthropic.claude-haiku-4-5-20251001-v1:0"
```

---

## `CLAUDE_HAIKU_4_5`

```python
CLAUDE_HAIKU_4_5 = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
```

Kept for backwards compatibility. Prefer `haiku_model_id(region)` for
region-aware code.

---

## `create_bedrock_llm()`

```python
def create_bedrock_llm(
    region: str | None = None,
    profile: str | None = None,
    role_arn: str | None = None,
) -> BedrockClient
```

Factory that reads `AWS_DEFAULT_REGION` (default: `eu-central-1`), `AWS_PROFILE`,
`AWS_ROLE_ARN` from environment variables. The model ID is derived automatically
from the region via `haiku_model_id()`. Passed arguments override env vars.

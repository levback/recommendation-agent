# Reference: `src/narration`

---

## Prompt Templates (`templates.py`)

### `RECOMMENDATION_SYSTEM_PROMPT`

System prompt instructing Claude to act as a recommendation assistant and
produce natural, friendly 2–3 sentence narratives.

### `NARRATION_PROMPT_TEMPLATE`

User-turn template. Placeholders:

| Placeholder | Content |
|-------------|---------|
| `{user_preferences}` | Comma-separated list of the user's top genres |
| `{item_list}` | Numbered list of recommended items with genres and scores |

### `SINGLE_ITEM_TEMPLATE`

Template for single-item explanation via `explain_item()`.

---

## `NarrationResult`

```python
@dataclass(frozen=True)
class NarrationResult:
    user_id:   str
    narrative: str
    model:     str
    items:     list[tuple[str, float]]   # (item_id, score)
```

---

## `RecommendationNarrator`

```python
class RecommendationNarrator:
    def __init__(
        self,
        llm: BaseLLM,
        max_items_in_prompt: int = 5,
    ) -> None
```

### `narrate()`

```python
def narrate(
    self,
    user_id:         str,
    recommendations: list[tuple[str, float]],
    items_by_id:     dict[str, ItemProfile],
    user_profile:    UserProfile | None = None,
) -> NarrationResult
```

Builds a prompt from the top `max_items_in_prompt` recommendations and calls
the LLM. The prompt is truncated to `_MAX_PROMPT_CHARS = 4000` characters
before sending to prevent prompt injection via long item titles.

On LLM error, returns a `NarrationResult` with a fallback narrative
(`"Based on your interest in {genres}, we selected {n} items we think you'll enjoy."`).

### `explain_item()`

```python
def explain_item(
    self,
    item:         ItemProfile,
    user_profile: UserProfile | None = None,
) -> str
```

Returns a one-sentence explanation for a single item. Falls back to a
genre-based template on LLM error.

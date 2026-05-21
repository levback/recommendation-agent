from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass(frozen=True)
class Message:
    role: str    # "user", "assistant", "system"
    content: str


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    model: str
    temperature: float = 0.5
    max_tokens: int = 1024
    extra: dict[str, Any] = field(default_factory=dict)


class BaseLLM(ABC):
    """Abstract base for all LLM backends."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @abstractmethod
    def complete(self, messages: list[Message]) -> LLMResponse: ...

    @abstractmethod
    def stream(self, messages: list[Message]) -> Iterator[str]: ...

    def chat(self, user_message: str, system_prompt: str | None = None) -> str:
        """Convenience wrapper: single-turn chat."""
        msgs: list[Message] = []
        if system_prompt:
            msgs.append(Message(role="system", content=system_prompt))
        msgs.append(Message(role="user", content=user_message))
        return self.complete(msgs).content

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(model={self.config.model!r})"

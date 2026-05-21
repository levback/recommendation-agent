"""Unit tests for core LLM base classes."""
from __future__ import annotations

import pytest

from src.core.base import BaseLLM, LLMConfig, LLMResponse, Message
from src.core.bedrock_client import CLAUDE_HAIKU_4_5, BedrockClient
from src.core.factory import create_bedrock_llm


# ─── Minimal mock LLM ────────────────────────────────────────────────────────

class _MockLLM(BaseLLM):
    def complete(self, messages):
        return LLMResponse(content="mock", model=self.config.model)

    def stream(self, messages):
        yield "mock"


# ─── Message ─────────────────────────────────────────────────────────────────

def test_message_frozen():
    m = Message(role="user", content="hello")
    with pytest.raises(Exception):
        m.role = "assistant"  # type: ignore[misc]


def test_llm_response_defaults():
    r = LLMResponse(content="hi", model="m")
    assert r.input_tokens == 0
    assert r.output_tokens == 0
    assert r.metadata == {}


# ─── LLMConfig ───────────────────────────────────────────────────────────────

def test_llm_config_defaults():
    cfg = LLMConfig(model="test-model")
    assert cfg.temperature == 0.5
    assert cfg.max_tokens == 1024
    assert cfg.extra == {}


# ─── BaseLLM chat helper ─────────────────────────────────────────────────────

def test_chat_no_system():
    llm = _MockLLM(LLMConfig(model="test"))
    assert llm.chat("hi") == "mock"


def test_chat_with_system():
    llm = _MockLLM(LLMConfig(model="test"))
    assert llm.chat("hi", system_prompt="be helpful") == "mock"


def test_stream_yields_token():
    llm = _MockLLM(LLMConfig(model="test"))
    tokens = list(llm.stream([Message(role="user", content="hi")]))
    assert tokens == ["mock"]


# ─── BedrockClient construction ──────────────────────────────────────────────

def test_bedrock_client_default_model(monkeypatch):
    """BedrockClient uses Claude Haiku 4.5 by default."""
    import boto3

    class _FakeSession:
        def client(self, *a, **kw):
            return object()

    monkeypatch.setattr(boto3, "Session", lambda **kw: _FakeSession())
    bc = BedrockClient()
    assert bc.config.model == CLAUDE_HAIKU_4_5


def test_bedrock_client_custom_model(monkeypatch):
    import boto3

    class _FakeSession:
        def client(self, *a, **kw):
            return object()

    monkeypatch.setattr(boto3, "Session", lambda **kw: _FakeSession())
    cfg = LLMConfig(model="custom-model")
    bc = BedrockClient(config=cfg)
    assert bc.config.model == "custom-model"


# ─── Factory ─────────────────────────────────────────────────────────────────

def test_create_bedrock_llm_defaults(monkeypatch):
    import boto3

    class _FakeSession:
        def client(self, *a, **kw):
            return object()

    monkeypatch.setattr(boto3, "Session", lambda **kw: _FakeSession())
    bc = create_bedrock_llm()
    assert isinstance(bc, BedrockClient)
    assert bc.config.model == CLAUDE_HAIKU_4_5


def test_create_bedrock_llm_custom_model(monkeypatch):
    import boto3

    class _FakeSession:
        def client(self, *a, **kw):
            return object()

    monkeypatch.setattr(boto3, "Session", lambda **kw: _FakeSession())
    bc = create_bedrock_llm(model="my-model", temperature=0.2, max_tokens=512)
    assert bc.config.model == "my-model"
    assert bc.config.temperature == 0.2
    assert bc.config.max_tokens == 512


def test_create_bedrock_llm_uses_env_region(monkeypatch):
    import boto3

    class _FakeSession:
        def __init__(self, **kw):
            self.kw = kw

        def client(self, svc, region_name=None):
            self._region = region_name
            return object()

    sessions = []

    def fake_session(**kw):
        s = _FakeSession(**kw)
        sessions.append(s)
        return s

    monkeypatch.setattr(boto3, "Session", fake_session)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    bc = create_bedrock_llm()
    assert bc is not None


# ─── BedrockClient complete/stream with mock boto3 ───────────────────────────

def _make_bedrock_client(monkeypatch, response_text="hello"):
    import boto3

    class _FakeBedrockRuntime:
        def converse(self, **kw):
            return {
                "output": {"message": {"content": [{"text": response_text}]}},
                "usage": {"inputTokens": 10, "outputTokens": 5},
            }

        def converse_stream(self, **kw):
            return {
                "stream": [
                    {"contentBlockDelta": {"delta": {"text": "tok1"}}},
                    {"contentBlockDelta": {"delta": {"text": "tok2"}}},
                    {"other": {}},
                ]
            }

    class _FakeSession:
        def client(self, svc, region_name=None):
            return _FakeBedrockRuntime()

    monkeypatch.setattr(boto3, "Session", lambda **kw: _FakeSession())
    return BedrockClient()


def test_bedrock_complete(monkeypatch):
    bc = _make_bedrock_client(monkeypatch, "test response")
    r = bc.complete([Message(role="user", content="hi")])
    assert r.content == "test response"
    assert r.input_tokens == 10
    assert r.output_tokens == 5


def test_bedrock_complete_with_system(monkeypatch):
    bc = _make_bedrock_client(monkeypatch)
    msgs = [
        Message(role="system", content="be helpful"),
        Message(role="user", content="hi"),
    ]
    r = bc.complete(msgs)
    assert r.content == "hello"


def test_bedrock_complete_no_messages(monkeypatch):
    bc = _make_bedrock_client(monkeypatch)
    with pytest.raises(ValueError):
        bc.complete([])


def test_bedrock_stream(monkeypatch):
    bc = _make_bedrock_client(monkeypatch)
    tokens = list(bc.stream([Message(role="user", content="hi")]))
    assert tokens == ["tok1", "tok2"]


def test_bedrock_stream_no_messages(monkeypatch):
    bc = _make_bedrock_client(monkeypatch)
    with pytest.raises(ValueError):
        list(bc.stream([]))


def test_bedrock_complete_raises_on_error(monkeypatch):
    import boto3
    from botocore.exceptions import ClientError

    class _ErrorClient:
        def converse(self, **kw):
            raise ClientError(
                {"Error": {"Code": "AccessDeniedException", "Message": "no"}},
                "Converse",
            )

    class _FakeSession:
        def client(self, *a, **kw):
            return _ErrorClient()

    monkeypatch.setattr(boto3, "Session", lambda **kw: _FakeSession())
    bc = BedrockClient()
    with pytest.raises(ClientError):
        bc.complete([Message(role="user", content="hi")])

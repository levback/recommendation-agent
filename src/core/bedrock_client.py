from __future__ import annotations

import logging
from typing import Iterator

import boto3  # type: ignore[import]
from botocore.exceptions import BotoCoreError, ClientError  # type: ignore[import]

from .base import BaseLLM, LLMConfig, LLMResponse, Message

logger = logging.getLogger(__name__)

# Claude Haiku 4.5 base ID (without cross-region prefix)
_HAIKU_4_5_BASE = "anthropic.claude-haiku-4-5-20251001-v1:0"

# Default (kept for backwards compatibility — resolves to us. prefix)
CLAUDE_HAIKU_4_5 = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Region-prefix map for cross-region inference profiles
_REGION_PREFIX: dict[str, str] = {
    "us": "us",
    "eu": "eu",
    "ap": "ap",
}


def haiku_model_id(region: str) -> str:
    """Return the correct cross-region inference profile ID for the given region.

    Examples:
        haiku_model_id("eu-central-1")  -> "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
        haiku_model_id("us-east-1")     -> "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        haiku_model_id("ap-northeast-1") -> "ap.anthropic.claude-haiku-4-5-20251001-v1:0"
    """
    geo = region.split("-")[0]  # "eu", "us", "ap", ...
    prefix = _REGION_PREFIX.get(geo, geo)
    return f"{prefix}.{_HAIKU_4_5_BASE}"


def _split_system(messages: list[Message]) -> tuple[str, list[dict]]:
    """Separate system prompt from user/assistant turns.

    Bedrock Converse API requires content to be a list of content blocks,
    not a plain string.
    """
    system = ""
    turns: list[dict] = []
    for m in messages:
        if m.role == "system":
            system = m.content
        else:
            turns.append({"role": m.role, "content": [{"text": m.content}]})
    return system, turns


class BedrockClient(BaseLLM):
    """Amazon Bedrock client using the Converse API.

    Model defaults to Claude Haiku 4.5.  AWS credentials are resolved via the
    standard boto3 chain (env vars → named profile → IAM role) — never
    hard-coded here.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        region: str = "eu-central-1",
        profile: str | None = None,
        role_arn: str | None = None,
    ) -> None:
        if config is None:
            config = LLMConfig(model=haiku_model_id(region))
        super().__init__(config)
        self._client = self._build_client(region, profile, role_arn)

    def _build_client(self, region: str, profile: str | None, role_arn: str | None):
        session_kwargs: dict = {}
        if profile:
            session_kwargs["profile_name"] = profile
        session = boto3.Session(**session_kwargs)

        if role_arn:
            sts = session.client("sts")
            creds = sts.assume_role(
                RoleArn=role_arn,
                RoleSessionName="recommendation-agent",
            )["Credentials"]
            session = boto3.Session(
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
                region_name=region,
            )

        return session.client("bedrock-runtime", region_name=region)

    def _inference_config(self) -> dict:
        return {
            "temperature": self.config.temperature,
            "maxTokens": self.config.max_tokens,
        }

    def complete(self, messages: list[Message]) -> LLMResponse:
        system_text, turns = _split_system(messages)
        if not turns:
            raise ValueError("At least one non-system message is required")

        kwargs: dict = {
            "modelId": self.config.model,
            "messages": turns,
            "inferenceConfig": self._inference_config(),
        }
        if system_text:
            kwargs["system"] = [{"text": system_text}]

        try:
            resp = self._client.converse(**kwargs)
        except (BotoCoreError, ClientError) as exc:
            logger.error("Bedrock converse error: %s", exc)
            raise

        content = resp["output"]["message"]["content"][0]["text"]
        usage = resp.get("usage", {})
        return LLMResponse(
            content=content,
            model=self.config.model,
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
        )

    def stream(self, messages: list[Message]) -> Iterator[str]:
        system_text, turns = _split_system(messages)
        if not turns:
            raise ValueError("At least one non-system message is required")

        kwargs: dict = {
            "modelId": self.config.model,
            "messages": turns,
            "inferenceConfig": self._inference_config(),
        }
        if system_text:
            kwargs["system"] = [{"text": system_text}]

        try:
            resp = self._client.converse_stream(**kwargs)
        except (BotoCoreError, ClientError) as exc:
            logger.error("Bedrock stream error: %s", exc)
            raise

        for event in resp["stream"]:
            delta = event.get("contentBlockDelta", {}).get("delta", {})
            if "text" in delta:
                yield delta["text"]

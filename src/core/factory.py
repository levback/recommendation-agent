from __future__ import annotations

import os
from typing import Any

from .base import BaseLLM, LLMConfig
from .bedrock_client import BedrockClient, CLAUDE_HAIKU_4_5


def create_bedrock_llm(
    model: str | None = None,
    region: str | None = None,
    profile: str | None = None,
    role_arn: str | None = None,
    temperature: float = 0.5,
    max_tokens: int = 1024,
) -> BedrockClient:
    """Factory: create a BedrockClient for Claude Haiku 4.5.

    AWS credentials are resolved from the environment (standard boto3 chain).
    Never pass credentials as arguments to this function.
    """
    config = LLMConfig(
        model=model or CLAUDE_HAIKU_4_5,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    resolved_region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    return BedrockClient(
        config=config,
        region=resolved_region,
        profile=profile or os.environ.get("AWS_PROFILE"),
        role_arn=role_arn,
    )

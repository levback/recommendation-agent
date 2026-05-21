from __future__ import annotations

import os
from typing import Any

from .base import BaseLLM, LLMConfig
from .bedrock_client import BedrockClient, haiku_model_id


def create_bedrock_llm(
    model: str | None = None,
    region: str | None = None,
    profile: str | None = None,
    role_arn: str | None = None,
    temperature: float = 0.5,
    max_tokens: int = 1024,
) -> BedrockClient:
    """Factory: create a BedrockClient for Claude Haiku 4.5.

    The model ID is derived automatically from the region (us./eu./ap. prefix).
    Pass ``model`` explicitly to override.

    AWS credentials are resolved from the environment (standard boto3 chain).
    Never pass credentials as arguments to this function.
    """
    resolved_region = region or os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")
    config = LLMConfig(
        model=model or haiku_model_id(resolved_region),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return BedrockClient(
        config=config,
        region=resolved_region,
        profile=profile or os.environ.get("AWS_PROFILE"),
        role_arn=role_arn,
    )

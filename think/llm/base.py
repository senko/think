from base64 import b64encode
from enum import Enum
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field


class LLM:
    provider: str
    base_url: str | None = None
    model: str

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    @classmethod
    def for_provider(cls, provider: str) -> type["LLM"]:
        if provider == "openai":
            from .openai import OpenAIClient

            return OpenAIClient
        elif provider == "anthropic":
            from .anthropic import AnthropicClient

            return AnthropicClient

        else:
            raise ValueError(f"Unknown provider: {provider}")

    @classmethod
    def from_url(cls, url: str) -> "LLM":
        result = urlparse(url)
        return cls.for_provider(result.scheme)(
            model=result.path.lstrip("/"),
            api_key=result.username,
            base_url=result.hostname,
        )

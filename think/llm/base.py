from __future__ import annotations

from abc import ABC, abstractmethod
from logging import getLogger
from typing import AsyncGenerator, Callable, TypeVar, overload
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel

from .chat import Chat, ContentPart, ContentType, Message, Role
from .tool import ToolKit, ToolResponse

CustomParserResultT = TypeVar("CustomParserResultT")
PydanticResultT = TypeVar("PydanticResultT", bound=BaseModel)

log = getLogger(__name__)


class LLM(ABC):
    PROVIDERS = ["anthropic", "ollama", "openai"]
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

        elif provider == "ollama":
            from .ollama import OllamaClient

            return OllamaClient

        else:
            raise ValueError(f"Unknown provider: {provider}")

    @classmethod
    def from_url(cls, url: str) -> "LLM":
        result = urlparse(url)
        query = parse_qs(result.query) if result.query else {}

        base_url_scheme = "http"
        provider = result.scheme
        model = query.get("model", [result.path.lstrip("/")])[0]
        base_url = None

        if provider.endswith("+ssl"):
            base_url_scheme = "https"
            provider = provider.replace("+ssl", "")

        if result.hostname:
            base_url_port = f":{result.port}" if result.port else ""
            base_url = (
                f"{base_url_scheme}://{result.hostname}{base_url_port}{result.path}"
            )

        if base_url and (not result.query or not query.get("model")):
            raise ValueError(
                "When providing a base URL, model must be passed as a query parameter"
            )

        return cls.for_provider(result.scheme)(
            model=model,
            api_key=result.username,
            base_url=base_url,
        )

    def _get_toolkit(
        self,
        tools: ToolKit | list[Callable[[str], str]] | None,
    ) -> ToolKit | None:
        if tools is None:
            return None

        if isinstance(tools, ToolKit):
            return tools

        return ToolKit(tools)

    @overload
    @abstractmethod
    async def __call__(
        self,
        chat: Chat,
        *,
        parser: type[PydanticResultT],
        temperature: float | None = None,
        tools: ToolKit | list[Callable] | None = None,
        max_retries: int = 3,
        max_steps: int = 5,
        max_tokens: int | None = None,
    ) -> PydanticResultT: ...

    @overload
    @abstractmethod
    async def __call__(
        self,
        chat: Chat,
        *,
        parser: Callable[[str], CustomParserResultT],
        temperature: float | None = None,
        tools: ToolKit | list[Callable] | None = None,
        max_retries: int = 3,
        max_steps: int = 5,
        max_tokens: int | None = None,
    ) -> CustomParserResultT: ...

    @overload
    @abstractmethod
    async def __call__(
        self,
        chat: Chat,
        *,
        temperature: float | None = None,
        tools: ToolKit | list[Callable] | None = None,
        max_retries: int = 3,
        max_steps: int = 5,
        max_tokens: int | None = None,
    ) -> str: ...

    @abstractmethod
    async def __call__(
        self,
        chat: Chat,
        *,
        parser: type[PydanticResultT]
        | Callable[[str], CustomParserResultT]
        | None = None,
        temperature: float | None = None,
        tools: ToolKit | list[Callable] | None = None,
        max_retries: int = 3,
        max_steps: int = 5,
        max_tokens: int | None = None,
    ) -> str | PydanticResultT | CustomParserResultT:
        """Make an LLM call with the chat conversation and return a response.

        :param chat: The chat conversation to process
        :param parser: Optional parser to process the response. Can be either:
            - A Pydantic model class to parse the response into
            - A callable that takes a string and returns a custom type
        :param temperature: Optional sampling temperature (0-1)
        :param tools: Optional tools/functions available to the model:
            - A ToolKit instance
            - A list of callables that take a string and return a string
        :param max_retries: Maximum number of retries on failure
        :param max_steps: Maximum number of steps for tool use
        :param max_tokens: Optional maximum tokens in response
        :return: Either:
            - An instance of the provided Pydantic model if a model class was provided as parser
            - The result of the parser callable if a custom parser was provided
            - The raw string response if no parser was provided
        :raises ValueError: If the temperature is not between 0 and 1
        :raises APIError: If there is an error communicating with the API
        """
        ...

    async def _process_message(
        self,
        chat: Chat,
        message: Message,
        toolkit: ToolKit,
    ) -> tuple[str, list[ToolResponse]]:
        chat.messages.append(message)

        text = ""
        response_list = []
        for part in message.content:
            if part.type == ContentType.text:
                text += part.text
            elif part.type == ContentType.tool_call:
                if toolkit is None:
                    log.warning("Tool call with no toolkit defined, ignoring")
                    continue
                tool_response = await toolkit.execute_tool_call(part.tool_call)
                response_list.append(tool_response)

        if response_list:
            chat.messages.append(
                Message(
                    role=Role.tool,
                    content=[
                        ContentPart(
                            type=ContentType.tool_response,
                            tool_response=tool_response,
                        )
                        for tool_response in response_list
                    ],
                )
            )

        return text, response_list

    @abstractmethod
    async def stream(
        self,
        chat: Chat,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream responses from the LLM model.

        :param chat: The chat conversation to process
        :param temperature: Optional sampling temperature (0-1)
        :param max_tokens: Optional maximum tokens in response
        :return: An async generator of response strings
        """
        ...

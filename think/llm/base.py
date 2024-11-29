from __future__ import annotations

from abc import ABC, abstractmethod
from json import JSONDecodeError, loads
from logging import getLogger
from time import time
from typing import AsyncGenerator, Callable, TypeVar, overload
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, ValidationError

from think.parser import JSONParser

from .chat import Chat, ContentPart, ContentType, Message, Role
from .tool import ToolDefinition, ToolKit, ToolResponse

CustomParserResultT = TypeVar("CustomParserResultT")
PydanticResultT = TypeVar("PydanticResultT", bound=BaseModel)

log = getLogger(__name__)


class BaseAdapter(ABC):
    toolkit: ToolKit

    def __init__(self, toolkit: ToolKit | None = None):
        self.toolkit = toolkit

    @abstractmethod
    def get_tool_spec(self, tool: ToolDefinition) -> dict: ...

    @property
    def spec(self) -> dict | None:
        if self.toolkit is None:
            return None
        return self.toolkit.generate_tool_spec(self.get_tool_spec)


class LLM(ABC):
    PROVIDERS = ["anthropic", "ollama", "openai"]

    provider: str
    adapter_class: type[BaseAdapter]
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

        elif provider == "google":
            from .google import GoogleClient

            return GoogleClient

        elif provider == "groq":
            from .groq import GroqClient

            return GroqClient
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
        toolkit = self._get_toolkit(tools)
        adapter = self.adapter_class(toolkit)

        if isinstance(parser, type) and issubclass(parser, BaseModel):
            response_format = parser
        else:
            response_format = None

        tools_dbg = f" and tools {', '.join(toolkit.tool_names)}" if toolkit else ""
        format_dbg = (
            f" expecting {response_format.__name__} response" if response_format else ""
        )
        log.debug(
            f"Making a {self.model} call with {len(chat)} messages{tools_dbg}{format_dbg}"
        )

        # FIXME: error handling!
        t0 = time()
        message = await self._internal_call(
            chat,
            temperature,
            max_tokens,
            adapter,
            response_format=response_format,
        )
        t1 = time()

        log.debug(f"Received response in {(t1 - t0):.1f}s")

        text, response_list = await self._process_message(chat, message, toolkit)

        if response_list:
            if max_steps < 1:
                log.warning("Tool call steps limit reached, stopping")
            else:
                return await self(
                    chat,
                    temperature=temperature,
                    tools=tools,
                    parser=parser,
                    max_steps=max_steps - 1,
                    max_retries=max_retries,
                )

        if message.parsed:
            return message.parsed

        elif parser:
            try:
                if isinstance(parser, type) and issubclass(parser, BaseModel):
                    message.parsed = JSONParser(spec=parser)(text)
                else:
                    message.parsed = parser(text)
                return message.parsed

            except (JSONDecodeError, ValidationError, ValueError) as err:
                log.debug(f"Error parsing response '{text}': {err}")

                if not max_retries:
                    raise

                error_text = f"Error parsing your response: {err}. Please output your response EXACTLY as requested."
                chat.user(error_text)
                return await self(
                    chat,
                    temperature=temperature,
                    tools=tools,
                    parser=parser,
                    max_steps=max_steps,
                    max_retries=max_retries - 1,
                )

        return text

    @abstractmethod
    async def _internal_call(
        self,
        chat: Chat,
        temperature: float | None,
        max_tokens: int | None,
        adapter: BaseAdapter,
        response_format: PydanticResultT | None = None,
    ) -> Message: ...

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
    async def _internal_stream(
        self,
        chat: Chat,
        adapter: BaseAdapter,
        temperature: float | None,
        max_tokens: int | None,
    ) -> AsyncGenerator[str, None]: ...

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
        :return: An async generator of response string chunks
        """
        ...
        adapter = self.adapter_class()

        log.debug(f"Making a {self.model} streaming request with {len(chat)} messages")
        text = ""
        async for chunk in self._internal_stream(
            chat,
            adapter,
            temperature,
            max_tokens,
        ):
            text += chunk
            yield chunk

        if text:
            chat.messages.append(
                Message(
                    role=Role.assistant,
                    content=[ContentPart(type=ContentType.text, text=text)],
                )
            )


__all__ = [
    "BaseAdapter",
    "LLM",
    "CustomParserResultT",
    "PydanticResultT",
]

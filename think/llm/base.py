"""
# Core LLM Functionality

The `llm.base` module provides the core functionality for interacting with large language models (LLMs).
It defines the `LLM` class, which is the main entry point for sending requests to LLMs and processing
their responses.

## Basic Usage

```python
# example: basic_llm.py
from think import LLM

# Initialize an LLM using a URL-based configuration
llm = LLM.from_url("openai:///gpt-5-nano")

# Create a simple chat
from think.llm.chat import Chat
chat = Chat("What is the capital of France?")

# Get a response
import asyncio
response = asyncio.run(llm(chat))
print(response)
```

## Model URL Format

Think uses a URL-like format to specify the model to use:

```
provider://[api_key@][host[:port]]/model[?query]
```

- `provider` is the model provider (openai, anthropic, google, etc.)
- `api-key` is the API key (optional if set via environment)
- `host[:port]` is the server to use (optional, for local LLMs)
- `model` is the name of the model to use

Examples:
- `openai:///gpt-5-nano` (API key from OPENAI_API_KEY environment variable)
- `anthropic://sk-my-key@/claude-3-opus-20240229` (explicit API key)
- `openai://localhost:8080/wizard-mega` (custom server over HTTP)
- `openai:///gpt-4o?service_tier=flex` (extra parameters passed to the API)

## Streaming

For generating responses incrementally:

```python
# example: streaming.py
import asyncio
from think import LLM
from think.llm.chat import Chat

llm = LLM.from_url("anthropic:///claude-3-haiku-20240307")

async def stream_response():
    chat = Chat("Generate a short poem about programming")
    async for chunk in llm.stream(chat):
        print(chunk, end="", flush=True)
    print()

asyncio.run(stream_response())
```

## Error Handling

The LLM class throws specific exceptions for different error cases:
- `ConfigError`: Configuration errors (invalid URL, missing API key)
- `BadRequestError`: Invalid requests (e.g., inappropriate content)
- Other standard exceptions like `ConnectionError`, `TimeoutError`

See [Supported Providers](#supported-providers) for provider-specific information.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from json import JSONDecodeError
from logging import getLogger
from time import time
from typing import (
    TYPE_CHECKING,
    AsyncGenerator,
    Awaitable,
    Callable,
    TypeVar,
    cast,
    overload,
)
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, ValidationError

from think.parser import JSONParser

from .chat import Chat, ContentPart, ContentType, Message, Role
from .tool import ToolCall, ToolDefinition, ToolKit, ToolResponse

CustomParserResultT = TypeVar("CustomParserResultT")
PydanticResultT = TypeVar("PydanticResultT", bound=BaseModel)
RetryResultT = TypeVar("RetryResultT")

log = getLogger(__name__)

if TYPE_CHECKING:
    import httpx


class BaseAdapter(ABC):
    """
    Abstract base class for the LLM API adapters

    Adapters are responsible for converting the LLM API calls into the
    format expected by the underlying API. They also handle the conversion
    of the API responses into the format expected by the LLM.
    """

    toolkit: ToolKit | None

    def __init__(self, toolkit: ToolKit | None = None):
        """
        Initialize the adapter.

        :param toolkit: Optional toolkit to provide tool functions

        Toolkit, if provided, is made available to the underlying LLM for tool
        (function) use.
        """
        self.toolkit = toolkit

    @abstractmethod
    def get_tool_spec(self, tool: ToolDefinition) -> dict:
        """
        Get the provider-specific tool specification for a tool definition.

        :param tool: The tool definition
        :return: The provider-specific tool specification
        """
        pass

    @property
    def spec(self) -> list[dict] | None:
        """
        Generate the provider-specific tool specification for all the
        tools passed to the LLM.

        Note that some LLM APIs require a sentinel value (NOT_GIVEN) instead
        of None if no tools are defined. This shouold be handled by the
        provider-specific LLM client.

        :return: The provider-specific tool specification or None if there
            are no tools defined.
        """
        if self.toolkit is None:
            return None
        return self.toolkit.generate_tool_spec(self.get_tool_spec)


class ConfigError(ValueError):
    """
    Configuration error

    Encompasses non-recoverable errors due to incorrect configuration
    values, such as:

    * incorrect or missing API keys
    * incorrect base URL (if provided)
    * unrecognized model
    * invalid parameters
    """

    message: str

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class BadRequestError(ValueError):
    """
    Bad request error

    Encompasses non-recoverable errors due to incorrect request
    values, such as:

    * invalid chat messages
    * invalid tool calls
    * invalid parameters
    """

    message: str

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class LLM(ABC):
    """
    LLM client

    This is a base class for the LLM clients. It provides the common
    functionality for making LLM API calls and processing the responses.
    The provider-specific LLM clients inherit from this class to implement
    the provider-specific API calls and response processing.

    Example usage:

    >>> client = LLM.from_url("openai:///gpt-3.5-turbo")
    >>> client(...)
    """

    PROVIDERS = ["anthropic", "google", "groq", "litellm", "ollama", "openai"]

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
        **kwargs,
    ):
        """
        Initialize the LLM client.

        This must be called on the provider-specific LLM class:

        >>> client_class = LLM.for_provider("openai")
        >>> client = client_class("gpt-3.5-turbo", api_key="secret")

        In most cases, you should use the `from_url` class method instead.

        :param model: The model to use
        :param api_key: Optional API key (if required by the provider and
            not available in the environment variables)
        :param **kwargs: Optional extra parameters for the provider API
        :param base_url: Optional base URL for the provider API
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.extra_params = kwargs

    @classmethod
    def for_provider(cls, provider: str) -> type["LLM"]:
        """
        Get the LLM client class for the specified provider.

        :param provider: The provider name
        :return: The LLM client class for the provider

        Raises a ValueError if the provider is not supported.
        The list of supported providers is available in the
        PROVIDERS class attribute.
        """
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

        elif provider == "bedrock":
            from .bedrock import BedrockClient

            return BedrockClient
        elif provider == "litellm":
            from .litellm import LiteLLMClient

            return LiteLLMClient
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @classmethod
    def from_url(cls, url: str) -> "LLM":
        """
        Initialize an LLM client from a URL.

        :param url: The URL to initialize the client from
        :return: The LLM client instance

        The URL format is: `provider://[api_key@][host[:port]]/model[?query]`

        Examples:
            - `openai:///gpt-3.5-turbo` (API key in environment)
            - `openai://sk-my-openai-key@/gpt-3-5-turbo` (explicit API key)
            - `openai://localhost:1234/v1?model=llama-3.2-8b` (custom server over HTTP)
            - `openai+https://openrouter.ai/api/v1?model=llama-3.2-8b` (custom server, HTTPS)
            - `bedrock:///amazon.nova-lite-v1:0?region=us-east-1 (AWS region as an extra param)

        Note that if the base URL is provided, the model must be passed
        as a query parameter.

        Query parameters (other than ``model``) are passed through as extra
        keyword arguments to the underlying provider API calls. For example,
        ``openai:///gpt-4o?service_tier=flex`` passes ``service_tier="flex"``
        to OpenAI's ``chat.completions.create()``.

        Note: query parameter values are always strings. For parameters that
        require numeric types, use ``LLM.for_provider()`` directly.
        """
        result = urlparse(url)
        query = parse_qs(result.query) if result.query else {}

        base_url_scheme = "http"
        provider = result.scheme
        model = query.pop("model", [result.path.lstrip("/")])[0]
        base_url = None

        if provider.endswith("+https"):
            base_url_scheme = "https"
            provider = provider.replace("+https", "")

        if result.hostname:
            base_url_port = f":{result.port}" if result.port else ""
            base_url = (
                f"{base_url_scheme}://{result.hostname}{base_url_port}{result.path}"
            )

        if base_url and not result.query:
            raise ValueError(
                "When providing a base URL, model must be passed as a query parameter"
            )

        extra_params = {k: v[0] for k, v in query.items()}
        if result.username and result.password:
            api_key = f"{result.username}:{result.password}"
        elif result.username:
            api_key = result.username
        else:
            api_key = None
        return cls.for_provider(result.scheme)(
            model=model,
            api_key=api_key,
            base_url=base_url,
            **extra_params,
        )

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
        :param max_retries: Maximum number of retries on failure. Governs both
            parser-error retries (re-prompting the model when JSON/Pydantic
            parsing fails) and SDK transport retries on transient errors
            (5xx responses, connection failures). For providers whose SDK
            exposes a retry knob, this value is forwarded; for the rest it
            drives the in-process retry wrapper (`LLM._retry`).
        :param max_steps: Maximum number of steps for tool use
        :param max_tokens: Optional maximum tokens in response
        :return: Either:
            - An instance of the provided Pydantic model if a model class was provided as parser
            - The result of the parser callable if a custom parser was provided
            - The raw string response if no parser was provided
        :raises ValueError: If the temperature is not between 0 and 1
        :raises APIError: If there is an error communicating with the API
        """
        if tools is None:
            toolkit = None
        elif isinstance(tools, ToolKit):
            toolkit = tools
        else:
            toolkit = ToolKit(tools)

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

        t0 = time()
        try:
            message = await self._internal_call(
                chat,
                temperature,
                max_tokens,
                adapter,
                max_retries=max_retries,
                response_format=response_format,
            )
        except (ConfigError, BadRequestError) as err:
            log.error(
                f"Error calling {self.provider} API: {err.message}",
                exc_info=True,
            )
            raise

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
                return message.parsed  # type: ignore[return-value]

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
        *,
        max_retries: int,
        response_format: PydanticResultT | None = None,
    ) -> Message:
        """
        Make the LLM API call - internal implementation.

        Each provider-specific LLM client must implement this
        method to make the API call to the provider's API.

        :param chat: The chat conversation to process
        :param temperature: Optional sampling temperature (0-1)
        :param max_tokens: Optional maximum tokens in response
        :param adapter: The adapter to convert the chat to the provider format
        :param max_retries: Number of retries to allow on transient errors;
            forwarded to the underlying SDK or used by `LLM._retry`
        :param response_format: Optional Pydantic model to parse the response into
            (to be used only if the provider supports structured responses)
        :return: The response message from the provider
        """
        pass

    async def _process_message(
        self,
        chat: Chat,
        message: Message,
        toolkit: ToolKit | None,
    ) -> tuple[str, list[ToolResponse]]:
        """
        Process the assistant response message - internal implementation.

        This methods appends the response message to the end of the
        chat and checks for and runs any tool calls requested by the
        AI assistant.

        If the list of tool responses is non-empty, the LLM client should
        resend the chat to the API with the tool responses appended.

        :param chat: The chat conversation to process
        :param message: The response message from the AI assistant
        :param toolkit: The toolkit to execute tool calls
        :return: A tuple of the response text and a list of tool responses
        """
        chat.messages.append(message)

        text = ""
        response_list = []
        for part in message.content:
            if part.type == ContentType.text:
                text += cast(str, part.text)
            elif part.type == ContentType.tool_call:
                if toolkit is None:
                    log.warning("Tool call with no toolkit defined, ignoring")
                    continue
                tool_response = await toolkit.execute_tool_call(
                    cast(ToolCall, part.tool_call)
                )
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
        *,
        max_retries: int,
    ) -> AsyncGenerator[str, None]:
        """
        Make a streaming LLM API call - internal implementation.

        Each provider-specific LLM client must implement this method to make
        the API call to the provider's API and stream the response.

        :param chat: The chat conversation to process
        :param adapter: The adapter to convert the chat to the provider format
        :param temperature: Optional sampling temperature (0-1)
        :param max_tokens: Optional maximum tokens in response
        :param max_retries: Number of retries to allow on transient errors when
            establishing the stream; only the connection establishment is
            retried, not chunk consumption.
        :return: An async generator of response string chunks
        """
        pass

    async def stream(
        self,
        chat: Chat,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 3,
    ) -> AsyncGenerator[str, None]:
        """Stream responses from the LLM model.

        :param chat: The chat conversation to process
        :param temperature: Optional sampling temperature (0-1)
        :param max_tokens: Optional maximum tokens in response
        :param max_retries: Number of retries to allow on transient errors when
            establishing the stream
        :return: An async generator of response string chunks
        """
        ...
        adapter = self.adapter_class()

        log.debug(f"Making a {self.model} streaming request with {len(chat)} messages")
        text = ""

        try:
            # Type-checker workaround: object async_generator can't be used in 'await' expression
            async for chunk in self._internal_stream(
                chat,
                adapter,
                temperature,
                max_tokens,
                max_retries=max_retries,
            ):  # type:ignore
                text += chunk
                yield chunk
        except ConfigError as err:
            log.error(
                f"Error calling {self.provider} API: {err.message}",
                exc_info=True,
            )
            raise

        if text:
            chat.messages.append(
                Message(
                    role=Role.assistant,
                    content=[ContentPart(type=ContentType.text, text=text)],
                )
            )

    @staticmethod
    async def _retry(
        fn: Callable[[], Awaitable[RetryResultT]],
        *,
        max_retries: int,
        backoff: float = 1.0,
        max_backoff: float = 30.0,
    ) -> RetryResultT:
        """
        Run an async LLM call with exponential-backoff retries on transient errors.

        Used by providers whose SDK does not expose a native retry knob.
        Mirrors OpenAI/Anthropic SDK semantics: ``max_retries=N`` means up to
        ``N`` retries on top of the initial attempt (``N+1`` total attempts).
        ``ConfigError`` and ``BadRequestError`` are treated as non-transient
        and re-raised immediately.

        :param fn: Zero-arg async callable that performs the request.
        :param max_retries: Maximum number of retries (excluding the initial
            attempt). ``0`` disables retries.
        :param backoff: Base delay (seconds) for exponential backoff.
        :param max_backoff: Cap on the delay between retries (seconds).
        """
        for attempt in range(max_retries + 1):
            try:
                return await fn()
            except (ConfigError, BadRequestError):
                raise
            except Exception as err:
                if attempt >= max_retries:
                    raise
                delay = min(backoff * (2**attempt), max_backoff)
                log.debug(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    max_retries + 1,
                    err,
                    delay,
                )
                await asyncio.sleep(delay)
        raise RuntimeError("unreachable")

    @staticmethod
    def _error_from_json_response(response: "httpx.Response") -> str:
        """Get the error message from a JSON response - internal method."""
        try:
            return response.json()["error"]["message"]
        except (JSONDecodeError, KeyError):
            return response.text


__all__ = [
    "BaseAdapter",
    "LLM",
    "CustomParserResultT",
    "PydanticResultT",
]

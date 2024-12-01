from __future__ import annotations

from abc import ABC, abstractmethod
from json import JSONDecodeError
from logging import getLogger
from time import time
from typing import TYPE_CHECKING, AsyncGenerator, Callable, TypeVar, overload
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, ValidationError

from think.parser import JSONParser

from .chat import Chat, ContentPart, ContentType, Message, Role
from .tool import ToolDefinition, ToolKit, ToolResponse

CustomParserResultT = TypeVar("CustomParserResultT")
PydanticResultT = TypeVar("PydanticResultT", bound=BaseModel)

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

    toolkit: ToolKit

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
    def spec(self) -> dict | None:
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

    PROVIDERS = ["anthropic", "google", "groq", "ollama", "openai"]

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
        """
        Initialize the LLM client.

        This must be called on the provider-specific LLM class:

        >>> client_class = LLM.for_provider("openai")
        >>> client = client_class("gpt-3.5-turbo", api_key="secret")

        In most cases, you should use the `from_url` class method instead.

        :param model: The model to use
        :param api_key: Optional API key (if required by the provider and
            not available in the environment variables)
        :param base_url: Optional base URL for the provider API
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

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
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @classmethod
    def from_url(cls, url: str) -> "LLM":
        """
        Initialize an LLM client from a URL.

        Arguments:
            - `url`: The URL to initialize the client from

        Returns the LLM client instance.

        :param url: The URL to initialize the client from
        :return: The LLM client instance

        The URL format is: `provider://[api_key@][host[:port]]/model[?query]`

        Examples:
            - `openai:///gpt-3.5-turbo` (API key in environment)
            - `openai://sk-my-openai-key@/gpt-3-5-turbo` (explicit API key)
            - `openai://localhost:1234/v1?model=llama-3.2-8b` (custom server over HTTP)
            - `openai+https://openrouter.ai/api/v1?model=llama-3.2-8b` (custom server, HTTPS)

        Note that if the base URL is provided, the model must be passed
        as a query parameter.
        """
        result = urlparse(url)
        query = parse_qs(result.query) if result.query else {}

        base_url_scheme = "http"
        provider = result.scheme
        model = query.get("model", [result.path.lstrip("/")])[0]
        base_url = None

        if provider.endswith("+https"):
            base_url_scheme = "https"
            provider = provider.replace("+https", "")

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
    ) -> Message:
        """
        Make the LLM API call - internal implementation.

        Each provider-specific LLM client must implement this
        method to make the API call to the provider's API.

        :param chat: The chat conversation to process
        :param temperature: Optional sampling temperature (0-1)
        :param max_tokens: Optional maximum tokens in response
        :param adapter: The adapter to convert the chat to the provider format
        :param response_format: Optional Pydantic model to parse the response into
            (to be used only if the provider supports structured responses)
        :return: The response message from the provider
        """
        pass

    async def _process_message(
        self,
        chat: Chat,
        message: Message,
        toolkit: ToolKit,
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
    ) -> AsyncGenerator[str, None]:
        """
        Make a streaming LLM API call - internal implementation.

        Each provider-specific LLM client must implement this method to make
        the API call to the provider's API and stream the response.

        :param chat: The chat conversation to process
        :param adapter: The adapter to convert the chat to the provider format
        :param temperature: Optional sampling temperature (0-1)
        :param max_tokens: Optional maximum tokens in response
        :return: An async generator of response string chunks
        """
        pass

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

        try:
            async for chunk in self._internal_stream(
                chat,
                adapter,
                temperature,
                max_tokens,
            ):
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

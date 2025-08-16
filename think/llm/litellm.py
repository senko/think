from __future__ import annotations

import json
from logging import getLogger
from typing import Any, AsyncGenerator

try:
    from litellm import acompletion
except ImportError as err:
    raise ImportError(
        "LiteLLM client requires the litellm Python SDK: pip install litellm"
    ) from err

from .base import LLM, BadRequestError, BaseAdapter, ConfigError, PydanticResultT
from .chat import Chat, ContentPart, ContentType, Message, Role, image_url, document_url
from .tool import ToolCall, ToolDefinition, ToolResponse

log = getLogger(__name__)


class LiteLLMAdapter(BaseAdapter):
    """
    Adapter for LiteLLM that converts think's internal formats to OpenAI-compatible format.

    LiteLLM uses OpenAI's API format as the standard, so we follow the same patterns
    as the OpenAI adapter but route through litellm for multi-provider support.
    """

    def get_tool_spec(self, tool: ToolDefinition) -> dict:
        """Convert think tool definition to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.schema,
            },
        }

    def dump_message(self, message: Message) -> list[dict]:
        """Convert think Message to OpenAI message format."""
        tool_calls = []
        tool_responses = {}
        text_parts = []
        image_parts = []
        doc_parts = []

        for part in message.content:
            match part:
                case ContentPart(type=ContentType.tool_call, tool_call=tool_call):
                    tool_calls.append(
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": json.dumps(tool_call.arguments),
                            },
                        }
                    )
                case ContentPart(
                    type=ContentType.tool_response,
                    tool_response=ToolResponse(
                        call=call,
                        response=response,
                        error=error,
                    ),
                ):
                    tool_responses[call.id] = (
                        response if response is not None else (error or "no response")
                    )
                case ContentPart(type=ContentType.text, text=text):
                    text_parts.append(
                        {
                            "type": "text",
                            "text": text,
                        }
                    )
                case ContentPart(type=ContentType.image, image=image):
                    image_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": image},
                        }
                    )
                case ContentPart(type=ContentType.document):
                    if part.is_document_url:
                        raise ValueError(
                            "LiteLLM adapter does not support document URLs"
                        )

                    mime_type = part.document_mime_type
                    if mime_type != "application/pdf":
                        raise ValueError(f"Unsupported document MIME type: {mime_type}")

                    doc_parts.append(
                        {
                            "type": "input_file",
                            "file_name": "document.pdf",
                            "file_data": part.document_data,
                        }
                    )

        # Handle tool responses
        if tool_responses:
            return [
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": response,
                }
                for call_id, response in tool_responses.items()
            ]

        # Handle assistant messages with tool calls
        if tool_calls or message.role == Role.assistant:
            if len(text_parts) == 1:
                text_parts = text_parts[0]["text"]
            return [
                {
                    "role": "assistant",
                    "content": text_parts or None,
                    "tool_calls": tool_calls or None,
                }
            ]

        # Handle system messages
        if message.role == Role.system:
            if len(text_parts) == 1:
                text_parts = text_parts[0]["text"]
            return [{"role": "system", "content": text_parts}]

        # Handle user messages
        if message.role == Role.user:
            content = text_parts + image_parts + doc_parts
            if len(content) == 1 and content[0]["type"] == "text":
                content = content[0]["text"]
            return [
                {
                    "role": "user",
                    "content": content,
                }
            ]

        raise ValueError(f"Unsupported message role: {message.role}")

    @staticmethod
    def text_content(
        content: str | list[dict[str, str]],
    ) -> str | None:
        """Extract text content from OpenAI message content."""
        if content is None:
            return None

        if isinstance(content, str):
            return content
        else:
            return "".join(part.get("text", "") for part in content)

    def parse_tool_call(self, message: dict[str, Any]) -> Message:
        """Parse OpenAI tool call message to think Message."""
        tool_call_id = message.get("tool_call_id")
        if tool_call_id is None:
            raise ValueError("Missing tool_call_id in tool message: %r", message)

        text = self.text_content(message.get("content"))

        return Message(
            role=Role.tool,
            content=[
                ContentPart(
                    type=ContentType.tool_response,
                    tool_response=ToolResponse(
                        call=ToolCall(
                            id=tool_call_id,
                            name="",
                            arguments={},
                        ),
                        response=text,
                    ),
                )
            ],
        )

    def parse_assistant_message(self, message: dict[str, Any]) -> Message:
        """Parse OpenAI assistant message to think Message."""
        raw_tool_calls = message.get("tool_calls")
        tool_calls = []
        if raw_tool_calls:
            for tc in raw_tool_calls:
                call_id = tc.get("id")
                if call_id is None:
                    raise ValueError(
                        "Missing tool call ID in assistant message: %r", tc
                    )
                name = tc.get("function", {}).get("name")
                if name is None:
                    raise ValueError(
                        "Missing function name in assistant message: %r", tc
                    )
                arguments = tc.get("function", {}).get("arguments")
                if arguments is None:
                    raise ValueError(
                        "Missing function arguments in assistant message: %r", tc
                    )

                tool_calls.append(
                    ContentPart(
                        type=ContentType.tool_call,
                        tool_call=ToolCall(
                            id=call_id,
                            name=name,
                            arguments=json.loads(arguments),
                        ),
                    )
                )

        text = self.text_content(message.get("content"))
        parts = []
        if text:
            parts.append(
                ContentPart(
                    type=ContentType.text,
                    text=text or "",
                )
            )

        return Message(
            role=Role.assistant,
            content=parts + tool_calls,
        )

    def parse_message(self, message: dict[str, Any]) -> Message:
        """Parse OpenAI message format to think Message."""
        role = message.get("role", "user")
        if role == "tool":
            return self.parse_tool_call(message)

        elif role == "assistant":
            return self.parse_assistant_message(message)

        elif role == "system":
            text = self.text_content(message.get("content"))
            return Message.system(text)

        elif role == "user":
            raw_content = message.get("content")
            if raw_content is None:
                raise ValueError("Missing content in user message: %r", message)

            content: list[ContentPart] = []
            if isinstance(raw_content, str):
                content.append(
                    ContentPart(
                        type=ContentType.text,
                        text=raw_content,
                    )
                )
            else:
                for part in raw_content:
                    part_type = part.get("type")
                    if part_type == "text":
                        content.append(
                            ContentPart(
                                type=ContentType.text,
                                text=part.get("text"),
                            )
                        )
                    elif part_type == "image_url":
                        content.append(
                            ContentPart(
                                type=ContentType.image,
                                image=image_url(part.get("image_url", {}).get("url")),
                            )
                        )
                    elif part_type == "input_file":
                        content.append(
                            ContentPart(
                                type=ContentType.document,
                                document=document_url(part.get("file_data")),
                            )
                        )
                    else:
                        raise ValueError(f"Unsupported content part type: {part_type}")

            return Message(role=Role.user, content=content)

        raise ValueError(f"Unsupported message type: {type(message)}")

    def dump_chat(self, chat: Chat) -> tuple[str, list[dict]]:
        """Convert think Chat to OpenAI messages format."""
        messages = []
        for m in chat.messages:
            messages.extend(self.dump_message(m))
        return "", messages

    def load_chat(self, messages: list[dict]) -> Chat:
        """Convert OpenAI messages to think Chat."""
        c = Chat()
        for m in messages:
            c.messages.append(self.parse_message(m))
        return c


class LiteLLMClient(LLM):
    """
    LiteLLM client that provides access to 100+ LLM providers through a unified interface.

    LiteLLM acts as a universal adapter for various AI providers, normalizing their APIs
    into a consistent OpenAI-compatible interface.
    """

    provider = "litellm"
    adapter_class = LiteLLMAdapter

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        super().__init__(model, api_key=api_key, base_url=base_url, **kwargs)

        # Set up litellm configuration
        if api_key:
            # LiteLLM can handle provider-specific API keys through environment
            # or direct configuration - we'll let it handle the specifics
            pass

        if base_url:
            # Some providers support custom base URLs
            pass

    async def _internal_call(
        self,
        chat: Chat,
        temperature: float | None,
        max_tokens: int | None,
        adapter: LiteLLMAdapter,
        response_format: PydanticResultT | None = None,
    ) -> Message:
        """Make an async LLM call using litellm."""
        _, messages = adapter.dump_chat(chat)

        # Prepare call parameters
        call_params = {
            "model": self.model,
            "messages": messages,
        }

        if temperature is not None:
            call_params["temperature"] = temperature

        if max_tokens is not None:
            call_params["max_tokens"] = max_tokens

        if adapter.spec:
            call_params["tools"] = adapter.spec

        # Add any extra parameters from initialization
        call_params.update(self.extra_params)

        try:
            # Use litellm's async completion function
            response = await acompletion(**call_params)

        except Exception as err:
            # Map litellm errors to think's error types
            error_msg = str(err)

            if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                raise ConfigError(f"Authentication error: {error_msg}") from err
            elif "not found" in error_msg.lower() or "model" in error_msg.lower():
                raise ConfigError(f"Model not found: {error_msg}") from err
            elif (
                "bad request" in error_msg.lower()
                or "invalid" in error_msg.lower()
                or "has no attribute 'get'" in error_msg.lower()
                or "openaiexception" in error_msg.lower()
            ):
                raise BadRequestError(f"Bad request: {error_msg}") from err
            else:
                raise ConfigError(f"LiteLLM error: {error_msg}") from err

        # Convert response to think Message format
        choice = response.choices[0]
        message_dict = {
            "role": choice.message.role,
            "content": choice.message.content,
        }

        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            message_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]

        message = adapter.parse_message(message_dict)

        # Handle structured output if response_format was specified
        if response_format and hasattr(choice.message, "parsed"):
            message.parsed = choice.message.parsed

        return message

    async def _internal_stream(
        self,
        chat: Chat,
        adapter: LiteLLMAdapter,
        temperature: float | None,
        max_tokens: int | None,
    ) -> AsyncGenerator[str, None]:
        """Make a streaming LLM call using litellm."""
        _, messages = adapter.dump_chat(chat)

        # Prepare call parameters
        call_params = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        if temperature is not None:
            call_params["temperature"] = temperature

        if max_tokens is not None:
            call_params["max_tokens"] = max_tokens

        # Add any extra parameters from initialization
        call_params.update(self.extra_params)

        try:
            # Use litellm's async streaming completion
            response = await acompletion(**call_params)

            async for chunk in response:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                if hasattr(delta, "content") and delta.content is not None:
                    yield delta.content
                elif choice.finish_reason == "stop":
                    pass  # ignore
                else:
                    log.debug(
                        "LiteLLMClient.stream(): ignoring unknown chunk %r", chunk
                    )

        except Exception as err:
            # Map litellm errors to think's error types
            error_msg = str(err)

            if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                raise ConfigError(f"Authentication error: {error_msg}") from err
            elif "not found" in error_msg.lower() or "model" in error_msg.lower():
                raise ConfigError(f"Model not found: {error_msg}") from err
            elif (
                "bad request" in error_msg.lower()
                or "invalid" in error_msg.lower()
                or "has no attribute 'get'" in error_msg.lower()
                or "openaiexception" in error_msg.lower()
            ):
                raise BadRequestError(f"Bad request: {error_msg}") from err
            else:
                raise ConfigError(f"LiteLLM error: {error_msg}") from err

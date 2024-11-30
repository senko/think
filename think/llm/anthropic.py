from __future__ import annotations

from base64 import b64decode
from logging import getLogger
from typing import AsyncGenerator, Literal

try:
    from anthropic import (
        NOT_GIVEN,
        AsyncAnthropic,
        AsyncStream,
        AuthenticationError,
        NotFoundError,
        BadRequestError as AnthropicBadRequestError,
    )
    from anthropic.types import Message as AnthropicMessage
    from anthropic.types import RawMessageStreamEvent
    from anthropic.types.image_block_param import Source
except ImportError as err:
    raise ImportError(
        "Anthropic client requires the Anthropic Python SDK: pip install anthropic"
    ) from err


from .base import LLM, BaseAdapter, ConfigError, BadRequestError, PydanticResultT
from .chat import Chat, ContentPart, ContentType, Message, Role
from .tool import ToolCall, ToolDefinition, ToolResponse

log = getLogger(__name__)


class AnthropicAdapter(BaseAdapter):
    def get_tool_spec(self, tool: ToolDefinition) -> dict:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.schema,
        }

    def dump_role(self, role: Role) -> Literal["user", "assistant"]:
        if role in [Role.system, Role.user, Role.tool]:
            return "user"
        else:
            return "assistant"

    def dump_content_part(self, part: ContentPart) -> dict:
        match part:
            case ContentPart(type=ContentType.text, text=text):
                return dict(
                    type="text",
                    text=text,
                )
            case ContentPart(type=ContentType.image):
                return dict(
                    type="image",
                    source=Source(
                        type="base64",
                        data=part.image_data,
                        media_type=part.image_mime_type,
                    ),
                )
            case ContentPart(
                type=ContentType.tool_call,
                tool_call=ToolCall(id=id, name=name, arguments=arguments),
            ):
                return dict(
                    type="tool_use",
                    id=id,
                    name=name,
                    input=arguments,
                )
            case ContentPart(
                type=ContentType.tool_response,
                tool_response=ToolResponse(
                    call=ToolCall(id=id),
                    response=response,
                    error=error,
                ),
            ):
                return dict(
                    type="tool_result",
                    tool_use_id=id,
                    content=response if response is not None else (error or ""),
                )
            case _:
                raise ValueError(f"Unknown content type: {part.type}")

    def parse_content_part(self, part: dict) -> ContentPart:
        match part:
            case {"type": "text", "text": text}:
                return ContentPart(type=ContentType.text, text=text)
            case {"type": "image", "source": {"data": data}}:
                return ContentPart(
                    type=ContentType.image,
                    image=b64decode(data.encode("ascii")),
                )
            case {"type": "tool_use", "id": id, "name": name, "input": input}:
                return ContentPart(
                    type=ContentType.tool_call,
                    tool_call=ToolCall(id=id, name=name, arguments=input),
                )
            case {"type": "tool_result", "tool_use_id": id, "content": content}:
                return ContentPart(
                    type=ContentType.tool_response,
                    tool_response=ToolResponse(
                        call=ToolCall(id=id, name="", arguments={}),
                        response=content,
                    ),
                )
            case _:
                raise ValueError(f"Unknown content type: {part.type}")

    def dump_message(self, message: Message) -> dict:
        if len(message.content) == 1 and message.content[0].type == ContentType.text:
            content = message.content[0].text
        else:
            content = [self.dump_content_part(part) for part in message.content]

        return dict(
            role=self.dump_role(message.role),
            content=content,
        )

    def parse_message(self, message: dict | AnthropicMessage) -> Message:
        if isinstance(message, AnthropicMessage):
            message = message.model_dump()

        role = Role.assistant if message.get("role") == "assistant" else Role.user
        content = message.get("content")
        if isinstance(content, str):
            return Message(
                role=role,
                content=[
                    ContentPart(type=ContentType.text, text=content),
                ],
            )

        parts = [self.parse_content_part(part) for part in content]
        if any(part.type == ContentType.tool_response for part in parts):
            role = Role.tool
        return Message(role=role, content=parts)

    def dump_chat(self, chat: Chat) -> tuple[str, list[dict]]:
        system_messages = []
        other_messages = []
        offset = 0

        # If the first message is a system one, extract it as a separate
        # string argument, but *only* if there are more messages. Otherwise
        # include it as usual (note that "system" role will be automatically
        # converted to "user").
        if len(chat) > 1 and chat.messages[0].role == Role.system:
            for part in chat.messages[0].content:
                system_messages.append(part.text)
            offset = 1

        for msg in chat.messages[offset:]:
            other_messages.append(self.dump_message(msg))

        system_message = "\n\n".join(system_messages) if system_messages else NOT_GIVEN
        return system_message, other_messages

    def load_chat(self, messages: list[dict], system: str | None = None) -> Chat:
        c = Chat(system)
        for m in messages:
            c.messages.append(self.parse_message(m))
        return c


class AnthropicClient(LLM):
    provider = "anthropic"
    adapter_class = AnthropicAdapter

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        super().__init__(model, api_key=api_key, base_url=base_url)
        self.client = AsyncAnthropic(api_key=api_key, base_url=base_url)

    async def _internal_call(
        self,
        chat: Chat,
        temperature: float | None,
        max_tokens: int | None,
        adapter: AnthropicAdapter,
        response_format: PydanticResultT | None = None,
    ) -> Message:
        if max_tokens is None:
            max_tokens = 4096

        system_message, messages = adapter.dump_chat(chat)

        try:
            anthropic_message: AnthropicMessage = await self.client.messages.create(
                model=self.model,
                messages=messages,
                temperature=NOT_GIVEN if temperature is None else temperature,
                tools=adapter.spec or NOT_GIVEN,
                max_tokens=max_tokens,
                system=system_message,
            )
        except AuthenticationError as err:
            raise ConfigError(f"Authentication error: {err.message}") from err
        except NotFoundError as err:
            msg = self._error_from_json_response(err.response)
            raise ConfigError(f"Model not found: {msg}") from err
        except AnthropicBadRequestError as err:
            msg = self._error_from_json_response(err.response)
            raise BadRequestError(f"Bad request: {msg}") from err

        return adapter.parse_message(anthropic_message.model_dump())

    async def _internal_stream(
        self,
        chat: Chat,
        adapter: AnthropicAdapter,
        temperature: float | None,
        max_tokens: int | None,
    ) -> AsyncGenerator[str, None]:
        _, messages = adapter.dump_chat(chat)

        system_message, messages = adapter.dump_chat(chat)
        if max_tokens is None:
            max_tokens = 4096

        try:
            stream: AsyncStream[
                RawMessageStreamEvent
            ] = await self.client.messages.create(
                model=self.model,
                messages=messages,
                temperature=NOT_GIVEN if temperature is None else temperature,
                stream=True,
                system=system_message,
                max_tokens=max_tokens,
            )
        except AuthenticationError as err:
            raise ConfigError(f"Authentication error: {err.message}") from err
        except NotFoundError as err:
            msg = self._error_from_json_response(err.response)
            raise ConfigError(f"Model not found: {msg}") from err
        except AnthropicBadRequestError as err:
            msg = self._error_from_json_response(err.response)
            raise BadRequestError(f"Bad request: {msg}") from err

        async for event in stream:
            if event.type == "content_block_delta":
                yield event.delta.text

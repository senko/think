from __future__ import annotations

from base64 import b64decode
from json import JSONDecodeError, loads
from logging import getLogger
from time import time
from typing import AsyncGenerator, Callable, Literal

try:
    from anthropic import NOT_GIVEN, AsyncAnthropic, AsyncStream
    from anthropic.types import Message as AnthropicMessage
    from anthropic.types import RawMessageStreamEvent
    from anthropic.types.image_block_param import Source
except ImportError as err:
    raise ImportError(
        "Anthropic client requires the Anthropic Python SDK: pip install anthropic"
    ) from err

from pydantic import BaseModel, ValidationError

from .base import LLM, CustomParserResultT, PydanticResultT
from .chat import Chat, ContentPart, ContentType, Message, Role
from .tool import ToolCall, ToolDefinition, ToolKit, ToolResponse

log = getLogger(__name__)


class AnthropicAdapter:
    toolkit: ToolKit

    def __init__(self, toolkit: ToolKit | None = None):
        self.toolkit = toolkit

    def get_tool_spec(self, tool: ToolDefinition) -> dict:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.schema,
        }

    @property
    def spec(self) -> list[dict] | None:
        if self.toolkit is None:
            return NOT_GIVEN

        return self.toolkit.generate_tool_spec(self.get_tool_spec)

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
                ),
            ):
                return dict(
                    type="tool_result",
                    tool_use_id=id,
                    content=response,
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

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        super().__init__(model, api_key=api_key, base_url=base_url)
        self.client = AsyncAnthropic(api_key=api_key)

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
        toolkit = self._get_toolkit(tools)

        if max_tokens is None:
            max_tokens = 4096

        adapter = AnthropicAdapter(toolkit)
        system_message, messages = adapter.dump_chat(chat)
        tools_dbg = f" and tools {{{toolkit.tool_names}}}" if toolkit else ""
        log.debug(f"Making a {self.model} call with messages: {messages}{tools_dbg}")
        t0 = time()
        anthropic_message: AnthropicMessage = await self.client.messages.create(
            model=self.model,
            messages=messages,
            temperature=NOT_GIVEN if temperature is None else temperature,
            tools=adapter.spec,
            max_tokens=max_tokens,
        )
        t1 = time()

        anthropic_message = anthropic_message.model_dump()
        log.debug(f"Received response in {(t1 - t0):.1f}s: {anthropic_message}")

        message = adapter.parse_message(anthropic_message)
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

        if parser:
            try:
                if isinstance(parser, type) and issubclass(parser, BaseModel):
                    return parser(**loads(text))
                else:
                    return parser(text)

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

    async def stream(
        self,
        chat: Chat,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        if max_tokens is None:
            max_tokens = 4096

        adapter = AnthropicAdapter()
        system_message, messages = adapter.dump_chat(chat)

        log.debug(f"Making a {self.model} stream call with messages: {messages}")
        stream: AsyncStream[RawMessageStreamEvent] = await self.client.messages.create(
            model=self.model,
            messages=messages,
            temperature=NOT_GIVEN if temperature is None else temperature,
            stream=True,
            system=system_message,
            max_tokens=max_tokens,
        )
        text = ""
        async for event in stream:
            if event.type == "content_block_delta":
                text += event.delta.text
                yield event.delta.text
            else:
                log.debug("AnthropicClient.stream(): ignoring unknown chunk %r", event)

        if text:
            chat.messages.append(
                Message(
                    role=Role.assistant,
                    content=[ContentPart(type=ContentType.text, text=text)],
                )
            )

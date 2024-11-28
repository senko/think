from __future__ import annotations

import json
from logging import getLogger
from time import time
from typing import Any, AsyncGenerator, Callable

from pydantic import BaseModel, ValidationError

try:
    from groq import NOT_GIVEN, AsyncGroq, AsyncStream
    from groq.types.chat import ChatCompletion, ChatCompletionChunk

except ImportError as err:
    raise ImportError(
        "Groq client requires the Groq Python SDK: pip install groq"
    ) from err

from .base import LLM, CustomParserResultT, PydanticResultT
from .chat import Chat, ContentPart, ContentType, Message, Role
from .tool import ToolCall, ToolDefinition, ToolKit, ToolResponse

log = getLogger(__name__)


class GroqAdapter:
    toolkit: ToolKit

    def __init__(self, toolkit: ToolKit | None = None):
        self.toolkit = toolkit

    def get_tool_spec(self, tool: ToolDefinition) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.schema,
            },
        }

    @property
    def spec(self) -> list[dict] | None:
        if self.toolkit is None:
            return NOT_GIVEN

        return self.toolkit.generate_tool_spec(self.get_tool_spec)

    def dump_message(self, message: Message) -> dict:
        role = "assistant" if message.role == Role.assistant else "user"

        if len(message.content) == 1 and message.content[0].type == ContentType.text:
            return {"role": role, "content": message.content[0].text}

        parts = []

        for part in message.content:
            match part:
                case ContentPart(type=ContentType.text, text=text):
                    parts.append(
                        {
                            "type": "text",
                            "text": text,
                        }
                    )
                case ContentPart(type=ContentType.image, image=image):
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image,
                                "detail": "auto",
                            },
                        }
                    )
                case ContentPart(type=ContentType.tool_call, tool_call=tool_call):
                    return {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.name,
                                    "arguments": json.dumps(tool_call.arguments),
                                },
                            },
                        ],
                    }
                case ContentPart(
                    type=ContentType.tool_response,
                    tool_response=ToolResponse(
                        call=call,
                        response=response,
                        error=error,
                    ),
                ):
                    # FIXME - this is ugly!
                    return {
                        "role": "tool",
                        "content": response if response is not None else (error or ""),
                        "name": call.name,
                        "tool_call_id": call.id,
                    }
                case _:
                    log.warning(f"Unsupported content type: {part.type}")
                    continue

        return {
            "role": role,
            "content": parts,
        }

    def dump_chat(self, chat: Chat) -> list[dict]:
        return [self.dump_message(msg) for msg in chat]

    def parse_message(self, message: dict) -> Message:
        role = Role.assistant if message.get("role") == "assistant" else Role.user
        raw_content = message.get("content")
        # FIXME: match on "type" instead
        if isinstance(raw_content, str):
            return Message(
                role=role,
                content=[ContentPart(type=ContentType.text, text=raw_content)],
            )

        parts = []
        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls:
            for raw_call in raw_tool_calls:
                parts.append(
                    ContentPart(
                        type=ContentType.tool_call,
                        tool_call=ToolCall(
                            id=raw_call["id"],
                            name=raw_call["function"]["name"],
                            # FIXME - guard
                            arguments=json.loads(raw_call["function"]["arguments"]),
                        ),
                    )
                )

        return Message(role=role, content=parts)


class GroqClient(LLM):
    provider = "groq"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        super().__init__(model, api_key=api_key, base_url=base_url)
        self.client = AsyncGroq(api_key=api_key, base_url=base_url)

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

        adapter = GroqAdapter(toolkit)
        messages = adapter.dump_chat(chat)
        tools_dbg = f" and tools {', '.join(toolkit.tool_names)}" if toolkit else ""
        log.debug(f"Making a {self.model} call with messages: {messages}{tools_dbg}")
        t0 = time()

        response: ChatCompletion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=NOT_GIVEN if temperature is None else temperature,
            tools=adapter.spec,
            max_tokens=max_tokens,
        )
        t1 = time()

        raw_message = response.choices[0].message.model_dump()
        log.debug(f"Received response in {(t1 - t0):.1f}s: {raw_message}")

        message = adapter.parse_message(raw_message)
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
                    max_retries=max_retries,
                    max_steps=max_steps - 1,
                    max_tokens=max_tokens,
                )

        if parser:
            try:
                if isinstance(parser, type) and issubclass(parser, BaseModel):
                    return parser(**json.loads(text))
                else:
                    return parser(text)

            except (json.JSONDecodeError, ValidationError, ValueError) as err:
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
        adapter = GroqAdapter()
        messages = adapter.dump_chat(chat)

        log.debug(f"Making a {self.model} stream call with messages: {messages}")
        stream: AsyncStream[
            ChatCompletionChunk
        ] = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=NOT_GIVEN if temperature is None else temperature,
            stream=True,
            max_tokens=max_tokens,
        )
        text = ""
        async for chunk in stream:
            cd = chunk.choices[0].delta
            if cd.content:
                text += cd.content
                yield cd.content

        if text:
            chat.messages.append(
                Message(
                    role=Role.assistant,
                    content=[ContentPart(type=ContentType.text, text=text)],
                )
            )

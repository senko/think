from __future__ import annotations

import json
from logging import getLogger
from time import time
from typing import Any, AsyncGenerator, Callable

from pydantic import BaseModel

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
        raise NotImplementedError("Tools are not yet supported on Gemini.")

    @property
    def spec(self) -> list[dict] | None:
        return None

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
        role = Role.assistant if message["role"] == "assistant" else Role.user
        content = [ContentPart(type=ContentType.text, text=message["content"])]
        return Message(role=role, content=content)


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
        return message

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
            chunk_text = chunk.choices[0].delta.content
            text += chunk_text
            yield chunk_text

        if text:
            chat.messages.append(
                Message(
                    role=Role.assistant,
                    content=[ContentPart(type=ContentType.text, text=text)],
                )
            )

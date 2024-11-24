from __future__ import annotations

import json
from logging import getLogger
from time import time
from typing import Any, AsyncGenerator, Callable

from pydantic import BaseModel

try:
    import google.generativeai as genai

except ImportError as err:
    raise ImportError(
        "Gemini client requires the Google GenerativeAI Python SDK: "
        "pip install google-generativeai"
    ) from err

from .base import LLM, CustomParserResultT, PydanticResultT
from .chat import Chat, ContentPart, ContentType, Message, Role
from .tool import ToolCall, ToolDefinition, ToolKit, ToolResponse

log = getLogger(__name__)


class GoogleAdapter:
    toolkit: ToolKit

    def __init__(self, toolkit: ToolKit | None = None):
        self.toolkit = toolkit

    def get_tool_spec(self, tool: ToolDefinition) -> dict:
        raise NotImplementedError("Tools are not yet supported on Gemini.")

    @property
    def spec(self) -> list[dict] | None:
        return None

    def dump_message(self, message: Message) -> list[dict]:
        role = "model" if message.role == Role.assistant else "user"
        parts = []

        for part in message.content:
            match part:
                case ContentPart(type=ContentType.text, text=text):
                    parts.append({"text": text})
                case ContentPart(type=ContentType.image):
                    if part.is_image_url:
                        parts.append(
                            {
                                "file_data": {
                                    "mime_type": part.image_mime_type,
                                    "uri": part.image,
                                }
                            }
                        )
                    else:
                        parts.append(
                            {
                                "inline_data": {
                                    "mime_type": part.image_mime_type,
                                    "data": part.image_data,
                                }
                            }
                        )
                case _:
                    # FIXME: add tool call/response support
                    # Docs: https://ai.google.dev/api/caching#Content
                    log.warning("Unsupported content type: %s", part.type)

        return [
            {
                "role": role,
                "parts": parts,
            }
        ]

    def dump_chat(self, chat: Chat) -> list[dict]:
        messages = []
        for m in chat.messages:
            messages.extend(self.dump_message(m))
        return messages


class GoogleClient(LLM):
    provider = "google"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        super().__init__(model, api_key=api_key, base_url=base_url)
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)

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

        adapter = GoogleAdapter(toolkit)
        messages = adapter.dump_chat(chat)
        tools_dbg = f" and tools {', '.join(toolkit.tool_names)}" if toolkit else ""
        t0 = time()
        log.debug(f"Making a {self.model} call with messages: {messages}{tools_dbg}")
        response = await self.client.generate_content_async(
            messages,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
            stream=False,
            tools=None,  # FIXME: add tool support
            tool_config=None,  # FIXME: add tool support
        )

        t1 = time()
        log.debug(f"Received response in {(t1 - t0):.1f}s: {response.candidates[0]}")

        return response.text

    async def stream(
        self,
        chat: Chat,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        adapter = GoogleAdapter()
        messages = adapter.dump_chat(chat)
        log.debug(f"Making a {self.model} stream call with messages: {messages}")
        response = await self.client.generate_content_async(
            messages,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
            stream=True,
        )

        text = ""
        async for chunk in response:
            text += chunk.text
            yield chunk.text

        if text:
            chat.messages.append(
                Message(
                    role=Role.assistant,
                    content=[
                        ContentPart(
                            type=ContentType.text,
                            text=text,
                        )
                    ],
                )
            )

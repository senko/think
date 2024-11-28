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
        from copy import deepcopy

        def remove_additional_properties(d: dict):
            for v in d.values():
                if isinstance(v, dict):
                    remove_additional_properties(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            remove_additional_properties(item)

            d.pop("additionalProperties", None)
            d.pop("title", None)

        schema = deepcopy(tool.schema)
        remove_additional_properties(schema)

        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": schema,
        }

    @property
    def spec(self) -> dict | None:
        if self.toolkit is None:
            return None

        defs = self.toolkit.generate_tool_spec(self.get_tool_spec)
        return {
            "function_declarations": defs,
        }

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
                case ContentPart(type=ContentType.tool_call, tool_call=tool_call):
                    parts.append(
                        {
                            "function_call": {
                                "name": tool_call.id,
                                "args": tool_call.arguments,
                            }
                        }
                    )
                case ContentPart(
                    type=ContentType.tool_response,
                    tool_response=ToolResponse(
                        call=ToolCall(id=id),
                        response=response,
                        error=error,
                    ),
                ):
                    retval = {}
                    if response:
                        retval["response"] = response
                    if error:
                        retval["error"] = error

                    parts.append(
                        {
                            "function_response": {
                                "name": id,
                                "response": retval,
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

    def parse_message(self, message: dict) -> Message:
        role = Role.assistant if message["role"] == "model" else Role.user

        content_parts = []

        for part in message["parts"]:
            match part:
                case {"text": text}:
                    content_parts.append(
                        ContentPart(
                            type=ContentType.text,
                            text=text,
                        )
                    )
                case {"function_call": {"name": name, "args": args}}:
                    content_parts.append(
                        ContentPart(
                            type=ContentType.tool_call,
                            tool_call=ToolCall(
                                id=name,
                                name=name,
                                arguments=args,
                            ),
                        )
                    )
        return Message(
            role=role,
            content=content_parts,
        )


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
            tools=adapter.spec,
        )

        t1 = time()
        log.debug(f"Received response in {(t1 - t0):.1f}s: {response.candidates[0]}")

        message = adapter.parse_message(response.to_dict()["candidates"][0]["content"])
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
        return text

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

from __future__ import annotations

import json
from logging import getLogger
from typing import AsyncGenerator

try:
    from groq import (
        NOT_GIVEN,
        AsyncGroq,
        AsyncStream,
        AuthenticationError,
        NotFoundError,
        BadRequestError as GroqBadRequestError,
    )
    from groq.types.chat import ChatCompletion, ChatCompletionChunk

except ImportError as err:
    raise ImportError(
        "Groq client requires the Groq Python SDK: pip install groq"
    ) from err

from .base import LLM, BaseAdapter, ConfigError, BadRequestError, PydanticResultT
from .chat import Chat, ContentPart, ContentType, Message, Role
from .tool import ToolCall, ToolDefinition, ToolResponse

log = getLogger(__name__)


class GroqAdapter(BaseAdapter):
    """
    Adapter for Groq API request/response format.

    See `BaseAdapter` for more details on the adapter interface
    and https://console.groq.com/docs/api-reference#chat-create-request-body
    for the Groq API reference.
    """

    def get_tool_spec(self, tool: ToolDefinition) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.schema,
            },
        }

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

    def dump_chat(self, chat: Chat) -> tuple[str, list[dict]]:
        return "", [self.dump_message(msg) for msg in chat]

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
    adapter_class = GroqAdapter

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        super().__init__(model, api_key=api_key, base_url=base_url, **kwargs)
        self.client = AsyncGroq(api_key=api_key, base_url=base_url)

    async def _internal_call(
        self,
        chat: Chat,
        temperature: float | None,
        max_tokens: int | None,
        adapter: GroqAdapter,
        *,
        max_retries: int,
        response_format: PydanticResultT | None = None,
    ) -> Message:
        _, messages = adapter.dump_chat(chat)

        client = self.client.with_options(max_retries=max_retries)

        try:
            response: ChatCompletion = await client.chat.completions.create(  # type: ignore[no-matching-overload]
                model=self.model,
                messages=messages,
                temperature=NOT_GIVEN if temperature is None else temperature,  # type: ignore[arg-type]
                tools=adapter.spec or NOT_GIVEN,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                **self.extra_params,
            )
        except AuthenticationError as err:
            raise ConfigError(f"Authentication error: {err.message}") from err
        except NotFoundError as err:
            msg = self._error_from_json_response(err.response)
            raise ConfigError(f"Model not found: {msg}") from err
        except GroqBadRequestError as err:
            msg = self._error_from_json_response(err.response)
            raise BadRequestError(f"Bad request: {msg}") from err

        return adapter.parse_message(response.choices[0].message.model_dump())

    async def _internal_stream(
        self,
        chat: Chat,
        adapter: GroqAdapter,
        temperature: float | None,
        max_tokens: int | None,
        *,
        max_retries: int,
    ) -> AsyncGenerator[str, None]:
        _, messages = adapter.dump_chat(chat)

        client = self.client.with_options(max_retries=max_retries)

        try:
            stream: AsyncStream[
                ChatCompletionChunk
            ] = await client.chat.completions.create(  # type: ignore[no-matching-overload]
                model=self.model,
                messages=messages,
                temperature=NOT_GIVEN if temperature is None else temperature,  # type: ignore[arg-type]
                stream=True,
                max_tokens=max_tokens,
                **self.extra_params,
            )
        except AuthenticationError as err:
            raise ConfigError(f"Authentication error: {err.message}") from err
        except NotFoundError as err:
            msg = self._error_from_json_response(err.response)
            raise ConfigError(f"Model not found: {msg}") from err
        except GroqBadRequestError as err:
            msg = self._error_from_json_response(err.response)
            raise BadRequestError(f"Bad request: {msg}") from err

        async for chunk in stream:
            cd = chunk.choices[0].delta
            if cd.content:
                yield cd.content

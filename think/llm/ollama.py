from __future__ import annotations

from logging import getLogger
from typing import Any, AsyncGenerator

try:
    from ollama import AsyncClient, Options, ResponseError
except ImportError as err:
    raise ImportError(
        "Ollama client requires the Ollama client library: pip install ollama"
    ) from err

from .base import LLM, BadRequestError, BaseAdapter, ConfigError, PydanticResultT
from .chat import Chat, ContentPart, ContentType, Message, Role
from .tool import ToolCall, ToolDefinition, ToolResponse

log = getLogger(__name__)


class OllamaAdapter(BaseAdapter):
    def get_tool_spec(self, tool: ToolDefinition) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "arguments": tool.schema,
            },
        }

    def dump_message(self, message: Message) -> list[dict[str, str]]:
        message_text = ""
        tool_calls = []
        tool_responses = []
        images = []

        for part in message.content:
            match part:
                case ContentPart(type=ContentType.text, text=text):
                    message_text += text
                case ContentPart(type=ContentType.tool_call, tool_call=tool_call):
                    tool_calls.append(
                        {
                            "function": {
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                            },
                        }
                    )
                case ContentPart(
                    type=ContentType.tool_response, tool_response=tool_response
                ):
                    tool_responses.append(tool_response.response)
                case ContentPart(type=ContentType.image):
                    images.append(part.image_data)

        retval = []
        if message_text or tool_calls or images:
            retval.append(
                {
                    "role": message.role.value,
                    "content": message_text,
                }
            )
        if tool_calls:
            retval[0]["tool_calls"] = tool_calls
        if images:
            retval[0]["images"] = images

        for response in tool_responses:
            retval.append(
                {
                    "role": "tool",
                    "content": response,
                }
            )
        return retval

    def parse_message(self, message: dict[str, Any]) -> Message:
        role_str = message.get("role", "user")
        try:
            role = Role(role_str)
        except ValueError:
            log.error(f"Unknown role: {role_str}, assuming user")
            role = Role.user

        content_parts = []
        text = message.get("content")
        if text:
            if role == Role.tool:
                content_parts.append(
                    ContentPart(
                        type=ContentType.tool_response,
                        tool_response=ToolResponse(
                            response=text,
                            call=ToolCall(
                                id="",
                                name="",
                                arguments={},
                            ),
                        ),
                    )
                )
            else:
                content_parts.append(ContentPart(type=ContentType.text, text=text))

        tool_calls = message.get("tool_calls")
        if tool_calls:
            for call in tool_calls:
                content_parts.append(
                    ContentPart(
                        type=ContentType.tool_call,
                        tool_call=ToolCall(
                            id=call["function"]["name"],
                            name=call["function"]["name"],
                            arguments=call["function"]["arguments"],
                        ),
                    )
                )

        images = message.get("images")
        if images:
            for image in images:
                content_parts.append(ContentPart(type=ContentType.image, image=image))
        return Message(
            role=role,
            content=content_parts,
        )

    def dump_chat(self, chat: Chat) -> tuple[str, list[dict]]:
        messages = []
        for m in chat.messages:
            messages.extend(self.dump_message(m))
        return "", messages

    def load_chat(self, messages: list[dict]) -> Chat:
        c = Chat()
        for m in messages:
            c.messages.append(self.parse_message(m))
        return c


class OllamaClient(LLM):
    provider = "ollama"
    adapter_class = OllamaAdapter

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        super().__init__(model, api_key=api_key, base_url=base_url)
        self.client = AsyncClient(base_url)

    async def _internal_call(
        self,
        chat: Chat,
        temperature: float | None,
        max_tokens: int | None,
        adapter: OllamaAdapter,
        response_format: PydanticResultT | None = None,
    ) -> Message:
        _, messages = adapter.dump_chat(chat)

        try:
            response = await self.client.chat(
                model=self.model,
                messages=messages,
                stream=False,
                options=Options(
                    num_predict=max_tokens,
                    temperature=temperature,
                ),
                tools=adapter.spec,
            )
        except ResponseError as err:
            if err.status_code == 404:
                raise ConfigError(f"Model not found: {err.error}") from err
            else:
                raise BadRequestError(f"Bad request: {err.error}") from err
        except AttributeError as err:
            raise BadRequestError(f"Bad request: {err}") from err

        return adapter.parse_message(response.get("message", {}))

    async def _internal_stream(
        self,
        chat: Chat,
        adapter: OllamaAdapter,
        temperature: float | None,
        max_tokens: int | None,
    ) -> AsyncGenerator[str, None]:
        _, messages = adapter.dump_chat(chat)

        try:
            stream = await self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options=Options(
                    num_predict=max_tokens,
                    temperature=temperature,
                ),
            )

            async for chunk in stream:
                msg = chunk.get("message", {})
                chunk_text = msg.get("content", "")
                if chunk_text:
                    yield chunk_text

        except ResponseError as err:
            if err.status_code == 404:
                raise ConfigError(f"Model not found: {err.error}") from err
            else:
                raise BadRequestError(f"Bad request: {err.error}") from err
        except AttributeError as err:
            raise BadRequestError(f"Bad request: {err}") from err

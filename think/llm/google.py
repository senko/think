from __future__ import annotations

from logging import getLogger
from typing import AsyncGenerator

try:
    import google.generativeai as genai
    from google.api_core.exceptions import InvalidArgument, NotFound

except ImportError as err:
    raise ImportError(
        "Gemini client requires the Google GenerativeAI Python SDK: "
        "pip install google-generativeai"
    ) from err

from .base import LLM, BadRequestError, BaseAdapter, ConfigError, PydanticResultT
from .chat import Chat, ContentPart, ContentType, Message, Role
from .tool import ToolCall, ToolDefinition, ToolResponse

log = getLogger(__name__)


class GoogleAdapter(BaseAdapter):
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
        defs = super().spec
        if not defs:
            return None

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

    def dump_chat(self, chat: Chat) -> tuple[str, list[dict]]:
        messages = []
        for m in chat.messages:
            messages.extend(self.dump_message(m))
        return "", messages

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
    adapter_class = GoogleAdapter

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

    async def _internal_call(
        self,
        chat: Chat,
        temperature: float | None,
        max_tokens: int | None,
        adapter: GoogleAdapter,
        response_format: PydanticResultT | None = None,
    ) -> Message:
        _, messages = adapter.dump_chat(chat)
        try:
            response = await self.client.generate_content_async(
                messages,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
                stream=False,
                tools=adapter.spec,
            )
        except InvalidArgument as err:
            if "API_KEY" in err.reason:
                msg = f"Authentication error: {err.message}"
            else:
                msg = f"Invalid argument: {err.message}"
            raise ConfigError(msg) from err
        except NotFound as err:
            raise ConfigError(f"Unknown model: {err.message}") from err
        except KeyError as err:
            raise BadRequestError(f"Bad request: {err}") from err

        return adapter.parse_message(response.to_dict()["candidates"][0]["content"])

    async def _internal_stream(
        self,
        chat: Chat,
        adapter: GoogleAdapter,
        temperature: float | None,
        max_tokens: int | None,
    ) -> AsyncGenerator[str, None]:
        _, messages = adapter.dump_chat(chat)

        try:
            response = await self.client.generate_content_async(
                messages,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
                stream=True,
            )
        except InvalidArgument as err:
            if "API_KEY" in err.reason:
                msg = f"Authentication error: {err.message}"
            else:
                msg = f"Invalid argument: {err.message}"
            raise ConfigError(msg) from err
        except NotFound as err:
            raise ConfigError(f"Unknown model: {err.message}") from err
        except KeyError as err:
            raise BadRequestError(f"Bad request: {err}") from err

        async for chunk in response:
            yield chunk.text

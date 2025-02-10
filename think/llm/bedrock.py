from __future__ import annotations

from base64 import b64decode
from logging import getLogger
from typing import AsyncGenerator, Literal

try:
    from aioboto3 import Session
    from botocore.exceptions import (
        ClientError,
        EndpointConnectionError,
        NoCredentialsError,
        ParamValidationError,
    )
except ImportError as err:
    raise ImportError(
        "AWS Bedrock client requires the async Boto3 SDK: pip install aioboto3"
    ) from err


from .base import LLM, BadRequestError, BaseAdapter, ConfigError, PydanticResultT
from .chat import Chat, ContentPart, ContentType, Message, Role
from .tool import ToolCall, ToolDefinition, ToolResponse

log = getLogger(__name__)


class BedrockAdapter(BaseAdapter):
    def get_tool_spec(self, tool: ToolDefinition) -> dict:
        from copy import deepcopy

        # COPYPASTED FROM GEMINI
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
            "toolSpec": {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {
                    "json": schema,
                },
            }
        }

    @property
    def spec(self) -> dict | None:
        spec = super().spec
        if spec is None:
            return None
        return {"tools": spec, "toolChoice": {"auto": {}}}

    def dump_role(self, role: Role) -> Literal["user", "assistant"]:
        if role in [Role.system, Role.user, Role.tool]:
            return "user"
        else:
            return "assistant"

    def dump_content_part(self, part: ContentPart) -> dict:
        match part:
            case ContentPart(type=ContentType.text, text=text):
                return dict(
                    text=text,
                )
            case ContentPart(type=ContentType.image):
                return dict(
                    image=dict(
                        source=dict(
                            bytes=part.image_bytes,
                        ),
                        format=part.image_mime_type.split("/")[1],
                    )
                )
            case ContentPart(
                type=ContentType.tool_call,
                tool_call=ToolCall(id=id, name=name, arguments=arguments),
            ):
                return dict(
                    toolUse=dict(
                        toolUseId=id,
                        name=name,
                        input=arguments,
                    ),
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
                    toolResult=dict(
                        toolUseId=id,
                        content=[
                            dict(
                                text=response
                                if response is not None
                                else (error or ""),
                            )
                        ],
                    ),
                )
            case _:
                raise ValueError(f"Unknown content type: {part.type}")

    def parse_content_part(self, part: dict) -> ContentPart:
        match part:
            case {"text": text}:
                return ContentPart(type=ContentType.text, text=text)
            case {"type": "image", "source": {"data": data}}:
                return ContentPart(
                    type=ContentType.image,
                    image=b64decode(data.encode("ascii")),
                )
            case {"toolUse": {"toolUseId": id, "name": name, "input": input}}:
                return ContentPart(
                    type=ContentType.tool_call,
                    tool_call=ToolCall(id=id, name=name, arguments=input),
                )
            case {"toolResult": {"toolUseId": id, "content": {"text": content}}}:
                return ContentPart(
                    type=ContentType.tool_response,
                    tool_response=ToolResponse(
                        call=ToolCall(id=id, name="", arguments={}),
                        response=content,
                    ),
                )
            case _:
                raise ValueError(f"Unknown content type for {part}")

    def dump_message(self, message: Message) -> dict:
        return dict(
            role=self.dump_role(message.role),
            content=[self.dump_content_part(part) for part in message.content],
        )

    def parse_message(self, message: dict) -> Message:
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

        system_message = "\n\n".join(system_messages) if system_messages else None
        return system_message, other_messages

    def load_chat(self, messages: list[dict], system: str | None = None) -> Chat:
        c = Chat(system)
        for m in messages:
            c.messages.append(self.parse_message(m))
        return c


class BedrockClient(LLM):
    provider = "bedrock"
    adapter_class = BedrockAdapter

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: str,
    ):
        super().__init__(model, api_key=api_key, base_url=base_url)

        region = kwargs.get("region")
        if region is None:
            raise ValueError("AWS Bedrock client requires a region to be specified")

        key_id = None
        key_secret = None
        if api_key:
            if ":" not in api_key:
                raise ValueError("AWS Bedrock client requires key ID and secret")
            else:
                key_id, secret = api_key.split(":", 1)
        self.session = Session(
            region_name=region,
            aws_access_key_id=key_id,
            aws_secret_access_key=key_secret,
        )

    async def _internal_call(
        self,
        chat: Chat,
        temperature: float | None,
        max_tokens: int | None,
        adapter: BedrockAdapter,
        response_format: PydanticResultT | None = None,
    ) -> Message:
        system_message, messages = adapter.dump_chat(chat)
        system_block = [{"text": system_message}] if system_message else None

        cfg = {}
        if temperature is not None:
            cfg["temperature"] = temperature
        if max_tokens is not None:
            cfg["maxTokens"] = max_tokens

        async with self.session.client("bedrock-runtime") as client:
            try:
                kwargs = dict(
                    modelId=self.model,
                    messages=messages,
                    system=system_block,
                )
                if cfg:
                    cfg["inferenceConfig"] = cfg
                if adapter.spec:
                    kwargs["toolConfig"] = adapter.spec

                raw_message = await client.converse(**kwargs)
            except (NoCredentialsError, EndpointConnectionError) as err:
                raise ConfigError(err.fmt) from err
            except ClientError as err:
                error = err.response.get("Error", {})
                error_code = error.get("Code")
                error_message = error.get("Message")
                if error_code in [
                    "InvalidSignatureException",
                    "UnrecognizedClientException",
                ]:
                    raise ConfigError(
                        error_message or "Unknown client/credentials error"
                    )
                raise
            except ParamValidationError as err:
                raise BadRequestError(err.fmt) from err
            except:
                raise

        return adapter.parse_message(raw_message["output"]["message"])

    async def _internal_stream(
        self,
        chat: Chat,
        adapter: BedrockAdapter,
        temperature: float | None,
        max_tokens: int | None,
    ) -> AsyncGenerator[str, None]:
        system_message, messages = adapter.dump_chat(chat)
        system_block = [{"text": system_message}] if system_message else None

        cfg = {}
        if temperature is not None:
            cfg["temperature"] = temperature
        if max_tokens is not None:
            cfg["maxTokens"] = max_tokens

        async with self.session.client("bedrock-runtime") as client:
            try:
                response = await client.converse_stream(
                    modelId=self.model,
                    messages=messages,
                    system=system_block,
                    inferenceConfig=cfg,
                )
                stream = response.get("stream")

                async for event in stream:
                    if "contentBlockDelta" in event:
                        yield event["contentBlockDelta"]["delta"]["text"]
            except (NoCredentialsError, EndpointConnectionError) as err:
                raise ConfigError(err.fmt) from err
            except ClientError as err:
                error = err.response.get("Error", {})
                error_code = error.get("Code")
                error_message = error.get("Message")
                if error_code in [
                    "InvalidSignatureException",
                    "UnrecognizedClientException",
                ]:
                    raise ConfigError(
                        error_message or "Unknown client/credentials error"
                    )
                raise
            except ParamValidationError as err:
                raise BadRequestError(err.fmt) from err
            except:
                raise

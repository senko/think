import json
from logging import getLogger
from time import time
from typing import Any, AsyncGenerator, Callable

from pydantic import BaseModel

try:
    from openai import NOT_GIVEN, AsyncOpenAI, AsyncStream
    from openai.types.chat import ChatCompletionChunk

except ImportError as err:
    raise ImportError(
        "OpenAI client requires the OpenAI Python SDK: pip install openai"
    ) from err

from .base import LLM, CustomParserResultT, PydanticResultT
from .chat import Chat, ContentPart, ContentType, Message, Role
from .tool import ToolCall, ToolDefinition, ToolKit, ToolResponse

log = getLogger(__name__)


class OpenAIAdapter:
    toolkit: ToolKit

    def __init__(self, toolkit: ToolKit | None = None):
        self.toolkit = toolkit

    def get_tool_spec(self, tool: ToolDefinition) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "arguments": tool.schema,
            },
        }

    @property
    def spec(self) -> list[dict] | None:
        if self.toolkit is None:
            return NOT_GIVEN

        return self.toolkit.generate_tool_spec(self.get_tool_spec)

    def dump_message(self, message: Message) -> list[dict]:
        tool_calls = []
        tool_responses = {}
        text_parts = []
        image_parts = []

        for part in message.content:
            match part:
                case ContentPart(type=ContentType.tool_call, tool_call=tool_call):
                    tool_calls.append(
                        dict(
                            id=tool_call.id,
                            type="function",
                            function=dict(
                                name=tool_call.name,
                                arguments=json.dumps(tool_call.arguments),
                            ),
                        )
                    )
                case ContentPart(
                    type=ContentType.tool_response, tool_response=tool_response
                ):
                    tool_responses[tool_response.call.id] = tool_response.response
                case ContentPart(type=ContentType.text, text=text):
                    text_parts.append(
                        dict(
                            type="text",
                            text=text,
                        )
                    )
                case ContentPart(type=ContentType.image, image=image):
                    image_parts.append(
                        dict(
                            type="image_url",
                            image_url=dict(url=image),
                        )
                    )

        if tool_responses:
            return [
                dict(
                    role="tool",
                    tool_call_id=call_id,
                    content=response,
                )
                for call_id, response in tool_responses.items()
            ]

        if tool_calls or message.role == Role.assistant:
            if len(text_parts) == 1:
                text_parts = text_parts[0]["text"]
            return [
                dict(
                    role="assistant",
                    content=text_parts or None,
                    tool_calls=tool_calls,
                )
            ]

        if message.role == Role.system:
            if len(text_parts) == 1:
                text_parts = text_parts[0]["text"]
            return [dict(role="system", content=text_parts)]

        if message.role == Role.user:
            content = text_parts + image_parts
            if len(content) == 1 and content[0]["type"] == "text":
                content = content[0]["text"]
            return [
                dict(
                    role="user",
                    content=content,
                )
            ]

        raise ValueError(f"Unsupported message role: {message.role}")

    @staticmethod
    def text_content(
        content: str | list[dict[str, str]],
    ) -> str | None:
        if content is None:
            return None

        if isinstance(content, str):
            return content
        else:
            return "".join(part.get("text", "") for part in content)

    def parse_tool_call(self, message: dict[str, Any]) -> Message:
        tool_call_id = message.get("tool_call_id")
        if tool_call_id is None:
            raise ValueError("Missing tool_call_id in tool message: %r", message)

        text = self.text_content(message.get("content"))

        return Message(
            role=Role.tool,
            content=[
                ContentPart(
                    type=ContentType.tool_response,
                    tool_response=ToolResponse(
                        call=ToolCall(
                            id=tool_call_id,
                            name="",
                            arguments={},
                        ),
                        response=text,
                    ),
                )
            ],
        )

    def parse_assistant_message(self, message: dict[str, Any]) -> Message:
        raw_tool_calls = message.get("tool_calls")
        tool_calls = []
        if raw_tool_calls:
            for tc in raw_tool_calls:
                call_id = tc.get("id")
                if call_id is None:
                    raise ValueError(
                        "Missing tool call ID in assistant message: %r", tc
                    )
                name = tc.get("function", {}).get("name")
                if name is None:
                    raise ValueError(
                        "Missing function name in assistant message: %r", tc
                    )
                arguments = tc.get("function", {}).get("arguments")
                if arguments is None:
                    raise ValueError(
                        "Missing function arguments in assistant message: %r", tc
                    )

                tool_calls.append(
                    ContentPart(
                        type=ContentType.tool_call,
                        tool_call=ToolCall(
                            id=call_id,
                            name=name,
                            arguments=json.loads(arguments),
                        ),
                    )
                )

        text = self.text_content(message.get("content"))
        parts = []
        if text:
            parts.append(
                ContentPart(
                    type=ContentType.text,
                    text=text or "",
                )
            )

        return Message(
            role=Role.assistant,
            content=parts + tool_calls,
        )

    def parse_message(self, message: dict[str, Any]) -> Message:
        role = message.get("role", "user")
        if role == "tool":
            return self.parse_tool_call(message)

        elif role == "assistant":
            return self.parse_assistant_message(message)

        elif role == "system":
            text = self.text_content(message.get("content"))
            return Message.system(text)

        elif role == "user":
            raw_content = message.get("content")
            if raw_content is None:
                raise ValueError("Missing content in user message: %r", message)

            content: list[ContentPart] = []
            if isinstance(raw_content, str):
                content.append(
                    ContentPart(
                        type=ContentType.text,
                        text=raw_content,
                    )
                )
            else:
                for part in raw_content:
                    part_type = part.get("type")
                    if part_type == "text":
                        content.append(
                            ContentPart(
                                type=ContentType.text,
                                text=part.get("text"),
                            )
                        )
                    elif part_type == "image_url":
                        content.append(
                            ContentPart(
                                type=ContentType.image,
                                image=part.get("image_url", {}).get("url"),
                            )
                        )
                    else:
                        raise ValueError(f"Unsupported content part type: {part_type}")

            return Message(role=Role.user, content=content)

        raise ValueError(f"Unsupported message type: {type(message)}")

    def dump_chat(self, chat: Chat) -> list[dict]:
        messages = []
        for m in chat.messages:
            messages.extend(self.dump_message(m))
        return messages

    def load_chat(self, messages: list[dict]) -> Chat:
        c = Chat()
        for m in messages:
            c.messages.append(self.parse_message(m))
        return c


class OpenAIClient(LLM):
    provider = "openai"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        super().__init__(model, api_key=api_key, base_url=base_url)
        self.client = AsyncOpenAI(api_key=api_key)

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

        if isinstance(parser, type) and issubclass(parser, BaseModel):
            response_format = parser
        else:
            response_format = None

        adapter = OpenAIAdapter(toolkit)
        messages = adapter.dump_chat(chat)
        tools_dbg = f" and tools {{{toolkit.tool_names}}}" if toolkit else ""
        t0 = time()
        if response_format:
            log.debug(
                f"Making a {self.model} call with messages: {messages}{tools_dbg}"
                f"expecting {response_format.__name__} response"
            )
            response = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                temperature=temperature,
                tools=adapter.spec,
                response_format=response_format,
                max_completion_tokens=max_tokens or NOT_GIVEN,
            )
        else:
            log.debug(
                f"Making a {self.model} call with messages: {messages}{tools_dbg}"
            )
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                tools=adapter.spec,
                max_completion_tokens=max_tokens or NOT_GIVEN,
            )

        t1 = time()
        log.debug(
            f"Received response in {(t1 - t0):.1f}s: {response.choices[0].message}"
        )

        message = adapter.parse_message(response.choices[0].message.model_dump())
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

        if response_format and response.choices[0].message.parsed:
            return response.choices[0].message.parsed

        if parser:
            try:
                return parser(text)
            except ValueError as err:
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
                    max_retries=max_retries - 1,
                    max_steps=max_steps,
                    max_tokens=max_tokens,
                )

        return text

    async def stream(
        self,
        chat: Chat,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        adapter = OpenAIAdapter()
        messages = adapter.dump_chat()
        log.debug(f"Making a {self.model} stream call with messages: {messages}")
        stream: AsyncStream[
            ChatCompletionChunk
        ] = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
            max_completion_tokens=max_tokens or NOT_GIVEN,
        )
        text = ""
        async for chunk in stream:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content is not None:
                text += delta.content
                yield delta.content
            elif choice.finish_reason == "stop":
                pass  # ignore
            else:
                log.debug("OpenAIClient.stream(): ignoring unknown chunk %r", chunk)

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

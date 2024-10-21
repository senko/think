from base64 import b64decode, b64encode
from json import loads
from logging import getLogger
from typing import AsyncGenerator, Callable, Literal, TypeGuard, Union

from anthropic import (
    NOT_GIVEN,
    AsyncAnthropic,
    AsyncStream,
)
from anthropic.types import (
    ContentBlock,
    ImageBlockParam,
    MessageParam,
    RawMessageStreamEvent,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from anthropic.types.image_block_param import Source
from pydantic import BaseModel

from .base import LLM
from .chat import Chat, ContentPart, ContentType, Message, Role
from .tool import ToolCall, ToolDefinition, ToolKit, ToolResponse

AnthropicContentPart = Union[
    TextBlockParam,
    ImageBlockParam,
    ToolUseBlockParam,
    ToolResultBlockParam,
    ContentBlock,
    Source,
]


log = getLogger(__name__)


class AnthropicMessageAdapter:
    def dump_role(self, role: Role) -> Literal["user", "assistant"]:
        if role in [Role.system, Role.user, Role.tool]:
            return "user"
        else:
            return "assistant"

    def dump_content_part(self, part: ContentPart) -> AnthropicContentPart:
        if part.type == ContentType.text:
            return TextBlockParam(
                type="text",
                text=part.text,
            )
        elif part.type == ContentType.image:
            return ImageBlockParam(
                type="image",
                source=Source(
                    url=b64encode(part.get_image_data()).decode("ascii"),
                    media_type=part.get_image_mime_type(),
                ),
            )
        elif part.type == ContentType.tool_call:
            return ToolUseBlockParam(
                type="tool_use",
                id=part.tool_call.id,
                name=part.tool_call.name,
                input=part.tool_call.arguments,
            )
        elif part.type == ContentType.tool_response:
            return ToolResultBlockParam(
                type="tool_result",
                tool_use_id=part.tool_response.call.id,
                content=part.tool_response.response,
            )
        else:
            raise ValueError(f"Unknown content type: {part.type}")

    def is_text(self, part: AnthropicContentPart) -> TypeGuard[TextBlockParam]:
        return part.type == "text"

    def is_image(self, part: AnthropicContentPart) -> TypeGuard[ImageBlockParam]:
        return part.type == "image"

    def is_tool_use(self, part: AnthropicContentPart) -> TypeGuard[ToolUseBlockParam]:
        return part.type == "tool_use"

    def is_tool_result(
        self,
        part: AnthropicContentPart,
    ) -> TypeGuard[ToolResultBlockParam]:
        return part.type == "tool_result"

    def parse_content_part(self, part: AnthropicContentPart) -> ContentPart:
        if self.is_text(part):
            return ContentPart(type=ContentType.text, text=part.text)
        elif self.is_image(part):
            return ContentPart(
                type=ContentType.image,
                image=b64decode(part.source.data.encode("ascii")),
            )
        elif self.is_tool_use(part):
            return ContentPart(
                type=ContentType.tool_call,
                tool_call=ToolCall(id=part.id, name=part.name, arguments=part.input),
            )
        elif self.is_tool_result(part):
            return ContentPart(
                type=ContentType.tool_response,
                tool_response=ToolResponse(
                    call=ToolCall(id=part.id, name=None, arguments=None),
                    response=part.content,
                ),
            )
        else:
            raise ValueError(f"Unknown content type: {part.type}")

    def dump_message(self, message: Message) -> MessageParam:
        if len(message.content) == 1 and message.content[0].type == ContentType.text:
            content = message.content[0].text
        else:
            content = [self.dump_content_part(part) for part in message.content]

        return MessageParam(
            role=self.dump_role(message.role),
            content=content,
        )

    def parse_message(self, message: MessageParam) -> Message:
        role = Role.assistant if message.role == "assistant" else Role.user
        if isinstance(message.content, str):
            return Message.create(role=role, text=message.content)

        content = [self.parse_content_part(part) for part in message.content]
        return Message(role=role, content=content)

    def dump_chat(self, chat: Chat) -> tuple[str, list[MessageParam]]:
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


class AnthropicToolAdapter:
    toolkit: ToolKit

    def __init__(self, toolkit: ToolKit | None):
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
            return None

        return [self.get_tool_spec(tool) for tool in self.toolkit.tools.values()]


class AnthropicClient(LLM):
    provider = "openai"

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
        temperature: float | None = None,
        tools: ToolKit | list[Callable] | None = None,
        parser: BaseModel | Callable | None = None,
        max_retries: int = 3,
    ) -> str:
        if tools:
            if isinstance(tools, ToolKit):
                toolkit = tools
            elif isinstance(tools[0], Callable):
                toolkit = ToolKit(tools)
            else:
                raise TypeError(f"Unsupported tools type: {type(tools)}")
        else:
            toolkit = None

        adapter = AnthropicMessageAdapter()
        tool_adapter = AnthropicToolAdapter(toolkit) if toolkit else None
        system_message, messages = adapter.dump_chat(chat)
        print("TOOLS", toolkit.tool_schemas if toolkit else None)
        print("MESSAGES", system_message, messages)
        message: Message = await self.client.messages.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            tools=tool_adapter.spec,
            stream=False,
            max_tokens=4096,  # FIXME
        )

        # TODO: Record response timings and potentially other metadata here

        text = ""
        response_list: list[ToolResponse] = []

        msg = adapter.parse_message(message)
        print("Parsed message", msg)
        chat.messages.append(msg)

        for part in msg.content:
            # FIXME: this completely ignores images in responses
            if part.type == ContentType.text:
                text += part.text
            elif part.type == ContentType.tool_call:
                tool_response = await toolkit.execute_tool_call(part.tool_call)
                print("TOOL RESPONSE", tool_response)
                response_list.append(tool_response)

        if response_list:
            chat.messages.append(
                Message(
                    role=Role.tool,
                    content=[
                        ContentPart(
                            type=ContentType.tool_response,
                            tool_response=tool_response,
                        )
                        for tool_response in response_list
                    ],
                )
            )
            return await self(
                chat,
                temperature=temperature,
                tools=tools,
                parser=parser,
                max_retries=max_retries,
            )

        if parser:
            if isinstance(parser, type) and issubclass(parser, BaseModel):
                try:
                    return parser(**loads(text))
                except Exception as err:
                    # FIXME: duplication
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
                    )

            try:
                return parser(text)
            except ValueError as err:
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
                )

        return text

    async def stream(
        self,
        chat: Chat,
        temperature: float | None = None,
    ) -> AsyncGenerator[str, None]:
        adapter = AnthropicMessageAdapter()
        system_message, messages = adapter.dump_chat(chat)

        print("MESSAGES", messages)
        stream: AsyncStream[RawMessageStreamEvent] = await self.client.messages.create(
            model=self.model,
            messages=messages,
            temperature=NOT_GIVEN if temperature is None else temperature,
            stream=True,
            system=system_message or NOT_GIVEN,
            max_tokens=4096,  # FIXME
        )
        text = ""
        async for event in stream:
            if event.type == "content_block_delta":
                text += event.delta.text
                yield event.delta.text
            else:
                log.debug("OpenAIClient.stream(): ignoring unknown chunk %r", event)

        if text:
            chat.messages.append(Message.create(Role.assistant, text=text))

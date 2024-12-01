import json
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from think.llm.base import LLM, BaseAdapter, PydanticResultT
from think.llm.chat import Chat, ContentPart, ContentType, Message, Role
from think.llm.tool import ToolCall, ToolDefinition, ToolError


class MyAdapter(BaseAdapter):
    def get_tool_spec(self, tool: ToolDefinition) -> dict:
        return {
            "name": tool.name,
        }


class MyClient(LLM):
    adapter_class = MyAdapter

    async def _internal_call(
        self,
        chat: Chat,
        temperature: float | None,
        max_tokens: int | None,
        adapter: BaseAdapter,
        response_format: PydanticResultT | None = None,
    ) -> Message: ...

    async def _internal_stream(
        self,
        chat: Chat,
        adapter: BaseAdapter,
        temperature: float | None,
        max_tokens: int | None,
    ) -> AsyncGenerator[str, None]: ...


def text_message(text: str) -> Message:
    return Message(
        role=Role.assistant,
        content=[
            ContentPart(
                type=ContentType.text,
                text=text,
            )
        ],
    )


def tool_call_message():
    return Message(
        role=Role.assistant,
        content=[
            ContentPart(
                type=ContentType.tool_call,
                tool_call=ToolCall(
                    id="id",
                    name="fake_tool",
                    arguments={
                        "a": 1,
                        "b": "hi",
                    },
                ),
            )
        ],
    )


@pytest.mark.asyncio
async def test_call_minimal():
    chat = Chat("system message").user("user message")
    client = MyClient(api_key="fake-key", model="fake-model")
    response_msg = text_message("Hi!")

    assert client.api_key == "fake-key"
    assert client.model == "fake-model"

    client._internal_call = AsyncMock(return_value=response_msg)
    response = await client(chat, temperature=0.5, max_tokens=10)

    assert response == "Hi!"

    client._internal_call.assert_called_once()
    args = client._internal_call.call_args

    assert args.args[0] == chat
    assert args.args[1] == 0.5  # temperature
    assert args.args[2] == 10  # max_tokens
    assert isinstance(args.args[3], MyAdapter)
    assert args.kwargs["response_format"] is None  # response_format

    assert chat.messages[-1] is response_msg


@pytest.mark.asyncio
async def test_call_with_tools():
    chat = Chat("system message").user("user message")
    client = MyClient(api_key="fake-key", model="fake-model")

    client._internal_call = AsyncMock(
        side_effect=[
            tool_call_message(),
            text_message("Hi!"),
        ]
    )

    def fake_tool(a: int, b: str) -> str:
        """Do something"""
        assert a == 1
        assert b == "hi"
        return "tool response"

    response = await client(chat, tools=[fake_tool], max_steps=1)

    client._internal_call.assert_called()
    assert client._internal_call.call_count == 2
    args = client._internal_call.call_args_list[1]
    assert args.args[0] == chat
    assert response == "Hi!"

    tc = chat.messages[-2].content[0].tool_response
    assert tc is not None
    assert tc.call.id == "id"
    assert tc.call.name == "fake_tool"
    assert tc.call.arguments == {"a": 1, "b": "hi"}
    assert tc.response == "tool response"


@pytest.mark.asyncio
async def test_call_with_tool_error():
    chat = Chat("system message").user("user message")
    client = MyClient(api_key="fake-key", model="fake-model")

    client._internal_call = AsyncMock(
        side_effect=[
            tool_call_message(),
            text_message("Hi!"),
        ]
    )

    def fake_tool(a: int, b: str) -> str:
        """Do something"""
        raise ToolError("some error")

    response = await client(chat, tools=[fake_tool], max_steps=1)

    client._internal_call.assert_called()
    assert client._internal_call.call_count == 2
    args = client._internal_call.call_args_list[1]
    assert args.args[0] == chat
    assert response == "Hi!"

    tc = chat.messages[-2].content[0].tool_response
    assert tc is not None
    assert "some error" in tc.error


@pytest.mark.asyncio
async def test_call_with_pydantic():
    chat = Chat("system message").user("user message")
    client = MyClient(api_key="fake-key", model="fake-model")
    client._internal_call = AsyncMock(
        return_value=text_message(
            json.dumps(
                {
                    "text": "Hi!",
                }
            )
        )
    )

    class TestModel(BaseModel):
        text: str

    response = await client(chat, parser=TestModel)

    assert isinstance(response, TestModel)
    assert response.text == "Hi!"
    assert chat.messages[-1].parsed == response

    client._internal_call.assert_called_once()
    args = client._internal_call.call_args

    assert args.args[0] == chat
    assert args.kwargs["response_format"] is TestModel


@pytest.mark.asyncio
async def test_call_with_custom_parser():
    chat = Chat("system message").user("user message")
    client = MyClient(api_key="fake-key", model="fake-model")
    client._internal_call = AsyncMock(return_value=text_message("Hi!"))

    def custom_parser(val: str) -> float:
        assert val == "Hi!"
        return 0.5

    response = await client(chat, parser=custom_parser)

    assert response == 0.5
    assert chat.messages[-1].content[0].text == "Hi!"
    assert chat.messages[-1].parsed == response


@pytest.mark.asyncio
async def test_streaming():
    chat = Chat("system message").user("user message")
    client = MyClient(api_key="fake-key", model="fake-model")

    assert client.api_key == "fake-key"
    assert client.model == "fake-model"

    original_message = "hello beautiful word"

    async def do_stream():
        for c in original_message:
            yield c

    client._internal_stream = MagicMock(return_value=do_stream())
    text = []
    async for word in client.stream(chat, temperature=0.5, max_tokens=10):
        text.append(word)

    assert "".join(text) == original_message

    client._internal_stream.assert_called_once()
    args = client._internal_stream.call_args

    assert args.args[0] == chat
    assert isinstance(args.args[1], MyAdapter)
    assert args.args[2] == 0.5  # temperature
    assert args.args[3] == 10  # max_tokens

    assert chat.messages[-1].content[0].text == original_message

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from anthropic import NOT_GIVEN
from pydantic import BaseModel
from test_anthropic_adapter import (
    BASIC_ANTHROPIC_MESSAGES,
    BASIC_CHAT,
)

from think.llm.anthropic import AnthropicClient
from think.llm.chat import Chat


def response_fixture(msg: dict):
    return MagicMock(model_dump=MagicMock(return_value=msg))


@pytest.mark.asyncio
@patch("think.llm.anthropic.AsyncAnthropic")
async def test_call_minimal(AsyncAnthropic):
    chat = Chat.load(BASIC_CHAT)
    client = AnthropicClient(api_key="fake-key", model="fake-model")

    mock_create = AsyncMock(return_value=response_fixture(BASIC_ANTHROPIC_MESSAGES[-1]))
    AsyncAnthropic.return_value.messages.create = mock_create

    response = await client(chat)

    assert response == "Hi!"
    mock_create.assert_called_once_with(
        model="fake-model",
        messages=BASIC_ANTHROPIC_MESSAGES,
        tools=NOT_GIVEN,
        temperature=NOT_GIVEN,
        max_tokens=4096,
    )

    AsyncAnthropic.assert_called_once_with(api_key="fake-key")


@pytest.mark.asyncio
@patch("think.llm.anthropic.AsyncAnthropic")
async def test_call_with_options(AsyncAnthropic):
    chat = Chat.load(BASIC_CHAT)
    client = AnthropicClient(api_key="fake-key", model="fake-model")

    mock_create = AsyncMock(return_value=response_fixture(BASIC_ANTHROPIC_MESSAGES[-1]))
    AsyncAnthropic.return_value.messages.create = mock_create

    response = await client(chat, temperature=0.5, max_tokens=10)

    assert response == "Hi!"
    mock_create.assert_called_once_with(
        model="fake-model",
        messages=BASIC_ANTHROPIC_MESSAGES,
        tools=NOT_GIVEN,
        temperature=0.5,
        max_tokens=10,
    )


@pytest.mark.asyncio
@patch("think.llm.anthropic.AsyncAnthropic")
async def test_call_with_tools(AsyncAnthropic):
    chat = Chat.load(BASIC_CHAT)
    client = AnthropicClient(api_key="fake-key", model="fake-model")

    tool_call = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "call_boPTOC8z660AYdRFr9oogH4O",
                "name": "fake_tool",
                "input": {"a": 1, "b": "hi"},
            }
        ],
    }

    tool_defs = [
        {
            "name": "fake_tool",
            "description": "Do something",
            "input_schema": {
                "additionalProperties": False,
                "properties": {
                    "a": {"title": "A", "type": "integer"},
                    "b": {"title": "B", "type": "string"},
                },
                "required": ["a", "b"],
                "type": "object",
            },
        }
    ]

    mock_create = AsyncMock(return_value=response_fixture(tool_call))
    AsyncAnthropic.return_value.messages.create = mock_create

    def fake_tool(a: int, b: str) -> str:
        """Do something"""
        assert a == 1
        assert b == "hi"
        return "tool response"

    response = await client(chat, tools=[fake_tool], max_steps=1)

    assert response == ""
    mock_create.assert_has_calls(
        [
            call(
                model="fake-model",
                messages=BASIC_ANTHROPIC_MESSAGES,
                temperature=NOT_GIVEN,
                tools=tool_defs,
                max_tokens=4096,
            ),
            call(
                model="fake-model",
                messages=BASIC_ANTHROPIC_MESSAGES
                + [
                    tool_call,
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call_boPTOC8z660AYdRFr9oogH4O",
                                "content": "tool response",
                            },
                        ],
                    },
                ],
                temperature=NOT_GIVEN,
                tools=tool_defs,
                max_tokens=4096,
            ),
        ]
    )


@pytest.mark.asyncio
@patch("think.llm.anthropic.AsyncAnthropic")
async def test_call_with_pydantic(AsyncAnthropic):
    chat = Chat.load(BASIC_CHAT)
    client = AnthropicClient(api_key="fake-key", model="fake-model")

    class TestModel(BaseModel):
        text: str

    mock_create = AsyncMock(
        return_value=response_fixture(
            {"role": "assistant", "content": '{"text": "Hi!"}'}
        )
    )
    AsyncAnthropic.return_value.messages.create = mock_create

    response = await client(chat, parser=TestModel)

    assert response.text == "Hi!"
    mock_create.assert_called_once_with(
        model="fake-model",
        messages=BASIC_ANTHROPIC_MESSAGES,
        tools=NOT_GIVEN,
        temperature=NOT_GIVEN,
        max_tokens=4096,
    )

    AsyncAnthropic.assert_called_once_with(api_key="fake-key")


@pytest.mark.asyncio
@patch("think.llm.anthropic.AsyncAnthropic")
async def test_call_with_custom_parser(AsyncAnthropic):
    chat = Chat.load(BASIC_CHAT)
    client = AnthropicClient(api_key="fake-key", model="fake-model")

    mock_create = AsyncMock(return_value=response_fixture(BASIC_ANTHROPIC_MESSAGES[-1]))
    AsyncAnthropic.return_value.messages.create = mock_create

    def custom_parser(val: str) -> float:
        assert val == "Hi!"
        return 0.5

    response = await client(chat, parser=custom_parser)

    assert response == 0.5
    mock_create.assert_called_once_with(
        model="fake-model",
        messages=BASIC_ANTHROPIC_MESSAGES,
        tools=NOT_GIVEN,
        temperature=NOT_GIVEN,
        max_tokens=4096,
    )

    AsyncAnthropic.assert_called_once_with(api_key="fake-key")

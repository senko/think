from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from openai import NOT_GIVEN
from pydantic import BaseModel
from test_openai_adapter import (
    BASIC_CHAT,
    BASIC_OPENAI_MESSAGES,
)

from think.llm.chat import Chat
from think.llm.openai import OpenAIClient


def response_fixture(msg: dict):
    return MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    model_dump=MagicMock(return_value=msg),
                ),
            ),
        ]
    )


def parsed_fixture(msg: dict, val: BaseModel):
    fixture = response_fixture(msg)
    fixture.choices[0].message.parsed = val
    return fixture


@pytest.mark.asyncio
@patch("think.llm.openai.AsyncOpenAI")
async def test_call_minimal(AsyncOpenAI):
    chat = Chat.load(BASIC_CHAT)
    client = OpenAIClient(api_key="fake-key", model="fake-model")

    mock_create = AsyncMock(return_value=response_fixture(BASIC_OPENAI_MESSAGES[-1]))
    AsyncOpenAI.return_value.chat.completions.create = mock_create

    response = await client(chat)

    assert response == "Hi!"
    mock_create.assert_called_once_with(
        model="fake-model",
        messages=BASIC_OPENAI_MESSAGES,
        tools=NOT_GIVEN,
        temperature=None,
        max_completion_tokens=NOT_GIVEN,
    )

    AsyncOpenAI.assert_called_once_with(api_key="fake-key")


@pytest.mark.asyncio
@patch("think.llm.openai.AsyncOpenAI")
async def test_call_with_options(AsyncOpenAI):
    chat = Chat.load(BASIC_CHAT)
    client = OpenAIClient(api_key="fake-key", model="fake-model")

    mock_create = AsyncMock(return_value=response_fixture(BASIC_OPENAI_MESSAGES[-1]))
    AsyncOpenAI.return_value.chat.completions.create = mock_create

    response = await client(chat, temperature=0.5, max_tokens=10)

    assert response == "Hi!"
    mock_create.assert_called_once_with(
        model="fake-model",
        messages=BASIC_OPENAI_MESSAGES,
        tools=NOT_GIVEN,
        temperature=0.5,
        max_completion_tokens=10,
    )


@pytest.mark.asyncio
@patch("think.llm.openai.AsyncOpenAI")
async def test_call_with_tools(AsyncOpenAI):
    chat = Chat.load(BASIC_CHAT)
    client = OpenAIClient(api_key="fake-key", model="fake-model")

    tool_defs = [
        {
            "type": "function",
            "function": {
                "name": "fake_tool",
                "description": "Do something",
                "arguments": {
                    "additionalProperties": False,
                    "properties": {
                        "a": {"title": "A", "type": "integer"},
                        "b": {"title": "B", "type": "string"},
                    },
                    "required": ["a", "b"],
                    "type": "object",
                },
            },
        }
    ]

    tool_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_boPTOC8z660AYdRFr9oogH4O",
                "type": "function",
                "function": {
                    "name": "fake_tool",
                    "arguments": '{"a": 1, "b": "hi"}',
                },
            }
        ],
    }

    mock_create = AsyncMock(return_value=response_fixture(tool_call))
    AsyncOpenAI.return_value.chat.completions.create = mock_create

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
                messages=BASIC_OPENAI_MESSAGES,
                tools=tool_defs,
                temperature=None,
                max_completion_tokens=NOT_GIVEN,
            ),
            call(
                model="fake-model",
                messages=BASIC_OPENAI_MESSAGES
                + [
                    tool_call,
                    {
                        "role": "tool",
                        "tool_call_id": "call_boPTOC8z660AYdRFr9oogH4O",
                        "content": "tool response",
                    },
                ],
                tools=tool_defs,
                temperature=None,
                max_completion_tokens=NOT_GIVEN,
            ),
        ]
    )


@pytest.mark.asyncio
@patch("think.llm.openai.AsyncOpenAI")
async def test_call_with_pydantic(AsyncOpenAI):
    chat = Chat.load(BASIC_CHAT)
    client = OpenAIClient(api_key="fake-key", model="fake-model")

    class TestModel(BaseModel):
        text: str

    mock_parse = AsyncMock(
        return_value=parsed_fixture(
            BASIC_OPENAI_MESSAGES[-1],
            TestModel(text="Hi!"),
        )
    )
    AsyncOpenAI.return_value.beta.chat.completions.parse = mock_parse

    response = await client(chat, parser=TestModel)

    assert response.text == "Hi!"
    mock_parse.assert_called_once_with(
        model="fake-model",
        messages=BASIC_OPENAI_MESSAGES,
        tools=NOT_GIVEN,
        temperature=None,
        response_format=TestModel,
        max_completion_tokens=NOT_GIVEN,
    )

    AsyncOpenAI.assert_called_once_with(api_key="fake-key")


@pytest.mark.asyncio
@patch("think.llm.openai.AsyncOpenAI")
async def test_call_with_custom_parser(AsyncOpenAI):
    chat = Chat.load(BASIC_CHAT)
    client = OpenAIClient(api_key="fake-key", model="fake-model")

    mock_create = AsyncMock(return_value=response_fixture(BASIC_OPENAI_MESSAGES[-1]))
    AsyncOpenAI.return_value.chat.completions.create = mock_create

    def custom_parser(val: str) -> float:
        assert val == "Hi!"
        return 0.5

    response = await client(chat, parser=custom_parser)

    assert response == 0.5
    mock_create.assert_called_once_with(
        model="fake-model",
        messages=BASIC_OPENAI_MESSAGES,
        tools=NOT_GIVEN,
        temperature=None,
        max_completion_tokens=NOT_GIVEN,
    )

    AsyncOpenAI.assert_called_once_with(api_key="fake-key")

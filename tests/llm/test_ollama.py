from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from pydantic import BaseModel
from test_ollama_adapter import (
    BASIC_CHAT,
    BASIC_OLLAMA_MESSAGES,
)

from think.llm.chat import Chat
from think.llm.ollama import OllamaClient


def response_fixture(msg: dict):
    return {
        "message": msg,
    }


def parsed_fixture(msg: dict, val: BaseModel):
    fixture = response_fixture(msg)
    fixture.choices[0].message.parsed = val
    return fixture


@pytest.mark.asyncio
@patch("think.llm.ollama.AsyncClient")
async def test_call_minimal(AsyncClient):
    chat = Chat.load(BASIC_CHAT)
    client = OllamaClient(base_url="http://localhost:11434/", model="fake-model")

    mock_chat = AsyncMock(return_value=response_fixture(BASIC_OLLAMA_MESSAGES[-1]))
    AsyncClient.return_value.chat = mock_chat

    response = await client(chat)

    assert response == "Hi!"
    mock_chat.assert_called_once_with(
        model="fake-model",
        messages=BASIC_OLLAMA_MESSAGES,
        stream=False,
        tools=None,
        options=dict(
            temperature=None,
            num_predict=None,
        ),
    )

    AsyncClient.assert_called_once_with("http://localhost:11434/")


@pytest.mark.asyncio
@patch("think.llm.ollama.AsyncClient")
async def test_call_with_options(AsyncClient):
    chat = Chat.load(BASIC_CHAT)
    client = OllamaClient(base_url="http://localhost:11434/", model="fake-model")

    mock_chat = AsyncMock(return_value=response_fixture(BASIC_OLLAMA_MESSAGES[-1]))
    AsyncClient.return_value.chat = mock_chat

    response = await client(chat, temperature=0.5, max_tokens=10)

    assert response == "Hi!"
    mock_chat.assert_called_once_with(
        model="fake-model",
        messages=BASIC_OLLAMA_MESSAGES,
        tools=None,
        stream=False,
        options=dict(
            temperature=0.5,
            num_predict=10,
        ),
    )


@pytest.mark.asyncio
@patch("think.llm.ollama.AsyncClient")
async def test_call_with_tools(AsyncClient):
    chat = Chat.load(BASIC_CHAT)
    client = OllamaClient(base_url="http://localhost:11434/", model="fake-model")

    tool_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "fake_tool",
                    "arguments": {
                        "a": 1,
                        "b": "hi",
                    },
                },
            }
        ],
    }

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

    mock_chat = AsyncMock(return_value=response_fixture(tool_call))
    AsyncClient.return_value.chat = mock_chat

    def fake_tool(a: int, b: str) -> str:
        """Do something"""
        assert a == 1
        assert b == "hi"
        return "tool response"

    response = await client(chat, tools=[fake_tool], max_steps=1)

    assert response == ""
    mock_chat.assert_has_calls(
        [
            call(
                model="fake-model",
                messages=BASIC_OLLAMA_MESSAGES,
                tools=tool_defs,
                stream=False,
                options=dict(
                    temperature=None,
                    num_predict=None,
                ),
            ),
            call(
                model="fake-model",
                messages=BASIC_OLLAMA_MESSAGES
                + [
                    tool_call,
                    {
                        "role": "tool",
                        "content": "tool response",
                    },
                ],
                tools=tool_defs,
                stream=False,
                options=dict(
                    temperature=None,
                    num_predict=None,
                ),
            ),
        ]
    )


@pytest.mark.asyncio
@patch("think.llm.ollama.AsyncClient")
async def test_call_with_pydantic(AsyncClient):
    chat = Chat.load(BASIC_CHAT)
    client = OllamaClient(base_url="http://localhost:11434/", model="fake-model")

    class TestModel(BaseModel):
        text: str

    mock_chat = AsyncMock(
        return_value=response_fixture(
            {"role": "assistant", "content": '{"text": "Hi!"}'}
        )
    )
    AsyncClient.return_value.chat = mock_chat

    response = await client(chat, parser=TestModel)

    assert response.text == "Hi!"
    assert chat.messages[-1].parsed == response

    mock_chat.assert_called_once_with(
        model="fake-model",
        messages=BASIC_OLLAMA_MESSAGES,
        tools=None,
        stream=False,
        options=dict(
            temperature=None,
            num_predict=None,
        ),
    )


@pytest.mark.asyncio
@patch("think.llm.ollama.AsyncClient")
async def test_call_with_custom_parser(AsyncClient):
    chat = Chat.load(BASIC_CHAT)
    client = OllamaClient(base_url="http://localhost:11434/", model="fake-model")

    mock_chat = AsyncMock(return_value=response_fixture(BASIC_OLLAMA_MESSAGES[-1]))
    AsyncClient.return_value.chat = mock_chat

    def custom_parser(val: str) -> float:
        assert val == "Hi!"
        return 0.5

    response = await client(chat, parser=custom_parser)

    assert response == 0.5
    assert chat.messages[-1].parsed == response

    mock_chat.assert_called_once_with(
        model="fake-model",
        messages=BASIC_OLLAMA_MESSAGES,
        tools=None,
        stream=False,
        options=dict(
            temperature=None,
            num_predict=None,
        ),
    )

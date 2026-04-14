from copy import deepcopy
from unittest.mock import AsyncMock, patch

import pytest
from ollama import ResponseError

from tests.llm.test_chat import BASIC_CHAT, IMAGE_CHAT, SIMPLE_TOOL_CHAT
from think.llm.base import ConfigError
from think.llm.chat import Chat
from think.llm.ollama import OllamaAdapter, OllamaClient

OLLAMA_TOOL_CHAT = deepcopy(SIMPLE_TOOL_CHAT)
OLLAMA_TOOL_CHAT[2]["content"][0]["tool_call"]["id"] = "ask_user"  # type: ignore[invalid-argument-type, invalid-assignment]
OLLAMA_TOOL_CHAT[3]["content"][0]["tool_response"]["call"]["id"] = ""  # type: ignore[invalid-argument-type, invalid-assignment]

BASIC_OLLAMA_MESSAGES = [
    {"role": "system", "content": "You're a friendly assistant"},
    {"role": "user", "content": "Say Hi."},
    {"role": "assistant", "content": "Hi!"},
]

SIMPLE_TOOL_OLLAMA_MESSAGES = [
    {"role": "system", "content": "You're a friendly assistant"},
    {
        "role": "user",
        "content": "Ask the user to give you a math question, then solve it yourself.",
    },
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "ask_user",
                    "arguments": {
                        "question": "Please give me a math question to solve?",
                    },
                },
            }
        ],
    },
    {
        "role": "tool",
        "content": "1 + 1",
    },
    {
        "role": "assistant",
        "content": "The solution to the math question \\(1 + 1\\) is \\(2\\).",
    },
]

IMAGE_OLLAMA_MESSAGES = [
    {
        "role": "user",
        "content": "Describe the image in detail",
        "images": [
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQAAAAA3bvkkAAAACklEQVR4AWNgAAAAAgABc3UBGAAAAABJRU5ErkJggg==",
        ],
    },
    {
        "role": "assistant",
        "content": "The image appears to be a simple black silhouette of a cat.",
    },
]


@pytest.mark.parametrize(
    "chat,expected",
    [
        (BASIC_CHAT, BASIC_OLLAMA_MESSAGES),
        (OLLAMA_TOOL_CHAT, SIMPLE_TOOL_OLLAMA_MESSAGES),
        (IMAGE_CHAT, IMAGE_OLLAMA_MESSAGES),
    ],
)
def test_adapter(chat, expected):
    adapter = OllamaAdapter()

    chat = Chat.load(chat)
    system, messages = adapter.dump_chat(chat)
    assert system == ""
    assert messages == expected

    chat2 = adapter.load_chat(messages)
    assert chat.messages == chat2.messages


@pytest.mark.asyncio
async def test_retries_on_transient_response_error():
    """Ollama 5xx responses should be retried via the LLM._retry wrapper."""
    success_response = {
        "message": {"role": "assistant", "content": "Hi!"},
    }
    chat_mock = AsyncMock(
        side_effect=[
            ResponseError("server overloaded", 503),
            ResponseError("server overloaded", 503),
            success_response,
        ]
    )

    with patch("think.llm.ollama.AsyncClient") as MockClient:
        MockClient.return_value.chat = chat_mock
        # Patch sleep so the test runs instantly
        with patch("think.llm.base.asyncio.sleep", new=AsyncMock()):
            client = OllamaClient(model="llama3", base_url="http://localhost:11434")
            chat = Chat("system").user("hi")
            result = await client(chat, max_retries=3)

    assert result == "Hi!"
    assert chat_mock.call_count == 3


@pytest.mark.asyncio
async def test_no_retry_on_404():
    """A 404 (model not found) is non-transient and should not be retried."""
    chat_mock = AsyncMock(side_effect=ResponseError("model not found", 404))

    with patch("think.llm.ollama.AsyncClient") as MockClient:
        MockClient.return_value.chat = chat_mock
        with patch("think.llm.base.asyncio.sleep", new=AsyncMock()):
            client = OllamaClient(model="llama3", base_url="http://localhost:11434")
            chat = Chat("system").user("hi")
            with pytest.raises(ConfigError):
                await client(chat, max_retries=5)

    assert chat_mock.call_count == 1

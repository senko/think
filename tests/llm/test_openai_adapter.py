from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.llm.test_chat import (
    BASIC_CHAT,
    IMAGE_CHAT,
    SIMPLE_TOOL_CHAT,
    DOCUMENT_CHAT,
    PDF_URI,
)
from think.llm.chat import Chat
from think.llm.openai import OpenAIAdapter, OpenAIClient

BASIC_OPENAI_MESSAGES = [
    {"role": "system", "content": "You're a friendly assistant"},
    {"role": "user", "content": "Say Hi."},
    {"role": "assistant", "content": "Hi!", "tool_calls": None},
]

SIMPLE_TOOL_OPENAI_MESSAGES = [
    {"role": "system", "content": "You're a friendly assistant"},
    {
        "role": "user",
        "content": "Ask the user to give you a math question, then solve it yourself.",
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_boPTOC8z660AYdRFr9oogH4O",
                "type": "function",
                "function": {
                    "name": "ask_user",
                    "arguments": '{"question": "Please give me a math question to solve?"}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_boPTOC8z660AYdRFr9oogH4O",
        "content": "1 + 1",
    },
    {
        "role": "assistant",
        "content": "The solution to the math question \\(1 + 1\\) is \\(2\\).",
        "tool_calls": None,
    },
]

IMAGE_OPENAI_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the image in detail"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQAAAAA3bvkkAAAACklEQVR4AWNgAAAAAgABc3UBGAAAAABJRU5ErkJggg=="
                },
            },
        ],
    },
    {
        "role": "assistant",
        "content": "The image appears to be a simple black silhouette of a cat.",
        "tool_calls": None,
    },
]

DOCUMENT_OPENAI_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the document in detail"},
            {
                "type": "input_file",
                "file_name": "document.pdf",
                "file_data": PDF_URI.split(",", 1)[1],
            },
        ],
    },
    {
        "role": "assistant",
        "content": "The document is one page long and contains text HELLO WORLD.",
        "tool_calls": None,
    },
]


@pytest.mark.parametrize(
    "chat,expected",
    [
        (BASIC_CHAT, BASIC_OPENAI_MESSAGES),
        (SIMPLE_TOOL_CHAT, SIMPLE_TOOL_OPENAI_MESSAGES),
        (IMAGE_CHAT, IMAGE_OPENAI_MESSAGES),
        (DOCUMENT_CHAT, DOCUMENT_OPENAI_MESSAGES),
    ],
)
def test_adapter(chat, expected):
    adapter = OpenAIAdapter()

    chat = Chat.load(chat)
    system, messages = adapter.dump_chat(chat)
    assert system == ""
    assert messages == expected

    chat2 = adapter.load_chat(messages)
    assert chat.messages == chat2.messages


@pytest.mark.asyncio
async def test_max_retries_forwarded_via_with_options():
    """OpenAIClient should call client.with_options(max_retries=N) on each call."""
    fake_message = MagicMock()
    fake_message.model_dump.return_value = {
        "role": "assistant",
        "content": "Hi!",
        "tool_calls": None,
    }
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]

    scoped_client = MagicMock()
    scoped_client.chat.completions.create = AsyncMock(return_value=fake_response)

    with patch("think.llm.openai.AsyncOpenAI") as MockSDK:
        sdk_instance = MockSDK.return_value
        sdk_instance.with_options = MagicMock(return_value=scoped_client)

        client = OpenAIClient(model="gpt-4o", api_key="fake")
        chat = Chat("system").user("hi")
        await client(chat, max_retries=5)

        sdk_instance.with_options.assert_called_with(max_retries=5)
        scoped_client.chat.completions.create.assert_called_once()

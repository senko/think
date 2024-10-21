from typing import cast

import pytest
from openai._utils._transform import transform
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from think.llm.chat import Chat, Message
from think.llm.openai import OpenAIMessageAdapter

BASIC_CHAT = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You're a friendly assistant"}],
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "Say Hi."}],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "Hi!"}],
    },
]

BASIC_OPENAI_MESSAGES = [
    {"role": "system", "content": "You're a friendly assistant"},
    {"role": "user", "content": "Say Hi."},
    {"role": "assistant", "content": "Hi!", "tool_calls": []},
]

SIMPLE_TOOL_CHAT = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You're a friendly assistant"}],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Ask the user to give you a math question, then solve it yourself.",
            }
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call": {
                    "id": "call_boPTOC8z660AYdRFr9oogH4O",
                    "name": "ask_user",
                    "arguments": {
                        "question": "Please give me a math question to solve?"
                    },
                },
            }
        ],
    },
    {
        "role": "tool",
        "content": [
            {
                "type": "tool_response",
                "tool_response": {
                    "call": {
                        "id": "call_boPTOC8z660AYdRFr9oogH4O",
                        "name": "",
                        "arguments": {},
                    },
                    "response": "1 + 1",
                },
            }
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "The solution to the math question \\(1 + 1\\) is \\(2\\).",
            }
        ],
    },
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
        "tool_calls": [],
    },
]


@pytest.mark.parametrize(
    "chat",
    [
        BASIC_CHAT,
        SIMPLE_TOOL_CHAT,
    ],
)
def test_parse_dump_chat(chat):
    assert Chat.load(chat).dump() == chat


@pytest.mark.parametrize(
    "chat,expected",
    [
        (BASIC_CHAT, BASIC_OPENAI_MESSAGES),
        (SIMPLE_TOOL_CHAT, SIMPLE_TOOL_OPENAI_MESSAGES),
    ],
)
def test_dump(chat, expected):
    adapter = OpenAIMessageAdapter()

    chat = Chat.load(chat)
    messages = adapter.dump_chat(chat)
    print(messages)
    assert messages == expected

    chat2 = adapter.load_chat(messages)
    assert chat.messages == chat2.messages

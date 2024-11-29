from copy import deepcopy

import pytest

from tests.llm.test_chat import BASIC_CHAT, IMAGE_CHAT, SIMPLE_TOOL_CHAT
from think.llm.chat import Chat
from think.llm.ollama import OllamaAdapter

OLLAMA_TOOL_CHAT = deepcopy(SIMPLE_TOOL_CHAT)
OLLAMA_TOOL_CHAT[2]["content"][0]["tool_call"]["id"] = "ask_user"
OLLAMA_TOOL_CHAT[3]["content"][0]["tool_response"]["call"]["id"] = ""

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

import pytest
from anthropic import NOT_GIVEN

from tests.llm.test_chat import BASIC_CHAT, IMAGE_CHAT, SIMPLE_TOOL_CHAT
from think.llm.anthropic import AnthropicAdapter
from think.llm.chat import Chat

BASIC_ANTHROPIC_SYSTEM = "You're a friendly assistant"

BASIC_ANTHROPIC_MESSAGES = [
    {"role": "user", "content": "Say Hi."},
    {"role": "assistant", "content": "Hi!"},
]


SIMPLE_TOOL_ANTHROPIC_MESSAGES = [
    {
        "role": "user",
        "content": "Ask the user to give you a math question, then solve it yourself.",
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "call_boPTOC8z660AYdRFr9oogH4O",
                "name": "ask_user",
                "input": {"question": "Please give me a math question to solve?"},
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call_boPTOC8z660AYdRFr9oogH4O",
                "content": "1 + 1",
            }
        ],
    },
    {
        "role": "assistant",
        "content": "The solution to the math question \\(1 + 1\\) is \\(2\\).",
    },
]


IMAGE_ANTHROPIC_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the image in detail"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQAAAAA3bvkkAAAACklEQVR4AWNgAAAAAgABc3UBGAAAAABJRU5ErkJggg==",
                    "media_type": "image/png",
                },
            },
        ],
    },
    {
        "role": "assistant",
        "content": "The image appears to be a simple black silhouette of a cat.",
    },
]


@pytest.mark.parametrize(
    "chat,ex_system,expected",
    [
        (BASIC_CHAT, BASIC_ANTHROPIC_SYSTEM, BASIC_ANTHROPIC_MESSAGES),
        (SIMPLE_TOOL_CHAT, BASIC_ANTHROPIC_SYSTEM, SIMPLE_TOOL_ANTHROPIC_MESSAGES),
        (IMAGE_CHAT, NOT_GIVEN, IMAGE_ANTHROPIC_MESSAGES),
    ],
)
def test_adapter(chat, ex_system, expected):
    adapter = AnthropicAdapter(None)

    chat = Chat.load(chat)
    system, messages = adapter.dump_chat(chat)
    assert system == ex_system
    assert messages == expected

    chat2 = adapter.load_chat(messages, system=system)
    assert chat.messages == chat2.messages

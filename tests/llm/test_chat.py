import pytest

from think.llm.chat import Chat

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

IMAGE_CHAT = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the image in detail"},
            {
                "type": "image",
                "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQAAAAA3bvkkAAAACklEQVR4AWNgAAAAAgABc3UBGAAAAABJRU5ErkJggg==",
            },
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "The image appears to be a simple black silhouette of a cat.",
            }
        ],
    },
]


@pytest.mark.parametrize(
    "chat",
    [
        BASIC_CHAT,
        SIMPLE_TOOL_CHAT,
        IMAGE_CHAT,
    ],
)
def test_parse_dump_chat(chat):
    assert Chat.load(chat).dump() == chat

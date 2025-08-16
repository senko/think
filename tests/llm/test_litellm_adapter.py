import pytest

from tests.llm.test_chat import (
    BASIC_CHAT,
    IMAGE_CHAT,
    SIMPLE_TOOL_CHAT,
    DOCUMENT_CHAT,
    PDF_URI,
)
from think.llm.chat import Chat
from think.llm.litellm import LiteLLMAdapter

# LiteLLM uses OpenAI-compatible format, so we reuse the same expected messages
BASIC_LITELLM_MESSAGES = [
    {"role": "system", "content": "You're a friendly assistant"},
    {"role": "user", "content": "Say Hi."},
    {"role": "assistant", "content": "Hi!", "tool_calls": None},
]

SIMPLE_TOOL_LITELLM_MESSAGES = [
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

IMAGE_LITELLM_MESSAGES = [
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

DOCUMENT_LITELLM_MESSAGES = [
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
        (BASIC_CHAT, BASIC_LITELLM_MESSAGES),
        (SIMPLE_TOOL_CHAT, SIMPLE_TOOL_LITELLM_MESSAGES),
        (IMAGE_CHAT, IMAGE_LITELLM_MESSAGES),
        (DOCUMENT_CHAT, DOCUMENT_LITELLM_MESSAGES),
    ],
)
def test_adapter(chat, expected):
    """Test that LiteLLMAdapter correctly converts think chats to OpenAI format."""
    adapter = LiteLLMAdapter()

    chat = Chat.load(chat)
    system, messages = adapter.dump_chat(chat)
    assert system == ""
    assert messages == expected

    chat2 = adapter.load_chat(messages)
    assert chat.messages == chat2.messages


def test_tool_spec():
    """Test that LiteLLMAdapter correctly converts tool definitions."""
    from think.llm.tool import ToolDefinition

    def test_function(param: str) -> str:
        """A test tool for testing.

        :param param: A test parameter
        :return: A test result
        """
        return f"test: {param}"

    adapter = LiteLLMAdapter()
    tool = ToolDefinition(test_function, name="test_tool")

    spec = adapter.get_tool_spec(tool)

    expected = {
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": tool.description,
            "parameters": tool.schema,
        },
    }

    assert spec == expected


def test_error_handling():
    """Test that LiteLLMAdapter handles unsupported features appropriately."""
    from think.llm.chat import ContentPart, ContentType, Message, Role

    adapter = LiteLLMAdapter()

    # Test document URL error (not supported)
    message = Message(
        role=Role.user,
        content=[
            ContentPart(
                type=ContentType.document,
                document="https://example.com/doc.pdf",
            )
        ],
    )

    with pytest.raises(ValueError, match="does not support document URLs"):
        adapter.dump_message(message)

    # Test unsupported document MIME type
    message = Message(
        role=Role.user,
        content=[
            ContentPart(
                type=ContentType.document,
                document="data:application/msword;base64,dGVzdA==",
            )
        ],
    )

    with pytest.raises(ValueError, match="Unsupported document MIME type"):
        adapter.dump_message(message)

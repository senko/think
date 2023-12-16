from unittest.mock import MagicMock, patch, call
import pytest

from think.llm.openai import ChatGPT, ToolError
from think.chat import Chat


def test_chatgpt_valid_api_key_and_default_values():
    chat_gpt = ChatGPT("key")
    assert chat_gpt.api_key == "key"
    assert chat_gpt.model == "gpt-4-1106-preview"
    assert chat_gpt.temperature == 0.7


@patch("think.llm.openai.getenv")
def test_chatgpt_empty_api_key_looks_up_environment(mock_getenv):
    mock_getenv.return_value = "key"
    chat_gpt = ChatGPT()
    assert chat_gpt.api_key == "key"


def test_chatgpt_empty_key_no_env_raises_error():
    with pytest.raises(ValueError, match="OpenAI API key is not set"):
        ChatGPT("")


def test_chatgpt_unsupported_model_raises_error():
    with pytest.raises(ValueError, match="Unsupported model"):
        ChatGPT("key", model="gpt-1")


def test_run_tool_with_unknown_tool():
    chatgpt = ChatGPT("key")
    function_call = MagicMock()
    function_call.name = "UnknownTool"
    tools = [MagicMock(__name__="KnownTool")]

    result = chatgpt._run_tool(function_call, tools=tools)

    assert result == "ERROR: Unknown tool: UnknownTool; available tools: KnownTool"


def test_run_tool_with_invalid_arguments():
    chatgpt = ChatGPT("key")
    tool = MagicMock(
        __name__="tool",
        _validate_arguments=MagicMock(side_effect=TypeError("Invalid arguments")),
    )
    function_call = MagicMock()
    function_call.name = "tool"

    result = chatgpt._run_tool(function_call=function_call, tools=[tool])

    assert result == "ERROR: Invalid arguments"


def test_run_tool_executes_with_valid_tool_and_arguments():
    chatgpt = ChatGPT(api_key="key")
    tool = MagicMock(
        __name__="tool",
        _validate_arguments=MagicMock(return_value={"arg1": "value1"}),
    )
    function_call = MagicMock(arguments={"arg1": "value1"})
    function_call.name = "tool"

    result = chatgpt._run_tool(function_call=function_call, tools=[tool])

    assert result == tool.return_value
    tool._validate_arguments.assert_called_once_with(function_call.arguments)
    tool.assert_called_once_with(arg1="value1")


def test_run_tool_catches_exception_from_tool_execution():
    chatgpt = ChatGPT(api_key="key")
    tool = MagicMock(__name__="tool")
    tool.side_effect = ValueError("some error")

    function_call = MagicMock()
    function_call.name = "tool"

    with pytest.raises(ToolError) as exc_info:
        chatgpt._run_tool(function_call=function_call, tools=[tool])

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert exc_info.value.__cause__.args == ("some error",)


def test_call_chatgpt_no_functions():
    chatgpt = ChatGPT(api_key="key")
    chatgpt.client = MagicMock()
    chatgpt.client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message="hello")],
        usage=MagicMock(total_tokens=10),
    )

    result = chatgpt._call_chatgpt(messages=[{"content": "Hello!"}])

    assert result == "hello"
    chatgpt.client.chat.completions.create.assert_called_once_with(
        model="gpt-4-1106-preview",
        temperature=0.7,
        messages=[{"content": "Hello!"}],
    )


def test_call_chatgpt_with_functions():
    chatgpt = ChatGPT(api_key="key")
    chatgpt.client = MagicMock()
    chatgpt.client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message="hello")],
        usage=MagicMock(total_tokens=10),
    )
    tool = MagicMock(
        __name__="tool",
    )

    result = chatgpt._call_chatgpt(
        messages=[{"content": "Hello!"}],
        tools=[tool],
    )

    assert result == "hello"
    chatgpt.client.chat.completions.create.assert_called_once_with(
        model="gpt-4-1106-preview",
        temperature=0.7,
        messages=[{"content": "Hello!"}],
        functions=[tool._get_json_schema.return_value],
        function_call="auto",
    )


def test_call_simple():
    chatgpt = ChatGPT(api_key="key")
    chat = Chat("hello")
    tool = MagicMock()

    with patch.object(chatgpt, "_call_chatgpt") as mock_call_chatgpt:
        mock_call_chatgpt.return_value = MagicMock(
            function_call=None,
            content="hello",
        )

        result = chatgpt(
            chat=chat,
            tools=[tool],
        )

        assert result == "hello"
        mock_call_chatgpt.assert_called_once_with(
            [{"role": "system", "content": "hello"}],
            [tool],
        )


def test_call_function_call():
    chatgpt = ChatGPT(api_key="key")
    chat = Chat("hello")
    tool = MagicMock(__name__="tool")
    function_call = MagicMock()
    function_call.name = "tool"

    with patch.object(chatgpt, "_call_chatgpt") as mock_call_chatgpt:
        with patch.object(chatgpt, "_run_tool") as mock_run_tool:
            mock_call_chatgpt.side_effect = [
                MagicMock(function_call=function_call),
                MagicMock(function_call=None, content="hello"),
            ]
            mock_run_tool.return_value = "hello from tool"
            result = chatgpt(
                chat=chat,
                tools=[tool],
            )

            assert result == "hello"

            mock_call_chatgpt.assert_has_calls(
                [
                    call(
                        [{"role": "system", "content": "hello"}],
                        [tool],
                    ),
                    call(
                        [
                            {"role": "system", "content": "hello"},
                            {"role": "assistant", "content": "Using tool 'tool'"},
                            {
                                "role": "function",
                                "content": "hello from tool",
                                "name": "tool",
                            },
                        ],
                        [tool],
                    ),
                ]
            )
            mock_run_tool.assert_called_once_with(
                function_call,
                [tool],
            )


def test_call_propagates_function_call_error():
    chatgpt = ChatGPT(api_key="key")
    chat = Chat("hello")
    tool = MagicMock(__name__="tool")
    function_call = MagicMock()
    function_call.name = "tool"

    with patch.object(chatgpt, "_call_chatgpt") as mock_call_chatgpt:
        with patch.object(chatgpt, "_run_tool") as mock_run_tool:
            mock_call_chatgpt.side_effect = [
                MagicMock(function_call=function_call),
                MagicMock(function_call=None, content="hello"),
            ]
            try:
                raise ValueError("some error")
            except ValueError as err:
                original_error = err
            tool_error = ToolError("tool error")
            tool_error.__cause__ = original_error
            mock_run_tool.side_effect = tool_error

            with pytest.raises(ValueError, match="some error"):
                chatgpt(chat=chat, tools=[tool])

            mock_call_chatgpt.assert_called_once_with(
                [{"role": "system", "content": "hello"}],
                [tool],
            )
            mock_run_tool.assert_called_once_with(
                function_call,
                [tool],
            )


def test_call_with_parser():
    chatgpt = ChatGPT(api_key="key")
    chat = Chat("hello")
    parser = MagicMock(return_value="HELLO")

    with patch.object(chatgpt, "_call_chatgpt") as mock_call_chatgpt:
        mock_call_chatgpt.return_value = MagicMock(
            function_call=None,
            content="hello",
        )

        result = chatgpt(
            chat=chat,
            parser=parser,
        )

        assert result == "HELLO"
        parser.assert_called_once_with("hello")


def test_call_retry_on_parse_error():
    chatgpt = ChatGPT(api_key="key")
    chat = Chat("hello")

    parser_calls = []

    def parser(content: str) -> str:
        parser_calls.append(call(content))

        if content == "hello":
            raise ValueError("test error")
        return content.upper()

    with patch.object(chatgpt, "_call_chatgpt") as mock_call_chatgpt:
        mock_call_chatgpt.side_effect = [
            MagicMock(
                function_call=None,
                content="hello",
            ),
            MagicMock(
                function_call=None,
                content="world",
            ),
        ]

        result = chatgpt(
            chat=chat,
            parser=parser,
        )

        assert result == "WORLD"
        mock_call_chatgpt.assert_has_calls(
            [
                call(
                    [{"role": "system", "content": "hello"}],
                    None,
                ),
                call(
                    [
                        {"role": "system", "content": "hello"},
                        {"role": "assistant", "content": "hello"},
                        {
                            "role": "user",
                            "content": (
                                "Error parsing response: test error. "
                                "Please output your response EXACTLY as requested."
                            ),
                        },
                    ],
                    None,
                ),
            ]
        )
        assert parser_calls == [
            call("hello"),
            call("world"),
        ]
        assert len(list(chat)) == 1

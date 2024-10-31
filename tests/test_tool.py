import pytest

from think.llm.tool import ToolCall, ToolKit


def add_numbers(a: int, b: int) -> str:
    """
    Add two numbers

    :param a: The first number
    :param b: The second number
    :return: The sum of the two numbers, as a string
    """
    return str(a + b)


def test_tool_definitions():
    tk = ToolKit([add_numbers])

    assert tk.tool_names == ["add_numbers"]

    tool_def = tk.tools["add_numbers"]
    assert tool_def.name == "add_numbers"
    assert "Add two numbers" in tool_def.description
    assert set(tool_def.schema["required"]) == {"a", "b"}


@pytest.mark.asyncio
async def test_tool_call():
    tk = ToolKit([add_numbers])

    call = ToolCall(id="123", name="add_numbers", arguments={"a": 1, "b": 2})
    result = await tk.execute_tool_call(call)

    assert result.response == "3"

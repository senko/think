import json

import pytest
from pydantic import BaseModel


from think.tool import (
    create_pydantic_model_from_function,
    parse_docstring,
    tool,
)


def test_parse_docstring_with_none():
    result = parse_docstring(None)
    assert result == ("", {})


def test_parse_docstring_with_empty_string():
    result = parse_docstring("")
    assert result == ("", {})


def test_parse_docstring_with_only_description():
    docstring = """
    Description line 1.
    Description line 2.
    """
    result = parse_docstring(docstring)
    expected_description = "Description line 1. Description line 2."
    assert result == (expected_description, {})


def test_parse_docstring_with_params():
    docstring = """
    :param a: Description of a
    :param b: Description of b
    """
    result = parse_docstring(docstring)
    expected_args = {"a": "Description of a", "b": "Description of b"}
    assert result == ("", expected_args)


def test_parse_docstring_with_returns():
    docstring = """
    :returns: Description of return
    """
    result = parse_docstring(docstring)
    expected_description = "Returns Description of return"
    assert result == (expected_description, {})


def test_parse_docstring_full():
    docstring = """
    Function description.
    :param a: Description of a
    :param b: Description of b
    :returns: Description of return
    """
    result = parse_docstring(docstring)
    expected_description = "Function description.\nReturns Description of return"
    expected_args = {"a": "Description of a", "b": "Description of b"}
    assert result == (expected_description, expected_args)


def test_create_pydantic_model_from_function_no_args():
    def test_func():
        """Test function without arguments"""
        pass

    model = create_pydantic_model_from_function(test_func)
    assert issubclass(model, BaseModel)
    assert model.__doc__ == "Test function without arguments"


def test_create_pydantic_model_from_function_without_type_annotation_fails():
    def test_func(a, b=2):
        """
        Test function with arguments that are not type annotated
        :param a: Argument a
        :param b: Argument b with default value 2
        """
        return a + b

    with pytest.raises(TypeError):
        create_pydantic_model_from_function(test_func)


def test_create_pydantic_model_from_function_with_args():
    def test_func(a: int, b: int = 2):
        """
        Test function with arguments
        :param a: Argument a
        :param b: Argument b with default value 2
        """
        return a + b

    model = create_pydantic_model_from_function(test_func)
    assert issubclass(model, BaseModel)
    assert model.__doc__ == "Test function with arguments"


def test_create_pydantic_model_from_method_with_args():
    class Foo:
        def test_func(self, a: int, b: int = 2):
            """
            Test Foo method with arguments
            :param a: Argument a
            :param b: Argument b with default value 2
            """
            return a + b

    model = create_pydantic_model_from_function(Foo.test_func)
    assert issubclass(model, BaseModel)
    assert model.__doc__ == "Test Foo method with arguments"


def test_create_pydantic_model_from_function_docstring_format_fails():
    def test_func(a: int, b: int = 2):
        """Test function with incorrect docstring format."""
        return a + b

    model = create_pydantic_model_from_function(test_func)
    assert issubclass(model, BaseModel)
    assert model.__doc__ == "Test function with incorrect docstring format."


def test_create_pydantic_model_from_function_no_docstring():
    def test_func(a: int, b: int = 2):
        return a + b

    model = create_pydantic_model_from_function(test_func)
    assert issubclass(model, BaseModel)
    assert model.__doc__ == ""


def test_tool_decorator_valid_args():
    @tool
    def my_func(a: int, b: int):
        """Some function
        :param a: First number
        :param b: Second number
        :returns: Sum of a and b
        """
        return a + b

    result = my_func._validate_arguments(json.dumps({"a": 1, "b": 2}))
    assert result == {"a": 1, "b": 2}


def test_tool_decorator_invalid_args():
    @tool
    def my_func(a: int, b: int):
        """Some function
        :param a: First number
        :param b: Second number
        :returns: Sum of a and b
        """
        return a + b

    with pytest.raises(TypeError):
        my_func._validate_arguments(json.dumps({"a": "one", "b": 2}))


def test_tool_decorator_invalid_json():
    @tool
    def my_func(a: int, b: int):
        """Some function
        :param a: First number
        :param b: Second number
        :returns: Sum of a and b
        """
        return a + b

    with pytest.raises(TypeError):
        my_func._validate_arguments("not a json")


def test_tool_decorator_json_schema():
    @tool
    def my_func(a: int, b: int):
        """Some function
        :param a: First number
        :param b: Second number
        :returns: Sum of a and b
        """
        return a + b

    schema = my_func._get_json_schema()
    assert schema["name"] == "my_func"
    assert schema["description"] == "Some function\nReturns Sum of a and b"
    assert schema["parameters"]["properties"]["a"]["description"] == "First number"
    assert schema["parameters"]["properties"]["b"]["description"] == "Second number"


def test_fn():
    """Test function"""
    pass


@tool
def tool_fn():
    """Tool test function"""
    pass


def test_validate_accepts_valid_json():
    try:
        tool_fn._validate_arguments("{}")
    except Exception:
        pytest.fail("_validate_arguments failed with valid JSON input")


def test_validate_rejects_invalid_json():
    with pytest.raises(TypeError):
        tool_fn._validate_arguments("{'invalid': 'json'}")


def test_validate_rejects_non_json_string():
    with pytest.raises(TypeError):
        tool_fn._validate_arguments("not a json string")


def test_validate_rejects_non_string_input():
    with pytest.raises(TypeError):
        tool_fn._validate_arguments(123)


def test_validate_rejects_invalid_arguments():
    with pytest.raises(TypeError):
        tool_fn._validate_arguments(json.dumps({"invalid_arg": "value"}))


def test_get_schema_with_simple_func():
    @tool
    def my_func(name: str):
        """This is my function
        :param name: The name
        """
        return f"Hello, {name}!"

    schema = my_func._get_json_schema()
    assert schema["name"] == "my_func"
    assert schema["description"] == "This is my function"
    assert schema["parameters"]["properties"]["name"]["type"] == "string"


def test_get_schema_with_defaults():
    @tool
    def my_func(name: str = "World"):
        """This is my function
        :param name: The name
        """
        return f"Hello, {name}!"

    schema = my_func._get_json_schema()
    assert schema["parameters"]["properties"]["name"]["default"] == "World"


def test_get_schema_with_no_docstring():
    @tool
    def my_func(name: str = "World"):
        return f"Hello, {name}!"

    schema = my_func._get_json_schema()
    assert schema["name"] == "my_func"
    assert schema["description"] == ""
    assert schema["parameters"]["properties"]["name"]["type"] == "string"


@tool("fancy_name")
def tool_with_fancy_name():
    """Tool test function with fancy name"""
    pass


def test_tool_get_schema_with_fancy_name():
    schema = tool_with_fancy_name._get_json_schema()
    assert schema["name"] == "fancy_name"
    assert schema["description"] == "Tool test function with fancy name"

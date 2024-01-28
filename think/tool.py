from pydantic import Field, create_model, ValidationError
from typing import Callable, Union
import re
import inspect
import json


def parse_docstring(docstring: str) -> tuple[str, dict[str, str]]:
    """
    Parse function arguments from reST docstring

    :param docstring: The text to parse in reST format
    :returns: The description and the dict of arguments and their descriptions
    """

    LINE_MATCH = re.compile(r"^:((param (\w+))|(returns)): (.+)$")

    desc = []
    args = {}
    returns = None

    if docstring is None:
        return "", {}

    for line in docstring.split("\n"):
        if not line:
            continue

        line = line.strip()
        m = LINE_MATCH.match(line)
        if m:
            arg_name = m.group(3)
            description = m.group(5).strip()
            if arg_name:
                args[arg_name] = description
            else:
                returns = description
        elif line:
            desc.append(line)

    desc = " ".join(desc)
    if returns:
        if desc:
            desc += "\n"
        desc += f"Returns {returns}"
    return desc, args


def create_pydantic_model_from_function(fn: Callable) -> type:
    """
    Inspect a function and create a Pydantic model from its arguments,
    return type and docstring.

    The model will be named after the function, have the docstring
    as its description and the function arguments as its fields.
    The function arguments must have type annotations and the docstring
    must contain reST formatted descriptions of the arguments.

    :param fn: The function to inspect
    :returns: The newly created Pydantic model
    """
    fields = {}

    desc, param_docs = parse_docstring(fn.__doc__)

    signature = inspect.signature(fn)
    for param_name, param in signature.parameters.items():
        if param_name == "self":
            continue

        param_doc = param_docs.get(param_name, "")
        param_default = param.default

        if param_default is not inspect._empty:
            field_def = Field(param_default, description=param_doc)
        else:
            field_def = Field(..., description=param_doc)

        if param.annotation is inspect._empty:
            raise TypeError(
                f"Parameter {param_name} of {fn.__name__} has no type annotation"
            )
        fields[param_name] = (param.annotation, field_def)

    model = create_model(
        fn.__name__,
        **fields,
        __cls_kwargs__={"extra": "forbid"},
    )
    model.__doc__ = desc
    return model


def tool(fn: Union[Callable, str]) -> Callable:
    """
    Decorator to mark a function as a tool

    Usage:

    >>> @tool
    >>> def my_tool(a: int, b: int = 2) -> int:
    >>>     '''
    >>>     My tool description
    >>>
    >>>     :param a: First argument
    >>>     :param b: Second argument
    >>>     :returns: Something
    >>>     '''
    >>>     return a + b

    The tool can also be named explicitly:

    >>> @tool("add_two_numbers")
    >>> def my_tool(a: int, b: int = 2) -> int:
    >>>     '''
    >>>     :param a: First argument
    >>>     :param b: Second argument
    >>>     :returns: Something
    >>>     '''
    >>>     return a + b
    """

    def decorator(fn: Callable, tool_name: str) -> Callable:
        model = create_pydantic_model_from_function(fn)

        def validate(json_args):
            try:
                kwargs = json.loads(json_args)
            except json.JSONDecodeError as err:
                raise TypeError(f"Arguments are not valid JSON: {err}")

            try:
                return model(**kwargs).model_dump()
            except ValidationError as err:
                raise TypeError(f"Invalid arguments: {err.errors()}")

        def get_schema():
            schema = model.model_json_schema()
            name = fn.__name__
            desc = schema.pop("description", "")
            return {
                "name": name,
                "description": desc,
                "parameters": schema,
            }

        fn._validate_arguments = validate
        fn._get_json_schema = get_schema
        fn.__name__ = tool_name

        schema = get_schema()
        fn._oneline_description = f"{schema['name']} - {schema['description']}"
        return fn

    if callable(fn):
        return decorator(fn, fn.__name__)
    else:
        return lambda func: decorator(func, fn)

import asyncio
import re
from inspect import Parameter, getdoc, signature
from typing import Any

from pydantic import Field, create_model, ConfigDict

def parse_docstring(docstring: str):
    """
    Parse the Sphinx-style docstring and extract parameter descriptions.
    """
    param_pattern = r":param (\w+): (.+)"
    param_descriptions = {}
    if docstring:
        matches = re.findall(param_pattern, docstring)
        for param, description in matches:
            param_descriptions[param] = description.strip()
    return param_descriptions


def clean_docstring(docstring: str):
    """
    Remove :param and :return lines from the docstring to use the rest
    as the Pydnatic model class docstring.
    """
    if not docstring:
        return None

    # Remove all lines starting with :param or :return
    cleaned_docstring = re.sub(r":param.*|:return.*", "", docstring)

    # Strip excessive newlines or whitespace
    return cleaned_docstring.strip()


def create_model_from_function(func):
    sig = signature(func)
    docstring = getdoc(func)
    param_descriptions = parse_docstring(docstring)
    fields = {}

    for name, param in sig.parameters.items():
        annotation = param.annotation if param.annotation != Parameter.empty else Any
        default = param.default if param.default != Parameter.empty else ...
        desc = param_descriptions.get(name)
        if desc:
            fields[name] = (annotation, Field(default, description=desc))
        else:
            fields[name] = (annotation, default)

    model_name = (
        "".join(part.capitalize() for part in func.__name__.split("_")) + "Args"
    )

    model = create_model(model_name, __doc__=clean_docstring(docstring), __config__=ConfigDict(extra="forbid"), **fields,)

    return model


def get_tool_description(func):
    schema = create_model_from_function(func).model_json_schema()
    schema.pop("title", None)
    desc = schema.pop("description", None)
    return {
        "name": func.__name__,
        "description": desc,
        "parameters": schema,
        "strict": True,
    }


def is_async():
    try:
        loop = asyncio.get_running_loop()
        return loop.is_running() if loop else False
    except RuntimeError:
        return False

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from inspect import Parameter, getdoc, signature
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model


class ToolDefinition:
    """
    A tool available to the LLM.
    """

    name: str
    func: Callable
    model: type[BaseModel]
    args_schema: dict[str, Any]

    def __init__(self, func: Callable, name: str | None = None):
        """
        Define a new tool that runs a function.

        The function should have a Sphinx-style docstring with :param: and
        :return: lines to describe the parameters and return value. The
        docstring should describe the function in detail so that the LLM
        can decide when to use it and to know how.

        :param func: The function to run.
        :param name: The name of the tool, exposed to the LLM. Defaults to the
            function name.
        """
        self.name = name or func.__name__
        self.func = func
        self.model = self.create_model_from_function(func)

        self.schema = self.model.model_json_schema()
        self.schema.pop("title", None)
        self.description = self.schema.pop("description", None)

    @staticmethod
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

    @classmethod
    def create_model_from_function(cls, func):
        sig = signature(func)
        docstring = getdoc(func)
        param_descriptions = cls.parse_docstring(docstring)
        fields = {}

        for name, param in sig.parameters.items():
            annotation = (
                param.annotation if param.annotation != Parameter.empty else Any
            )
            default = param.default if param.default != Parameter.empty else ...
            desc = param_descriptions.get(name)
            if desc:
                fields[name] = (annotation, Field(default, description=desc))
            else:
                fields[name] = (annotation, default)

        model_name = (
            "".join(part.capitalize() for part in func.__name__.split("_")) + "Args"
        )

        model = create_model(
            model_name,
            __doc__=docstring,
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )

        return model


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def from_openai(cls, tool_call: Any):
        from openai.types.chat import ParsedFunctionToolCall

        assert isinstance(tool_call, ParsedFunctionToolCall)

        return cls(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )

    @classmethod
    def from_anthropic(cls, block: Any):
        # assert isinstance(block, ToolUseBlock)

        return cls(
            id=block.id,
            name=block.name,
            arguments=json.dumps(block.input),
        )

    @property
    def json_message(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


@dataclass
class ToolResponse:
    call: ToolCall
    response: str = None
    error: str = None
    # exception also!


class ToolError(Exception):
    pass


class ToolKit:
    tools: dict[str, ToolDefinition]

    def __init__(self, functions: list[Callable]):
        tool_defs = [ToolDefinition(func) for func in functions]
        self.tools = {t.name: t for t in tool_defs}

    @property
    def tool_names(self) -> list[str]:
        return list(self.tools.keys())

    async def execute_tool_call(self, call: ToolCall) -> ToolResponse:
        tool = self.tools.get(call.name)
        if not tool:
            return ToolResponse(call=call, error=f"ERROR: Unknown tool: {call.name}")

        try:
            args = tool.model(**call.arguments)
        except (TypeError, ValidationError) as err:
            return ToolResponse(
                call=call,
                error=f"ERROR: Error parsing arguments for {call.name}: {err}",
            )

        try:
            response_text: str = tool.func(**args.__dict__)
            return ToolResponse(call=call, response=response_text)
        except ToolError as err:
            return ToolResponse(
                call=call, error=f"ERROR: Error running tool {call.name}: {err}"
            )

    def generate_tool_spec(
        self,
        formatter: Callable[[list[ToolDefinition]], dict],
    ) -> list[dict]:
        return [formatter(t) for t in self.tools.values()]

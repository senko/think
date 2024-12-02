from __future__ import annotations

import re
from dataclasses import dataclass
from inspect import Parameter, getdoc, isawaitable, signature
from logging import getLogger
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

log = getLogger(__name__)


class ToolDefinition:
    """
    A tool available to the LLM.

    A tool is a function that can be called by the LLM to perform some
    operation. The tool definition includes the function itself, a name
    for the tool, a Pydantic model for the function's arguments, and a
    description of the tool.
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
    def parse_docstring(docstring: str) -> dict[str, str]:
        """
        Parse the Sphinx-style docstring and extract parameter descriptions.

        :param docstring: The docstring to parse.
        :return: A dictionary mapping parameter names to descriptions.
        """
        param_pattern = r":param (\w+): (.+)"
        param_descriptions = {}
        if docstring:
            matches = re.findall(param_pattern, docstring)
            for param, description in matches:
                param_descriptions[param] = description.strip()
        return param_descriptions

    @classmethod
    def create_model_from_function(cls, func: Callable) -> type[BaseModel]:
        """
        Creates a Pydantic model for agiven function.

        This method extracts the function's signature and docstring,
        parses the docstring for parameter descriptions, and constructs
        a Pydantic model with fields corresponding to the function's
        parameters.

        :param func: The function from which to create the model.
        :return: A Pydantic model class with fields derived from the
            function's parameters and their annotations.
        """
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
    """
    A call to a tool.

    Parsed assistant/AI tool call.
    Contains the tool's ID, name, and arguments.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResponse:
    """
    A response from a tool call.

    Contains the reference to the tool call, the response string from
    the tool or an error message if the tool call failed.
    """

    call: ToolCall
    response: str = None
    error: str = None


class ToolError(Exception):
    """
    Tool error that should be passed back to the LLM.

    This exception should be raised by tools when the cause of
    the error is on the LLM's side. The LLM will be prompted
    to fix their invocation/arguments and try again.
    """

    pass


class ToolKit:
    """
    A collection of tools available to the LLM.

    The toolkit is a collection of functions that the LLM can use to
    perform various operations.

    Both synchronous and asynchronous functions are supported. Async
    functions will be automatically awaited.
    """

    tools: dict[str, ToolDefinition]

    def __init__(self, functions: list[Callable]):
        """
        Initialize the toolkit with a list of functions.

        Each function will be introspected to create a tool definition
        to be used by LLM to decide which tool to use (if any).

        The function should have type annotation for arguments and
        return value, and a Sphinx-style docstring with :param: and
        :return: lines to describe the parameters and return value.

        See `ToolDefinition` for more information on the tool definition.

        :param functions: A list of functions to add to the toolkit.
        """
        tool_defs = [ToolDefinition(func) for func in functions]
        self.tools = {t.name: t for t in tool_defs}

    @property
    def tool_names(self) -> list[str]:
        """Return a list of tool names."""
        return list(self.tools.keys())

    async def execute_tool_call(self, call: ToolCall) -> ToolResponse:
        """
        Execute a tool call, returning the response or an error message.

        :param call: The tool call containing the tool name and arguments.
        :return: The response from the tool execution.

        If the function is a coroutine, it will be awaited.

        If the tool call is successful, the response will contain the
        return value from the tool. If the tool call raises a `ToolError`,
        the response will contain the error message to pass to the LLM.

        If the tool raises any other exception, the exception will be
        propagated to the caller (user).
        """
        tool = self.tools.get(call.name)
        if not tool:
            log.debug(f"Tool call with an unknown tool: {call.name}")
            return ToolResponse(call=call, error=f"ERROR: Unknown tool: {call.name}")

        try:
            args = tool.model(**call.arguments)
        except (TypeError, ValidationError) as err:
            log.debug(
                f"Tool call for {call.name} with invalid arguments {call.arguments}: {err}",
                exc_info=True,
            )
            return ToolResponse(
                call=call,
                error=f"ERROR: Error parsing arguments for {call.name}: {err}",
            )

        log.debug(f"Tool call {call.name} requested with args {call.arguments}")

        try:
            response_text = tool.func(**args.__dict__)
            if isawaitable(response_text):
                response_text = await response_text
            return ToolResponse(call=call, response=response_text)
        except ToolError as err:
            log.debug(f"Tool {call.name} raised an error: {err}", exc_info=True)
            return ToolResponse(
                call=call, error=f"ERROR: Error running tool {call.name}: {err}"
            )

    def generate_tool_spec(
        self,
        formatter: Callable[[list[ToolDefinition]], dict],
    ) -> list[dict]:
        """Generate tool specifications to pass to the LLM."""
        return [formatter(t) for t in self.tools.values()]

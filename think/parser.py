import json
import re
from enum import Enum
from typing import Optional, Union, Type, overload

from pydantic import BaseModel


class MultiCodeBlockParser:
    """
    Parse multiple Markdown code blocks from a string.

    Expects zero or more blocks, and ignores any text
    outside of the code blocks.

    Example usage:

    >>> parser = MultiCodeBlockParser()
    >>> text = '''
    ... text outside block
    ...
    ... ```python
    ... first block
    ... ```
    ... some text between blocks
    ... ```js
    ... more
    ... code
    ... ```
    ... some text after blocks
    '''
    >>> assert parser(text) == ["first block", "more\ncode"]

    If no code blocks are found, an empty list is returned:
    """

    def __init__(self):
        self.pattern = re.compile(r"```([a-z0-9]+\n)?(.*?)```\s*", re.DOTALL)

    def __call__(self, text: str) -> list[str]:
        blocks: list[str] = []
        for block in self.pattern.findall(text):
            blocks.append(block[1].strip())
        return blocks


class CodeBlockParser(MultiCodeBlockParser):
    """
    Parse a Markdown code block from a string.

    Expects exactly one code block, and ignores
    any text before or after it.

    Usage:
    >>> parser = CodeBlockParser()
    >>> text = "text\n```py\ncodeblock\n'''\nmore text"
    >>> assert parser(text) == "codeblock"

    This is a special case of MultiCodeBlockParser,
    checking that there's exactly one block.
    """

    def __call__(self, text: str) -> str:
        blocks = super().__call__(text)
        if len(blocks) != 1:
            raise ValueError(f"Expected a single code block, got {len(blocks)}")
        return blocks[0]


class JSONParser:
    """
    Parse a JSON string into a Python structure or Pydantic model

    If the model is provided, the JSON will be parsed
    and validated against the model. If the model is
    not provided, the JSON will be returned as a dict.

    If the JSON is not valid and strict is True (default),
    a ValueError is raised. If strict is False,
    None is returned instead.

    The JSON can be provided as a string or inside a
    Markdown code block.
    """

    def __init__(self, spec: Optional[Type[BaseModel]] = None, strict: bool = True):
        self.spec = spec
        self.strict = strict or (spec is not None)

    @property
    def schema(self):
        return self.spec.model_json_schema() if self.spec else None

    @overload
    def __call__(self, text: str) -> BaseModel: ...

    @overload
    def __call__(self, text: str) -> dict: ...

    @overload
    def __call__(self, text: str) -> None: ...

    def __call__(self, text: str) -> Union[BaseModel, dict, None]:
        text = text.strip()
        if text.startswith("```"):
            try:
                text = CodeBlockParser()(text)
            except ValueError:
                if self.strict:
                    raise
                else:
                    return None

        try:
            data = json.loads(text.strip())
        except json.JSONDecodeError as e:
            if self.strict:
                raise ValueError("Could not parse JSON") from e
            else:
                return None

        if self.spec is None:
            return data

        try:
            model = self.spec(**data)
        except Exception as err:
            raise ValueError(f"Error parsing JSON: {err}")

        return model


class EnumParser:
    """
    Parse text into one of possible Enum values.

    If ignore_case is True (default), the text is
    converted to lowercase before parsing.

    Raises a ValueError if the text does not match
    any of the Enum values.
    """

    def __init__(self, spec: Type[Enum], ignore_case: bool = True):
        self.spec = spec
        self.ignore_case = ignore_case

    def __call__(self, text: str) -> Enum:
        text = text.strip()
        if self.ignore_case:
            text = text.lower()
        try:
            return self.spec(text)
        except ValueError as e:
            options = ", ".join([str(v) for v in self.spec])
            raise ValueError(
                f"Invalid option '{text}'; valid options: {options}"
            ) from e

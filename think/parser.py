"""
# Parsing Functionality

The `parser` module provides tools for parsing structured data from LLM responses,
making it easier to extract and validate information from raw text outputs.

## JSON Parsing

The `JSONParser` is useful for extracting and validating JSON from LLM responses:

```python
# example: json_parser.py
from think import LLM
from think.llm.chat import Chat
from think.parser import JSONParser
import asyncio

llm = LLM.from_url("openai:///gpt-4o-mini")
parser = JSONParser()

async def get_structured_data():
    chat = Chat("List the top 3 programming languages as a JSON array")
    response = await llm(chat, parser=parser)
    print(type(response))  # <class 'list'>
    for lang in response:
        print(lang)

asyncio.run(get_structured_data())
```

## Code Block Parsing

For extracting code blocks from LLM responses:

```python
# example: code_block_parser.py
from think import LLM
from think.llm.chat import Chat
from think.parser import CodeBlockParser
import asyncio

llm = LLM.from_url("openai:///gpt-4o-mini")
parser = CodeBlockParser()

async def get_python_code():
    chat = Chat("Write a Python function to calculate the factorial of a number")
    code = await llm(chat, parser=parser)
    print(code)
    # Execute the code to verify it works
    exec(code)
    print(f"Factorial of 5: {factorial(5)}")  # Uses the function from the code

asyncio.run(get_python_code())
```

## Multiple Code Blocks

For working with multiple code blocks:

```python
# example: multi_code_blocks.py
from think import LLM
from think.llm.chat import Chat
from think.parser import MultiCodeBlockParser
import asyncio

llm = LLM.from_url("openai:///gpt-4o-mini")
parser = MultiCodeBlockParser()

async def get_multiple_languages():
    chat = Chat("Write a 'Hello World' program in Python, JavaScript, and Go")
    code_blocks = await llm(chat, parser=parser)

    for i, block in enumerate(code_blocks):
        print(f"Code block {i+1}:")
        print(block)
        print()

asyncio.run(get_multiple_languages())
```

## Pydantic Integration

For validating responses against Pydantic models:

```python
# example: pydantic_parser.py
from pydantic import BaseModel
from typing import List
from think import LLM
from think.llm.chat import Chat
import asyncio

class Movie(BaseModel):
    title: str
    director: str
    year: int
    rating: float

llm = LLM.from_url("openai:///gpt-4o-mini")

async def get_movie_data():
    chat = Chat('''Return information about the movie "The Matrix" in JSON format
    with fields: title, director, year, and rating.''')

    response = await llm(chat, parser=Movie)
    print(f"Title: {response.title}")
    print(f"Director: {response.director}")
    print(f"Year: {response.year}")
    print(f"Rating: {response.rating}")

asyncio.run(get_movie_data())
```

See also:
- [Structured Outputs and Parsing](#structured-outputs-and-parsing) for more examples
- [Basic LLM Use](#basic-llm-use) for integrating parsers with LLM calls
"""

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
        """
        Initialize the parser with regex pattern for code blocks.
        """
        self.pattern = re.compile(r"```([a-z0-9]+\n)?(.*?)```\s*", re.DOTALL)

    def __call__(self, text: str) -> list[str]:
        """
        Extract all code blocks from the given text.

        :param text: The text to parse for code blocks
        :return: List of code block contents (without language specifiers)
        """
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
        """
        Initialize the JSON parser.

        :param spec: Optional Pydantic model class for validation
        :param strict: Whether to raise errors on invalid JSON (default True)
        """
        self.spec = spec
        self.strict = strict or (spec is not None)

    @property
    def schema(self):
        """
        Get the JSON schema for the Pydantic model if one is specified.

        :return: JSON schema dict or None if no spec provided
        """
        return self.spec.model_json_schema() if self.spec else None

    @overload
    def __call__(self, text: str) -> BaseModel: ...

    @overload
    def __call__(self, text: str) -> dict: ...

    @overload
    def __call__(self, text: str) -> None: ...

    def __call__(self, text: str) -> Union[BaseModel, dict, None]:
        """
        Parse JSON text into a Python structure or Pydantic model.

        :param text: The text to parse (may contain JSON in code blocks)
        :return: Parsed data as dict, Pydantic model, or None (if not strict)
        :raises ValueError: If JSON is invalid and strict=True
        """
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
        """
        Initialize the enum parser.

        :param spec: The Enum class to parse values into
        :param ignore_case: Whether to ignore case when matching (default True)
        """
        self.spec = spec
        self.ignore_case = ignore_case

    def __call__(self, text: str) -> Enum:
        """
        Parse text into an enum value.

        :param text: The text to parse
        :return: The corresponding enum value
        :raises ValueError: If text doesn't match any enum value
        """
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

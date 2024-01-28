# Think

Think is a Python package for creating thinking programs.

It provides simple but powerful primitives for *robust* integration of Large Language
Models (LLMs) into your Python programs.

## Examples

Using AI as an ordinary function:

```python
from think.llm.openai import ChatGPT
from think.ai import ai

@ai
def haiku(topic: str) -> str:
    """
    Write a haiku about {{ topic }}
    """

llm = ChatGPT()

print(haiku(llm, topic="computers"))
```

Allowing AI to use tools:

```python
from datetime import date

from think.llm.openai import ChatGPT
from think.chat import Chat
from think.tool import tool

@tool
def current_date() -> str:
    """
    Get the current date.

    :returns: current date in YYYY-MM-DD format
    """
    return date.today().isoformat()


llm = ChatGPT()
chat = Chat("You are a helpful assistant.")
chat.user("How old are you (in days since your knowledge cutoff)?")

print(llm(chat, tools=[current_date]))
```

Parsing AI output:

```python
import json
from pydantic import BaseModel
from think.llm.openai import ChatGPT
from think.chat import Chat
from think.parser import JSONParser


class CityInfo(BaseModel):
    name: str
    country: str
    population: int
    latitude: float
    longitude: float


llm = ChatGPT()
parser = JSONParser(spec=CityInfo)
chat = Chat(
    "You are a hepful assistant. Your task is to answer questions about cities, "
    "to the best of your knowledge. Your output must be a valid JSON conforming to "
    "this JSON schema:\n" + json.dumps(parser.schema)
).user(city)

answer = llm(chat, parser=parser)

print(f"{answer.name} is a city in {answer.country} with {answer.population} inhabitants.")
print(f"It is located at {answer.latitude} latitude and {answer.longitude} longitude.")
```

## Quickstart

Install via `pip`:

```bash
pip install think-llm
```

Note that the package name is `think-llm`, *not* `think`.

Set up your LLM credentials (OpenAI or Anthropic, depending on the LLM you want to use):

```bash
export OPENAI_API_KEY=<your-openai-key>
export ANTHROPIC_API_KEY=<your-anthropic-key>
```

And you're ready to go:

```python
from think.llm.openai import ChatGPT
from think.chat import Chat

llm = ChatGPT()
chat = Chat("You are a helpful assistant.").user("Tell me a funny joke.")
print(llm(chat))
```

Explore the [examples](./examples/) directory for more usage examples, and the
source code for documentation on how to use the library (until we build proper docs! if you
want to help out with that, please see below).

## Roadmap

Features and capabilities that are planned for the near future:

- documentation
- full support for Anthropic (tools, parsers, AI functions)
- support for other LLM APIs via LiteLLM or similar
- support for local LLMs via HuggingFace
- more examples

If you want to help with any of these, please look at the open issues, join the
conversation and submit a PR. Please read the Contributing section below.

## Contributing

Contributions are welcome!

To ensure that your contribution is accepted, please follow these guidelines:

- open an issue to discuss your idea before you start working on it, or if there's
  already an issue for your idea, join the conversation there and explain how you
  plan to implement it
- make sure that your code is well documented (docstrings, type annotations, comments,
  etc.) and tested (test coverage should only go up)
- make sure that your code is formatted and type-checked with `ruff` (default settings)

## Copyright

Copyright (C) 2023-2024. Senko Rasic and Think contributors. You may use and/or distribute
this project under the terms of MIT license. See the LICENSE file for more details.

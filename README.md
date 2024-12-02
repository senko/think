# Think

Think is a Python package for creating thinking programs.

It provides simple but powerful primitives for composable and robust
integration of Large Language Models (LLMs) into your Python programs.

Think supports OpenAI, Anthropic, Google (Gemini), Groq as LLM
providers, Ollama for local models, as well as any OpenAI-compatible
LLM API provider.

## Examples

Ask a question:

```python
from asyncio import run

from think import LLM, ask

llm = LLM.from_url("anthropic:///claude-3-haiku-20240307")

async def haiku(topic):
    return await ask(llm, "Write a haiku about {{ topic }}", topic=topic)

print(run(haiku("computers")))
```

Get answers as structured data:

```python
from asyncio import run

from think import LLM, LLMQuery

llm = LLM.from_url("openai:///gpt-4o-mini")

class CityInfo(LLMQuery):
    """
    Give me basic information about {{ city }}.
    """
    name: str
    country: str
    population: int
    latitude: float
    longitude: float

async def city_info(city):
    return await CityInfo.run(llm, city=city)

info = run(city_info("Paris"))
print(f"{info.name} is a city in {info.country} with {info.population} inhabitants.")
```

Integrate AI with custom tools:

```python
from asyncio import run
from datetime import date

from think import LLM, Chat

llm = LLM.from_url("openai:///gpt-4o-mini")

def current_date() -> str:
    """
    Get the current date.

    :returns: current date in YYYY-MM-DD format
    """
    return date.today().isoformat()

async def days_to_xmas() -> str:
    chat = Chat("How many days are left until Christmas?")
    return await llm(chat, tools=[current_date])

print(run(days_to_xmas()))
```

Use vision (with models that support it):

```python
from asyncio import run

from think import LLM, Chat

llm = LLM.from_url("openai:///gpt-4o-mini")

async def describe_image(path):
    image_data = open(path, "rb").read()
    chat = Chat().user("Describe the image in detail", images=[image_data])
    return await llm(chat)

print(run(describe_image("path/to/image.jpg")))
```

Use Pydantic or custom parsers for structured data:

```python
from asyncio import run
from ast import parse

from think import LLM, Chat
from think.parser import CodeBlockParser
from think.prompt import JinjaStringTemplate

llm = LLM.from_url("openai:///gpt-4o-mini")

def parse_python(text):
    # extract code block from the text
    block_parser = CodeBlockParser()
    code = block_parser(text)
    # check if the code is valid Python syntax
    try:
        parse(code)
        return code
    except SyntaxError as err:
        raise ValueError(f"Invalid Python code: {err}") from err

async def generate_python_script(task):
    system = "You always output the requested code in a single Markdown code block"
    prompt = "Write a Python script for the following task: {{ task }}"
    tpl = JinjaStringTemplate()
    chat = Chat(system).user(tpl(prompt, task=task))
    return await llm(chat, parser=parse_python)

print(run(generate_python_script("sort a list of numbers")))
```

For detailed documentation on usage and all available features, please refer to the
code docstrings and the integration tests.

## Quickstart

Install via `pip`:

```bash
pip install think-llm
```

Note that the package name is `think-llm`, *not* `think`.

You'll also need to install the providers you want to use:
`openai`, `anthropic`, `google-generativeai`, `groq`, or `ollama`. You
can install them together with Think via `pip` as well:

```bash
pip install think-llm[openai]
pip install think-llm[anthropic]
pip install think-llm[gemini]
pip install think-llm[groq]
pip install think-llm[ollama]
pip install think-llm[all]  # to install all of them
pip install think-llm[dev]  # if you want to run the tests or modify Think
```

You can set up your LLM credentials via environment variables, for example:

```bash
export OPENAI_API_KEY=<your-openai-key>
export ANTHROPIC_API_KEY=<your-anthropic-key>
...
```

Or pass them directly in the model URL:

```python
from think import LLM

llm = LLM.from_url(f"openai://{YOUR_OPENAI_KEY}@/gpt-4o-mini")
```

In practice, you might want to store the entire model URL in the environment
variable and just call `LLM.from_url(os.environ["LLM_URL"])`.

## Model URL

Think uses a URL-like format to specify the model to use. The format is:

```
provider://[api_key@][host[:port]]/model[?query]
```

- `provider` is the model provider, e.g. `openai` or `anthropic`
- `api-key` is the API key for the model provider (optional if set via environment)
- `server` is the server to use, useful for local LLMs; for OpenAI and Anthropic it
    should be empty to use their default base URL
- `model-name` is the name of the model to use

Examples:
    - `openai:///gpt-3.5-turbo` (API key in environment)
    - `openai://sk-my-openai-key@/gpt-3-5-turbo` (explicit API key)
    - `openai://localhost:1234/v1?model=llama-3.2-8b` (custom server over HTTP)
    - `openai+https://openrouter.ai/api/v1?model=llama-3.2-8b` (custom server, HTTPS)

(Note that if the base URL is provided, the model must be passed as a query parameter.)

Using the URL format allows you to easily switch between different models and providers
without changing your code, or using multiple models in the same program without
hardcoding anything.

## Roadmap

Features and capabilities that are planned for the near future:

- documentation
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

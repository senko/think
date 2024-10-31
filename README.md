# Think

Think is a Python package for creating thinking programs.

It provides simple but powerful primitives for composable and robust integration
of Large Language Models (LLMs) into your Python programs.

Think supports OpenAI and Anthropic models.

## Examples

Ask a question:

```python
from think import LLM, ask

llm = LLM.from_url("anthropic:///claude-3-haiku-20240307")

async def haiku(topic):
    return await ask(llm, "Write a haiku about {{ topic }}", topic=topic)

print(asyncio.run(haiku("computers")))
```

Get answers as structured data:

```python
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

info = asyncio.run(city_info("Paris"))
print(f"{info.name} is a city in {info.country} with {info.population} inhabitants.")
```

Integrate AI with custom tools:

```python
from datetime import date

from think import LLM
from think.llm.chat import Chat

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

print(asyncio.run(days_to_xmas()))
```

Use vision (with models that support it):

```python

from think import LLM
from think.llm.chat import Chat

llm = LLM.from_url("openai:///gpt-4o-mini")

async def describe_image(path):
    image_data = open(path, "rb").read()
    chat = Chat().user("Describe the image in detail", images=[image_data])
    return await llm(chat)

print(asyncio.run(describe_image("path/to/image.jpg")))
```

## Quickstart

Install via `pip`:

```bash
pip install think-llm
```

Note that the package name is `think-llm`, *not* `think`.

You can set up your LLM credentials via environment variables, for example:

```bash
export OPENAI_API_KEY=<your-openai-key>
export ANTHROPIC_API_KEY=<your-anthropic-key>
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
provider://[api-key@]server/model-name
```

- `provider` is the model provider, e.g. `openai` or `anthropic`
- `api-key` is the API key for the model provider (optional if set via environment)
- `server` is the server to use, useful for local LLMs; for OpenAI and Anthropic it
    should be empty to use their default base URL
- `model-name` is the name of the model to use

Using the URL format allows you to easily switch between different models and providers
without changing your code, or using multiple models in the same program without
hardcoding anything.

## Roadmap

Features and capabilities that are planned for the near future:

- documentation
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

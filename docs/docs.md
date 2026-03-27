# Think Documentation

Think is a Python package for creating thinking programs.

It provides simple but powerful primitives for composable and robust
integration of Large Language Models (LLMs) into your Python programs.

Think supports OpenAI, Anthropic, Google (Gemini), Amazon (Bedrock) and Groq as LLM
providers, Ollama for local models, as well as any OpenAI-compatible
LLM API provider.

## Table of Contents

- [Quickstart](#quickstart)

- [Basic LLM Use](#basic-llm-use)

- [Supported Providers](#supported-providers)

- [Chat/Conversation Manipulation](#chatconversation-manipulation)

- [Prompting](#prompting)

- [Structured Outputs and Parsing](#structured-outputs-and-parsing)

- [Vision and Document Handling](#vision-and-document-handling)

- [Streaming](#streaming)

- [Tool Use](#tool-use)

- [RAG (Retrieval-Augmented Generation)](#rag-(retrieval-augmented-generation))

- [Agents](#agents)

- [API Reference](#api-reference)

## Quickstart

Install via `pip`:

```bash
pip install think-llm
```

Note that the package name is `think-llm`, *not* `think`.

You'll also need to install the providers you want to use:
`openai`, `anthropic`, `google-generativeai`, `groq`, or `ollama`.

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

## Guide


### Basic LLM Use

#### Core LLM Functionality


# Core LLM Functionality

The `llm.base` module provides the core functionality for interacting with large language models (LLMs).
It defines the `LLM` class, which is the main entry point for sending requests to LLMs and processing
their responses.

## Basic Usage

```python
# example: basic_llm.py
from think import LLM

# Initialize an LLM using a URL-based configuration
llm = LLM.from_url("openai:///gpt-5-nano")

# Create a simple chat
from think.llm.chat import Chat
chat = Chat("What is the capital of France?")

# Get a response
import asyncio
response = asyncio.run(llm(chat))
print(response)
```

## Model URL Format

Think uses a URL-like format to specify the model to use:

```
provider://[api_key@][host[:port]]/model[?query]
```

- `provider` is the model provider (openai, anthropic, google, etc.)
- `api-key` is the API key (optional if set via environment)
- `host[:port]` is the server to use (optional, for local LLMs)
- `model` is the name of the model to use

Examples:
- `openai:///gpt-5-nano` (API key from OPENAI_API_KEY environment variable)
- `anthropic://sk-my-key@/claude-3-opus-20240229` (explicit API key)
- `openai://localhost:8080/wizard-mega` (custom server over HTTP)
- `openai:///gpt-4o?service_tier=flex` (extra parameters passed to the API)

## Streaming

For generating responses incrementally:

```python
# example: streaming.py
import asyncio
from think import LLM
from think.llm.chat import Chat

llm = LLM.from_url("anthropic:///claude-3-haiku-20240307")

async def stream_response():
    chat = Chat("Generate a short poem about programming")
    async for chunk in llm.stream(chat):
        print(chunk, end="", flush=True)
    print()

asyncio.run(stream_response())
```

## Error Handling

The LLM class throws specific exceptions for different error cases:
- `ConfigError`: Configuration errors (invalid URL, missing API key)
- `BadRequestError`: Invalid requests (e.g., inappropriate content)
- Other standard exceptions like `ConnectionError`, `TimeoutError`

See [Supported Providers](#supported-providers) for provider-specific information.


#### High-level API


# High-level API

The `ai` module provides high-level functions and classes for interacting with LLMs,
making it easier to perform common tasks without dealing with the lower-level details.

## Quick API Calls

The `ask` function provides a simple way to get a response from an LLM:

```python
# example: ask_quick.py
import asyncio
from think import LLM, ask

llm = LLM.from_url("openai:///gpt-3.5-turbo")

async def main():
    response = await ask(llm, "What is the capital of France?")
    print(response)

    # With template variables
    response = await ask(llm, "Write a haiku about {{ topic }}", topic="autumn leaves")
    print(response)

asyncio.run(main())
```

## Structured Outputs with Pydantic

The `LLMQuery` class makes it easy to get structured data from LLMs using Pydantic models:

```python
# example: structured_query.py
import asyncio
from think import LLM, LLMQuery

llm = LLM.from_url("openai:///gpt-5-nano")

class WeatherForecast(LLMQuery):
    '''
    Provide a weather forecast for {{ city }} for today.
    Include temperature in Celsius and conditions.
    '''

    temperature_celsius: float
    conditions: str
    humidity_percent: int
    wind_speed_kmh: float

async def main():
    forecast = await WeatherForecast.run(llm, city="London")
    print(f"Temperature: {forecast.temperature_celsius}°C")
    print(f"Conditions: {forecast.conditions}")
    print(f"Humidity: {forecast.humidity_percent}%")
    print(f"Wind: {forecast.wind_speed_kmh} km/h")

asyncio.run(main())
```

See also:
- [Basic LLM Use](#basic-llm-use) for more detailed LLM interaction
- [Structured Outputs and Parsing](#structured-outputs-and-parsing) for advanced parsing options



### Supported Providers

#### Overview


# Supported Providers

Think supports multiple LLM providers through a consistent interface, making it easy to
switch between different providers or use multiple providers in the same application.

## Available Providers

### OpenAI

```python
# example: openai_provider.py
from think import LLM
import asyncio

# Using API key from environment variable OPENAI_API_KEY
llm = LLM.from_url("openai:///gpt-5-nano")

# With explicit API key
llm = LLM.from_url("openai://sk-your-api-key@/gpt-4o-mini")

async def main():
    response = await llm("What is artificial intelligence?")
    print(response)

asyncio.run(main())
```

Supported models: gpt-3.5-turbo, gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini, and others.

### Anthropic

```python
# example: anthropic_provider.py
from think import LLM
import asyncio

# Using API key from environment variable ANTHROPIC_API_KEY
llm = LLM.from_url("anthropic:///claude-3-haiku-20240307")

# With explicit API key
llm = LLM.from_url("anthropic://your-api-key@/claude-3-opus-20240229")

async def main():
    response = await llm("What is artificial intelligence?")
    print(response)

asyncio.run(main())
```

Supported models: claude-3-opus, claude-3-sonnet, claude-3-haiku, and others.

### Google (Gemini)

```python
# example: google_provider.py
from think import LLM
import asyncio

# Using API key from environment variable GOOGLE_API_KEY
llm = LLM.from_url("google:///gemini-1.5-pro")

async def main():
    response = await llm("What is artificial intelligence?")
    print(response)

asyncio.run(main())
```

Supported models: gemini-1.0-pro, gemini-1.5-pro, gemini-1.5-flash, and others.

### Amazon Bedrock

```python
# example: bedrock_provider.py
from think import LLM
import asyncio

# Using AWS credentials from environment variables
llm = LLM.from_url("bedrock:///anthropic.claude-3-sonnet-20240229-v1:0")

async def main():
    response = await llm("What is artificial intelligence?")
    print(response)

asyncio.run(main())
```

Amazon Bedrock supports models from multiple providers like Anthropic, AI21, Cohere, etc.

### Groq

```python
# example: groq_provider.py
from think import LLM
import asyncio

# Using API key from environment variable GROQ_API_KEY
llm = LLM.from_url("groq:///?model=openai/gpt-oss-20b")

async def main():
    response = await llm("What is artificial intelligence?")
    print(response)

asyncio.run(main())
```

Supported models: llama-3-8b, llama-3-70b, mixtral-8x7b, gemma-7b, and others.

### Ollama (Local Models)

```python
# example: ollama_provider.py
from think import LLM
import asyncio

# Connect to local Ollama server
llm = LLM.from_url("ollama://localhost:11434/qwen3:8b")

async def main():
    response = await llm("What is artificial intelligence?")
    print(response)

asyncio.run(main())
```

Supports any model available in Ollama.

## Custom Provider Configuration

For OpenAI-compatible APIs (like LiteLLM, vLLM, etc.):

```python
# example: custom_provider.py
from think import LLM
import asyncio

# Custom OpenAI-compatible API server
llm = LLM.from_url("openai://api-key@localhost:8000/v1?model=llama-3-8b")

async def main():
    response = await llm("What is artificial intelligence?")
    print(response)

asyncio.run(main())
```

## Extra Parameters

Query parameters in the model URL (other than `model`) are passed through as extra
keyword arguments to the underlying provider API calls:

```python
# example: extra_params.py
from think import LLM

# OpenAI with service tier
llm = LLM.from_url("openai:///gpt-4o?service_tier=flex")

# Bedrock with region (required)
llm = LLM.from_url("bedrock:///anthropic.claude-3-sonnet-20240229-v1:0?region=us-east-1")
```

Note: query parameter values are always strings. For parameters that require numeric
types, use `LLM.for_provider()` directly:

```python
client_class = LLM.for_provider("openai")
llm = client_class("gpt-4o", timeout=30)
```

See also:
- [Basic LLM Use](#basic-llm-use) for more detailed usage
- [Model URL](#model-url) format documentation



### Chat/Conversation Manipulation


# Chat/Conversation Manipulation

The `llm.chat` module provides the core functionality for creating and managing chat conversations with LLMs.
It defines the `Chat` class and related components for structuring messages, managing conversation history,
and handling different content types (text, images, documents).

## Basic Chat Usage

```python
# example: basic_chat.py
from think import LLM
from think.llm.chat import Chat
import asyncio

llm = LLM.from_url("openai:///gpt-5-nano")

async def simple_chat():
    # Create a chat with a system prompt and user message
    chat = Chat("You are a helpful assistant.")
    chat.user("What is the capital of France?")

    # Send to LLM and get response
    response = await llm(chat)
    print(response)

    # Continue the conversation
    chat.user("What's the population of that city?")
    response = await llm(chat)
    print(response)

asyncio.run(simple_chat())
```

## Role-Based Messages

Chat supports different message roles:

```python
# example: chat_roles.py
from think import LLM
from think.llm.chat import Chat
import asyncio

llm = LLM.from_url("openai:///gpt-5-nano")

async def role_based_chat():
    chat = Chat()
    chat.system("You are a helpful but sarcastic assistant.")
    chat.user("Tell me about the solar system.")
    chat.assistant("The solar system? Oh, just a small collection of cosmic bodies " +
                   "orbiting a giant nuclear furnace we call the Sun. No big deal.")
    chat.user("And what about Earth?")

    response = await llm(chat)
    print(response)

asyncio.run(role_based_chat())
```

## Vision Capabilities

For models that support vision, you can include images in your messages:

```python
# example: vision_chat.py
from think import LLM
from think.llm.chat import Chat
import asyncio

llm = LLM.from_url("openai:///gpt-5-nano")  # Use a vision-capable model

async def analyze_image():
    # Load image data
    with open("image.jpg", "rb") as f:
        image_data = f.read()

    # Create chat with image
    chat = Chat().user("What's in this image?", images=[image_data])

    response = await llm(chat)
    print(response)

asyncio.run(analyze_image())
```

## Document Handling

For models supporting documents (like PDFs):

```python
# example: document_chat.py
from think import LLM
from think.llm.chat import Chat
import asyncio

llm = LLM.from_url("google:///gemini-1.5-pro")  # Use a document-capable model

async def analyze_document():
    # Load PDF data
    with open("document.pdf", "rb") as f:
        pdf_data = f.read()

    # Create chat with document
    chat = Chat().user("Summarize this document", documents=[pdf_data])

    response = await llm(chat)
    print(response)

asyncio.run(analyze_document())
```

See also:
- [Basic LLM Use](#basic-llm-use) for more about using Chat with LLMs
- [Vision and Document Handling](#vision-and-document-handling) for advanced usage
- [Tool Use](#tool-use) for using Chat with tools



### Prompting


# Prompt Templates

The `prompt` module provides functionality for creating and managing templates
for prompts to be sent to LLMs. It's built on top of Jinja2 and offers both
string-based and file-based templating.

## String Templates

The simplest way to use templates is with `JinjaStringTemplate`:

```python
# example: string_template.py
from think.prompt import JinjaStringTemplate

template = JinjaStringTemplate()
prompt = template("Hello, my name is {{ name }} and I'm {{ age }} years old.",
                  name="Alice", age=30)
print(prompt)  # Outputs: Hello, my name is Alice and I'm 30 years old.
```

## File Templates

For more complex prompts, you can use file-based templates:

```python
# example: file_template.py
from pathlib import Path
from think.prompt import JinjaFileTemplate

# Create a template file
template_path = Path("my_template.txt")
template_path.write_text("Hello, my name is {{ name }} and I'm {{ age }} years old.")

# Use the template
template = JinjaFileTemplate(template_path.parent)
prompt = template("my_template.txt", name="Bob", age=25)
print(prompt)  # Outputs: Hello, my name is Bob and I'm 25 years old.
```

## Multi-line Templates

When working with multi-line templates, the `strip_block` function helps
preserve the relative indentation while removing the overall indentation:

```python
# example: multiline_template.py
from think.prompt import JinjaStringTemplate, strip_block

template = JinjaStringTemplate()
prompt_text = strip_block('''
    System:
        You are a helpful assistant.

    User:
        {{ question }}
''')

prompt = template(prompt_text, question="How does photosynthesis work?")
print(prompt)
```

## Using with LLMs

Templates integrate seamlessly with the Think LLM interface:

```python
# example: template_with_llm.py
import asyncio
from think import LLM, ask
from think.prompt import JinjaStringTemplate

llm = LLM.from_url("openai:///gpt-3.5-turbo")
template = JinjaStringTemplate()

async def main():
    prompt = template("Write a {{ length }} poem about {{ topic }}.",
                     length="short", topic="artificial intelligence")
    response = await ask(llm, prompt)
    print(response)

asyncio.run(main())
```

See also:
- [Basic LLM Use](#basic-llm-use) for using templates with LLMs
- [Structured Outputs and Parsing](#structured-outputs-and-parsing) for combining templates with structured outputs



### Structured Outputs and Parsing


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

llm = LLM.from_url("openai:///gpt-5-nano")
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

llm = LLM.from_url("openai:///gpt-5-nano")
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

llm = LLM.from_url("openai:///gpt-5-nano")
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

llm = LLM.from_url("openai:///gpt-5-nano")

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



### Vision and Document Handling


# Chat/Conversation Manipulation

The `llm.chat` module provides the core functionality for creating and managing chat conversations with LLMs.
It defines the `Chat` class and related components for structuring messages, managing conversation history,
and handling different content types (text, images, documents).

## Basic Chat Usage

```python
# example: basic_chat.py
from think import LLM
from think.llm.chat import Chat
import asyncio

llm = LLM.from_url("openai:///gpt-5-nano")

async def simple_chat():
    # Create a chat with a system prompt and user message
    chat = Chat("You are a helpful assistant.")
    chat.user("What is the capital of France?")

    # Send to LLM and get response
    response = await llm(chat)
    print(response)

    # Continue the conversation
    chat.user("What's the population of that city?")
    response = await llm(chat)
    print(response)

asyncio.run(simple_chat())
```

## Role-Based Messages

Chat supports different message roles:

```python
# example: chat_roles.py
from think import LLM
from think.llm.chat import Chat
import asyncio

llm = LLM.from_url("openai:///gpt-5-nano")

async def role_based_chat():
    chat = Chat()
    chat.system("You are a helpful but sarcastic assistant.")
    chat.user("Tell me about the solar system.")
    chat.assistant("The solar system? Oh, just a small collection of cosmic bodies " +
                   "orbiting a giant nuclear furnace we call the Sun. No big deal.")
    chat.user("And what about Earth?")

    response = await llm(chat)
    print(response)

asyncio.run(role_based_chat())
```

## Vision Capabilities

For models that support vision, you can include images in your messages:

```python
# example: vision_chat.py
from think import LLM
from think.llm.chat import Chat
import asyncio

llm = LLM.from_url("openai:///gpt-5-nano")  # Use a vision-capable model

async def analyze_image():
    # Load image data
    with open("image.jpg", "rb") as f:
        image_data = f.read()

    # Create chat with image
    chat = Chat().user("What's in this image?", images=[image_data])

    response = await llm(chat)
    print(response)

asyncio.run(analyze_image())
```

## Document Handling

For models supporting documents (like PDFs):

```python
# example: document_chat.py
from think import LLM
from think.llm.chat import Chat
import asyncio

llm = LLM.from_url("google:///gemini-1.5-pro")  # Use a document-capable model

async def analyze_document():
    # Load PDF data
    with open("document.pdf", "rb") as f:
        pdf_data = f.read()

    # Create chat with document
    chat = Chat().user("Summarize this document", documents=[pdf_data])

    response = await llm(chat)
    print(response)

asyncio.run(analyze_document())
```

See also:
- [Basic LLM Use](#basic-llm-use) for more about using Chat with LLMs
- [Vision and Document Handling](#vision-and-document-handling) for advanced usage
- [Tool Use](#tool-use) for using Chat with tools



### Streaming


# Core LLM Functionality

The `llm.base` module provides the core functionality for interacting with large language models (LLMs).
It defines the `LLM` class, which is the main entry point for sending requests to LLMs and processing
their responses.

## Basic Usage

```python
# example: basic_llm.py
from think import LLM

# Initialize an LLM using a URL-based configuration
llm = LLM.from_url("openai:///gpt-5-nano")

# Create a simple chat
from think.llm.chat import Chat
chat = Chat("What is the capital of France?")

# Get a response
import asyncio
response = asyncio.run(llm(chat))
print(response)
```

## Model URL Format

Think uses a URL-like format to specify the model to use:

```
provider://[api_key@][host[:port]]/model[?query]
```

- `provider` is the model provider (openai, anthropic, google, etc.)
- `api-key` is the API key (optional if set via environment)
- `host[:port]` is the server to use (optional, for local LLMs)
- `model` is the name of the model to use

Examples:
- `openai:///gpt-5-nano` (API key from OPENAI_API_KEY environment variable)
- `anthropic://sk-my-key@/claude-3-opus-20240229` (explicit API key)
- `openai://localhost:8080/wizard-mega` (custom server over HTTP)
- `openai:///gpt-4o?service_tier=flex` (extra parameters passed to the API)

## Streaming

For generating responses incrementally:

```python
# example: streaming.py
import asyncio
from think import LLM
from think.llm.chat import Chat

llm = LLM.from_url("anthropic:///claude-3-haiku-20240307")

async def stream_response():
    chat = Chat("Generate a short poem about programming")
    async for chunk in llm.stream(chat):
        print(chunk, end="", flush=True)
    print()

asyncio.run(stream_response())
```

## Error Handling

The LLM class throws specific exceptions for different error cases:
- `ConfigError`: Configuration errors (invalid URL, missing API key)
- `BadRequestError`: Invalid requests (e.g., inappropriate content)
- Other standard exceptions like `ConnectionError`, `TimeoutError`

See [Supported Providers](#supported-providers) for provider-specific information.



### Tool Use


# Tool Integration

The `llm.tool` module provides functionality for creating and using tools with LLMs.
Tools are functions that LLMs can call to perform actions or retrieve information
during a conversation, enabling more interactive and capable AI assistants.

## Basic Tool Usage

```python
# example: basic_tools.py
import asyncio
from think import LLM
from think.llm.chat import Chat

llm = LLM.from_url("openai:///gpt-5-nano")

def get_weather(location: str) -> str:
    '''
    Get the current weather for a location.

    :param location: The city name or location to get weather for
    :return: Current weather information
    '''
    # In a real app, this would call a weather API
    return f"It's currently sunny and 22°C in {location}"

async def travel_assistant():
    chat = Chat("You are a helpful travel assistant.")
    chat.user("What's the weather like in Paris?")

    # Pass the tool to the LLM
    response = await llm(chat, tools=[get_weather])
    print(response)

asyncio.run(travel_assistant())
```

## Multiple Tools

You can provide multiple tools for the LLM to choose from:

```python
# example: multiple_tools.py
import asyncio
from datetime import datetime
from think import LLM
from think.llm.chat import Chat

llm = LLM.from_url("openai:///gpt-5-nano")

def get_time() -> str:
    '''Get the current time.'''
    return datetime.now().strftime("%H:%M:%S")

def calculate_age(birth_year: int) -> int:
    '''
    Calculate a person's age.

    :param birth_year: The year of birth
    :return: The calculated age
    '''
    current_year = datetime.now().year
    return current_year - birth_year

async def assistant_with_tools():
    chat = Chat("You are a helpful assistant.")
    chat.user("What time is it now? Also, how old is someone born in 1990?")

    response = await llm(chat, tools=[get_time, calculate_age])
    print(response)

asyncio.run(assistant_with_tools())
```

## Tool Kits

For organizing related tools:

```python
# example: tool_kit.py
import asyncio
from think import LLM
from think.llm.chat import Chat
from think.llm.tool import ToolKit

llm = LLM.from_url("openai:///gpt-5-nano")

# Create a toolkit for math operations
math_tools = ToolKit("math")

@math_tools.tool
def add(a: float, b: float) -> float:
    '''Add two numbers.'''
    return a + b

@math_tools.tool
def multiply(a: float, b: float) -> float:
    '''Multiply two numbers.'''
    return a * b

async def math_assistant():
    chat = Chat("You are a math assistant.")
    chat.user("What is 25 + 17, and what is 8 * 9?")

    response = await llm(chat, tools=math_tools)
    print(response)

asyncio.run(math_assistant())
```

See also:
- [Agents](#agents) for building more complex tool-using systems
- [Basic LLM Use](#basic-llm-use) for general LLM interaction



### RAG (Retrieval-Augmented Generation)

#### RAG Base Functionality


# RAG Base Functionality

Retrieval-Augmented Generation (RAG) enhances LLM responses by incorporating relevant information
from external sources. The `rag.base` module provides the core abstractions for building
RAG systems with Think.

## Basic RAG Usage

```python
# example: basic_rag.py
import asyncio
from think import LLM
from think.rag.base import RAG, RagDocument

llm = LLM.from_url("openai:///gpt-5-nano")
rag = RAG.for_provider("txtai")(llm)

async def index_and_query():
    # Step 1: Add documents to the RAG system
    documents = [
        RagDocument(id="doc1", text="Paris is the capital of France and known for the Eiffel Tower."),
        RagDocument(id="doc2", text="London is the capital of the United Kingdom."),
        RagDocument(id="doc3", text="Rome is the capital of Italy and home to the Colosseum.")
    ]
    await rag.add_documents(documents)

    # Step 2: Query the RAG system
    result = await rag("What are some European capitals and their landmarks?")
    print(result)

asyncio.run(index_and_query())
```

## Available RAG Providers

Think supports multiple vector database backends:

- **TxtAI**: Simple in-memory vector database (`"txtai"`)
- **ChromaDB**: Persistent document storage (`"chroma"`)
- **Pinecone**: Scalable cloud vector database (`"pinecone"`)

## Customizing RAG Behavior

You can customize the retrieval process by extending the base RAG classes:

```python
# example: custom_rag.py
import asyncio
from think import LLM
from think.rag.base import RAG, RagDocument
from think.rag.txtai_rag import TxtAIRag

llm = LLM.from_url("openai:///gpt-5-nano")

class CustomRag(TxtAIRag):
    '''Custom RAG implementation with specialized prompting.'''

    async def query_prompt(self, query: str, context: str) -> str:
        '''Override the default prompt template.'''
        return f'''
        Based on the following context:

        {context}

        Please answer this question: {query}

        If the context doesn't contain relevant information, please say so.
        '''

async def custom_rag_demo():
    rag = CustomRag(llm)

    # Add documents
    documents = [
        RagDocument(id="doc1", text="Neural networks are a class of machine learning models."),
        RagDocument(id="doc2", text="Transformers revolutionized natural language processing."),
    ]
    await rag.add_documents(documents)

    # Query
    result = await rag("How do neural networks work?")
    print(result)

asyncio.run(custom_rag_demo())
```

See also:
- [RAG Evaluation](#rag-retrieval-augmented-generation) for benchmarking RAG systems
- [Tool Use](#tool-use) for integrating RAG with other tools


#### RAG Evaluation


# RAG Evaluation

The `rag.eval` module provides tools for evaluating the performance of RAG systems.
It includes metrics for measuring different aspects of RAG quality and functionality.

## Basic Evaluation

```python
# example: rag_eval_basic.py
import asyncio
from think import LLM
from think.rag.base import RAG, RagDocument
from think.rag.eval import RagEval

# Set up the LLM and RAG system
llm = LLM.from_url("openai:///gpt-5-nano")
rag = RAG.for_provider("txtai")(llm)

# Set up the evaluator
evaluator = RagEval(llm)

async def evaluate_rag():
    # Add test documents
    documents = [
        RagDocument(id="doc1", text="The Eiffel Tower is 330 meters tall and located in Paris, France."),
        RagDocument(id="doc2", text="The Great Wall of China is over 21,000 kilometers long."),
        RagDocument(id="doc3", text="The Grand Canyon is 446 km long and up to 29 km wide.")
    ]
    await rag.add_documents(documents)

    # Generate answer
    query = "How tall is the Eiffel Tower?"
    answer = await rag(query)

    # Evaluate answer
    precision = await evaluator.context_precision(query, rag.last_context, answer)
    relevance = await evaluator.answer_relevance(query, answer)

    print(f"Answer: {answer}")
    print(f"Context Precision: {precision}")
    print(f"Answer Relevance: {relevance}")

asyncio.run(evaluate_rag())
```

## Available Metrics

The RagEval class provides several metrics:

1. **Context Precision**: Measures if retrieved documents are relevant to the query
2. **Context Recall**: Measures if all relevant information is retrieved
3. **Faithfulness**: Evaluates if the answer is supported by the retrieved context
4. **Answer Relevance**: Assesses if the answer addresses the query

## Comprehensive Evaluation

```python
# example: rag_eval_comprehensive.py
import asyncio
from think import LLM
from think.rag.base import RAG, RagDocument
from think.rag.eval import RagEval

llm = LLM.from_url("openai:///gpt-5-nano")
rag = RAG.for_provider("txtai")(llm)
evaluator = RagEval(llm)

async def comprehensive_eval():
    # Add documents (assume already done)

    # Define test cases
    test_cases = [
        {"query": "What are the dimensions of the Grand Canyon?", "ground_truth": "The Grand Canyon is 446 km long and up to 29 km wide."},
        {"query": "How tall is the Eiffel Tower?", "ground_truth": "The Eiffel Tower is 330 meters tall."}
    ]

    results = {}
    for tc in test_cases:
        query = tc["query"]
        ground_truth = tc["ground_truth"]

        # Get RAG answer
        answer = await rag(query)

        # Evaluate all metrics
        metrics = {
            "precision": await evaluator.context_precision(query, rag.last_context, answer),
            "recall": await evaluator.context_recall(query, rag.last_context, ground_truth),
            "faithfulness": await evaluator.faithfulness(rag.last_context, answer),
            "relevance": await evaluator.answer_relevance(query, answer)
        }

        results[query] = {"answer": answer, "metrics": metrics}

    # Print results
    for query, result in results.items():
        print(f"Query: {query}")
        print(f"Answer: {result['answer']}")
        print(f"Metrics: {result['metrics']}")
        print()

asyncio.run(comprehensive_eval())
```

See also:
- [RAG Base Functionality](#rag-base-functionality) for RAG implementation details
- [Basic LLM Use](#basic-llm-use) for general LLM interaction



### Agents


# Building Agents

The `agent` module provides classes and utilities for building autonomous AI agents
that can interact with users, use tools, access information, and perform tasks over time.

## Basic Agent

```python
# example: simple_agent.py
import asyncio
from think import LLM
from think.agent import BaseAgent, tool

llm = LLM.from_url("openai:///gpt-5-nano")

class WeatherAgent(BaseAgent):
    '''You are a helpful weather assistant.'''

    @tool
    def get_current_temperature(self, city: str) -> float:
        '''Get the current temperature for a city.'''
        # In a real app, this would call a weather API
        temperatures = {"New York": 22.5, "London": 15.0, "Tokyo": 26.8}
        return temperatures.get(city, 20.0)  # Default temperature if city not found

    @tool
    def convert_celsius_to_fahrenheit(self, celsius: float) -> float:
        '''Convert Celsius to Fahrenheit.'''
        return (celsius * 9/5) + 32

async def main():
    agent = WeatherAgent(llm)
    await agent.run("What's the temperature in London? Can you also convert it to Fahrenheit?")

asyncio.run(main())
```

## Agents with RAG

Agents can be integrated with Retrieval-Augmented Generation (RAG) systems:

```python
# example: rag_agent.py
import asyncio
from think import LLM
from think.agent import BaseAgent, tool
from think.rag.base import RAG, RagDocument

llm = LLM.from_url("openai:///gpt-5-nano")

class KnowledgeAgent(BaseAgent):
    '''You are a helpful assistant with access to a knowledge base.'''

    def __init__(self, llm: LLM):
        super().__init__(llm)
        # Initialize RAG system
        self.rag = RAG.for_provider("txtai")(llm)

    async def setup(self):
        '''Initialize the knowledge base.'''
        documents = [
            RagDocument(id="doc1", text="The speed of light is approximately 299,792,458 meters per second."),
            RagDocument(id="doc2", text="Water boils at 100 degrees Celsius at standard pressure."),
            RagDocument(id="doc3", text="The Earth orbits the Sun at an average distance of 149.6 million kilometers.")
        ]
        await self.rag.add_documents(documents)

    @tool
    async def search_knowledge(self, query: str) -> str:
        '''Search the knowledge base for information.'''
        return await self.rag(query)

async def main():
    agent = KnowledgeAgent(llm)
    await agent.setup()
    await agent.run("What is the speed of light? And what is the boiling point of water?")

asyncio.run(main())
```

## Interactive Agents

Agents can maintain ongoing conversations with users:

```python
# example: interactive_agent.py
import asyncio
from datetime import datetime
from think import LLM
from think.agent import BaseAgent, tool

llm = LLM.from_url("openai:///gpt-5-nano")

class ChatbotAgent(BaseAgent):
    '''You are a friendly and helpful assistant.'''

    @tool
    def get_current_time(self) -> str:
        '''Get the current time.'''
        return datetime.now().strftime("%H:%M:%S")

    async def interact(self, response: str) -> str:
        '''
        Handle interaction with the user.

        This method displays the agent's response and
        gets the next input from the user.
        '''
        print(f"Agent: {response}")
        return input("You: ").strip()

async def main():
    agent = ChatbotAgent(llm)
    # Start with an initial greeting
    await agent.run("Hello! How can I help you today?")

asyncio.run(main())
```

See also:
- [Tool Use](#tool-use) for more about integrating tools
- [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation) for more about RAG




## API Reference


### think.agent


# Building Agents

The `agent` module provides classes and utilities for building autonomous AI agents
that can interact with users, use tools, access information, and perform tasks over time.

## Basic Agent

```python
# example: simple_agent.py
import asyncio
from think import LLM
from think.agent import BaseAgent, tool

llm = LLM.from_url("openai:///gpt-5-nano")

class WeatherAgent(BaseAgent):
    '''You are a helpful weather assistant.'''

    @tool
    def get_current_temperature(self, city: str) -> float:
        '''Get the current temperature for a city.'''
        # In a real app, this would call a weather API
        temperatures = {"New York": 22.5, "London": 15.0, "Tokyo": 26.8}
        return temperatures.get(city, 20.0)  # Default temperature if city not found

    @tool
    def convert_celsius_to_fahrenheit(self, celsius: float) -> float:
        '''Convert Celsius to Fahrenheit.'''
        return (celsius * 9/5) + 32

async def main():
    agent = WeatherAgent(llm)
    await agent.run("What's the temperature in London? Can you also convert it to Fahrenheit?")

asyncio.run(main())
```

## Agents with RAG

Agents can be integrated with Retrieval-Augmented Generation (RAG) systems:

```python
# example: rag_agent.py
import asyncio
from think import LLM
from think.agent import BaseAgent, tool
from think.rag.base import RAG, RagDocument

llm = LLM.from_url("openai:///gpt-5-nano")

class KnowledgeAgent(BaseAgent):
    '''You are a helpful assistant with access to a knowledge base.'''

    def __init__(self, llm: LLM):
        super().__init__(llm)
        # Initialize RAG system
        self.rag = RAG.for_provider("txtai")(llm)

    async def setup(self):
        '''Initialize the knowledge base.'''
        documents = [
            RagDocument(id="doc1", text="The speed of light is approximately 299,792,458 meters per second."),
            RagDocument(id="doc2", text="Water boils at 100 degrees Celsius at standard pressure."),
            RagDocument(id="doc3", text="The Earth orbits the Sun at an average distance of 149.6 million kilometers.")
        ]
        await self.rag.add_documents(documents)

    @tool
    async def search_knowledge(self, query: str) -> str:
        '''Search the knowledge base for information.'''
        return await self.rag(query)

async def main():
    agent = KnowledgeAgent(llm)
    await agent.setup()
    await agent.run("What is the speed of light? And what is the boiling point of water?")

asyncio.run(main())
```

## Interactive Agents

Agents can maintain ongoing conversations with users:

```python
# example: interactive_agent.py
import asyncio
from datetime import datetime
from think import LLM
from think.agent import BaseAgent, tool

llm = LLM.from_url("openai:///gpt-5-nano")

class ChatbotAgent(BaseAgent):
    '''You are a friendly and helpful assistant.'''

    @tool
    def get_current_time(self) -> str:
        '''Get the current time.'''
        return datetime.now().strftime("%H:%M:%S")

    async def interact(self, response: str) -> str:
        '''
        Handle interaction with the user.

        This method displays the agent's response and
        gets the next input from the user.
        '''
        print(f"Agent: {response}")
        return input("You: ").strip()

async def main():
    agent = ChatbotAgent(llm)
    # Start with an initial greeting
    await agent.run("Hello! How can I help you today?")

asyncio.run(main())
```

See also:
- [Tool Use](#tool-use) for more about integrating tools
- [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation) for more about RAG



#### BaseAgent

Base class for agents.

This class provides a framework for creating agents that can
interact with users and perform tasks using a language model (LLM).
It supports the use of tools, which are methods that can be called
by the agent to perform specific actions. The agent can also
interact with users in a conversational manner, asking follow-up
questions or providing additional information based on the user's
input.

The agent docstring, if provided, will be used as the system prompt template
(interpolated with variables passed in via kwargs). This can be overriden
per-instance by passing a string or Path to a file containing the system prompt
template to the constructor. In all cases, Jinja is used as the template engine.

Tools can be defined as methods in the class using the @tool decorator, listed
in the `tools` attribute (useful for reusing existing external functions),
and passed in as a list of callables to the constructor.


##### `BaseAgent.__init__(self, llm: LLM, system: str | Path | None, tools: ToolKit | list[Callable] | None, **kwargs: Any)`

Initialize a new Agent instance.

:param llm: The LLM instance to use for generating responses.
:param system: System prompt to use for the agent (if provided, overrides the docstring).
:param tools: List of tools to add to the agent (adds to @tool/tools).
:param kwargs: Additional keyword arguments for the system template.


##### `BaseAgent.add_tool(self, name: str, tool: Callable) -> None`

Add a tool to the toolkit.

See `think.llm.tool.ToolDefinition` for more details on how the docstring
is used for the tool's description and parameters.

:param name: Name of the tool
:param tool: Tool function



#### RagMixin

Agent mixin for integrating RAG (Retrieval-Augmented Generation) sources.

This mixin allows the agent to use multiple RAG sources for
document retrieval and generation. It provides a method to
initialize the RAG sources and adds lookup functions for each
source to the agent's toolkit.


##### `RagMixin.rag_init(self, rag_sources: dict[str, RAG])`

Initialize the RAG mixin.

The provided dictionary of RAG sources is name → RAG instance, where
"name" should be a single word describing the thing to look up
(for example "movie", "person", etc.)

:param rag_sources: Dictionary of RAG sources to initialize.



#### SimpleRagAgent

Simple RAG agent that uses a single RAG source for document retrieval.

This agent is designed to work with a single RAG source and provides
a simple interface for querying the source and generating responses.

The `rag_name` attribute must be set to the name of the source/object to look up
(for example "movie"; see `RAGMixin` for details).

See `BaseAgent` for more details on how to use the agent.


##### `SimpleRagAgent.__init__(self, llm: LLM, rag: RAG, **kwargs: Any)`

Initialize a new SimpleRAGAgent instance.

:param llm: The LLM instance to use for generating responses.
:param rag: The RAG instance to use for document retrieval.
:param kwargs: Additional keyword arguments for the system template





#### `tool(func: Optional[F]) -> Callable[[F], F]`

Decorator to mark a method as a tool that can be used by the AI Agent.

See `think.llm.tool.ToolDefinition` for more details on how the docstring
is used for the tool's description and parameters.

:param func: The function to decorate
:param name: Optional custom name for the tool (defaults to method name)
:return: The decorated function

Usage:
    @tool
    def my_tool(self, arg1: str) -> str:
        '''Tool docstring'''
        return f"Result: {arg1}"

    @tool(name="custom_name")
    def another_tool(self, arg1: int) -> int:
        '''Another tool docstring'''
        return arg1 * 2



### think.ai


# High-level API

The `ai` module provides high-level functions and classes for interacting with LLMs,
making it easier to perform common tasks without dealing with the lower-level details.

## Quick API Calls

The `ask` function provides a simple way to get a response from an LLM:

```python
# example: ask_quick.py
import asyncio
from think import LLM, ask

llm = LLM.from_url("openai:///gpt-3.5-turbo")

async def main():
    response = await ask(llm, "What is the capital of France?")
    print(response)

    # With template variables
    response = await ask(llm, "Write a haiku about {{ topic }}", topic="autumn leaves")
    print(response)

asyncio.run(main())
```

## Structured Outputs with Pydantic

The `LLMQuery` class makes it easy to get structured data from LLMs using Pydantic models:

```python
# example: structured_query.py
import asyncio
from think import LLM, LLMQuery

llm = LLM.from_url("openai:///gpt-5-nano")

class WeatherForecast(LLMQuery):
    '''
    Provide a weather forecast for {{ city }} for today.
    Include temperature in Celsius and conditions.
    '''

    temperature_celsius: float
    conditions: str
    humidity_percent: int
    wind_speed_kmh: float

async def main():
    forecast = await WeatherForecast.run(llm, city="London")
    print(f"Temperature: {forecast.temperature_celsius}°C")
    print(f"Conditions: {forecast.conditions}")
    print(f"Humidity: {forecast.humidity_percent}%")
    print(f"Wind: {forecast.wind_speed_kmh} km/h")

asyncio.run(main())
```

See also:
- [Basic LLM Use](#basic-llm-use) for more detailed LLM interaction
- [Structured Outputs and Parsing](#structured-outputs-and-parsing) for advanced parsing options



#### LLMQuery








### think.llm.anthropic




#### AnthropicAdapter

Adapter for Anthropic Claude API.

See `BaseAdapter` for more details.


##### `AnthropicAdapter.get_tool_spec(self, tool: ToolDefinition) -> dict`




##### `AnthropicAdapter.dump_role(self, role: Role) -> Literal['user', 'assistant']`




##### `AnthropicAdapter.dump_content_part(self, part: ContentPart) -> dict`




##### `AnthropicAdapter.parse_content_part(self, part: dict) -> ContentPart`




##### `AnthropicAdapter.dump_message(self, message: Message) -> dict`




##### `AnthropicAdapter.parse_message(self, message: dict | AnthropicMessage) -> Message`




##### `AnthropicAdapter.dump_chat(self, chat: Chat) -> tuple[str | NotGiven, list[dict]]`




##### `AnthropicAdapter.load_chat(self, messages: list[dict], system: str | None) -> Chat`





#### AnthropicClient

LLM client for Anthropic Claude API.

See `LLM` for more details.


##### `AnthropicClient.__init__(self, model: str, **kwargs)`








### think.llm.base


# Core LLM Functionality

The `llm.base` module provides the core functionality for interacting with large language models (LLMs).
It defines the `LLM` class, which is the main entry point for sending requests to LLMs and processing
their responses.

## Basic Usage

```python
# example: basic_llm.py
from think import LLM

# Initialize an LLM using a URL-based configuration
llm = LLM.from_url("openai:///gpt-5-nano")

# Create a simple chat
from think.llm.chat import Chat
chat = Chat("What is the capital of France?")

# Get a response
import asyncio
response = asyncio.run(llm(chat))
print(response)
```

## Model URL Format

Think uses a URL-like format to specify the model to use:

```
provider://[api_key@][host[:port]]/model[?query]
```

- `provider` is the model provider (openai, anthropic, google, etc.)
- `api-key` is the API key (optional if set via environment)
- `host[:port]` is the server to use (optional, for local LLMs)
- `model` is the name of the model to use

Examples:
- `openai:///gpt-5-nano` (API key from OPENAI_API_KEY environment variable)
- `anthropic://sk-my-key@/claude-3-opus-20240229` (explicit API key)
- `openai://localhost:8080/wizard-mega` (custom server over HTTP)
- `openai:///gpt-4o?service_tier=flex` (extra parameters passed to the API)

## Streaming

For generating responses incrementally:

```python
# example: streaming.py
import asyncio
from think import LLM
from think.llm.chat import Chat

llm = LLM.from_url("anthropic:///claude-3-haiku-20240307")

async def stream_response():
    chat = Chat("Generate a short poem about programming")
    async for chunk in llm.stream(chat):
        print(chunk, end="", flush=True)
    print()

asyncio.run(stream_response())
```

## Error Handling

The LLM class throws specific exceptions for different error cases:
- `ConfigError`: Configuration errors (invalid URL, missing API key)
- `BadRequestError`: Invalid requests (e.g., inappropriate content)
- Other standard exceptions like `ConnectionError`, `TimeoutError`

See [Supported Providers](#supported-providers) for provider-specific information.



#### BaseAdapter

Abstract base class for the LLM API adapters

Adapters are responsible for converting the LLM API calls into the
format expected by the underlying API. They also handle the conversion
of the API responses into the format expected by the LLM.


##### `BaseAdapter.__init__(self, toolkit: ToolKit | None)`

Initialize the adapter.

:param toolkit: Optional toolkit to provide tool functions

Toolkit, if provided, is made available to the underlying LLM for tool
(function) use.


##### `BaseAdapter.get_tool_spec(self, tool: ToolDefinition) -> dict`

Get the provider-specific tool specification for a tool definition.

:param tool: The tool definition
:return: The provider-specific tool specification


##### `BaseAdapter.spec(self) -> list[dict] | None`

Generate the provider-specific tool specification for all the
tools passed to the LLM.

Note that some LLM APIs require a sentinel value (NOT_GIVEN) instead
of None if no tools are defined. This shouold be handled by the
provider-specific LLM client.

:return: The provider-specific tool specification or None if there
    are no tools defined.



#### ConfigError

Configuration error

Encompasses non-recoverable errors due to incorrect configuration
values, such as:

* incorrect or missing API keys
* incorrect base URL (if provided)
* unrecognized model
* invalid parameters


##### `ConfigError.__init__(self, message: str)`





#### BadRequestError

Bad request error

Encompasses non-recoverable errors due to incorrect request
values, such as:

* invalid chat messages
* invalid tool calls
* invalid parameters


##### `BadRequestError.__init__(self, message: str)`





#### LLM

LLM client

This is a base class for the LLM clients. It provides the common
functionality for making LLM API calls and processing the responses.
The provider-specific LLM clients inherit from this class to implement
the provider-specific API calls and response processing.

Example usage:

>>> client = LLM.from_url("openai:///gpt-3.5-turbo")
>>> client(...)


##### `LLM.__init__(self, model: str, **kwargs)`

Initialize the LLM client.

This must be called on the provider-specific LLM class:

>>> client_class = LLM.for_provider("openai")
>>> client = client_class("gpt-3.5-turbo", api_key="secret")

In most cases, you should use the `from_url` class method instead.

:param model: The model to use
:param api_key: Optional API key (if required by the provider and
    not available in the environment variables)
:param **kwargs: Optional extra parameters for the provider API
:param base_url: Optional base URL for the provider API


##### `LLM.for_provider(cls, provider: str) -> type['LLM']`

Get the LLM client class for the specified provider.

:param provider: The provider name
:return: The LLM client class for the provider

Raises a ValueError if the provider is not supported.
The list of supported providers is available in the
PROVIDERS class attribute.


##### `LLM.from_url(cls, url: str) -> 'LLM'`

Initialize an LLM client from a URL.

:param url: The URL to initialize the client from
:return: The LLM client instance

The URL format is: `provider://[api_key@][host[:port]]/model[?query]`

Examples:
    - `openai:///gpt-3.5-turbo` (API key in environment)
    - `openai://sk-my-openai-key@/gpt-3-5-turbo` (explicit API key)
    - `openai://localhost:1234/v1?model=llama-3.2-8b` (custom server over HTTP)
    - `openai+https://openrouter.ai/api/v1?model=llama-3.2-8b` (custom server, HTTPS)
    - `bedrock:///amazon.nova-lite-v1:0?region=us-east-1 (AWS region as an extra param)

Note that if the base URL is provided, the model must be passed
as a query parameter.

Query parameters (other than ``model``) are passed through as extra
keyword arguments to the underlying provider API calls. For example,
``openai:///gpt-4o?service_tier=flex`` passes ``service_tier="flex"``
to OpenAI's ``chat.completions.create()``.

Note: query parameter values are always strings. For parameters that
require numeric types, use ``LLM.for_provider()`` directly.






### think.llm.bedrock




#### BedrockAdapter

Adapter for AWS Bedrock API request/response format.

See `BaseAdapter` for more details on the adapter interface
and https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html#BedrockRuntime.Client.converse
for the Bedrock API reference.


##### `BedrockAdapter.get_tool_spec(self, tool: ToolDefinition) -> dict`




##### `BedrockAdapter.spec(self) -> dict | None`




##### `BedrockAdapter.dump_role(self, role: Role) -> Literal['user', 'assistant']`




##### `BedrockAdapter.dump_content_part(self, part: ContentPart) -> dict`




##### `BedrockAdapter.parse_content_part(self, part: dict) -> ContentPart`




##### `BedrockAdapter.dump_message(self, message: Message) -> dict`




##### `BedrockAdapter.parse_message(self, message: dict) -> Message`




##### `BedrockAdapter.dump_chat(self, chat: Chat) -> tuple[str | None, list[dict]]`




##### `BedrockAdapter.load_chat(self, messages: list[dict], system: str | None) -> Chat`





#### BedrockClient

LLM client for AWS Bedrock API.

See `LLM` for more details.


##### `BedrockClient.__init__(self, model: str, **kwargs)`








### think.llm.chat


# Chat/Conversation Manipulation

The `llm.chat` module provides the core functionality for creating and managing chat conversations with LLMs.
It defines the `Chat` class and related components for structuring messages, managing conversation history,
and handling different content types (text, images, documents).

## Basic Chat Usage

```python
# example: basic_chat.py
from think import LLM
from think.llm.chat import Chat
import asyncio

llm = LLM.from_url("openai:///gpt-5-nano")

async def simple_chat():
    # Create a chat with a system prompt and user message
    chat = Chat("You are a helpful assistant.")
    chat.user("What is the capital of France?")

    # Send to LLM and get response
    response = await llm(chat)
    print(response)

    # Continue the conversation
    chat.user("What's the population of that city?")
    response = await llm(chat)
    print(response)

asyncio.run(simple_chat())
```

## Role-Based Messages

Chat supports different message roles:

```python
# example: chat_roles.py
from think import LLM
from think.llm.chat import Chat
import asyncio

llm = LLM.from_url("openai:///gpt-5-nano")

async def role_based_chat():
    chat = Chat()
    chat.system("You are a helpful but sarcastic assistant.")
    chat.user("Tell me about the solar system.")
    chat.assistant("The solar system? Oh, just a small collection of cosmic bodies " +
                   "orbiting a giant nuclear furnace we call the Sun. No big deal.")
    chat.user("And what about Earth?")

    response = await llm(chat)
    print(response)

asyncio.run(role_based_chat())
```

## Vision Capabilities

For models that support vision, you can include images in your messages:

```python
# example: vision_chat.py
from think import LLM
from think.llm.chat import Chat
import asyncio

llm = LLM.from_url("openai:///gpt-5-nano")  # Use a vision-capable model

async def analyze_image():
    # Load image data
    with open("image.jpg", "rb") as f:
        image_data = f.read()

    # Create chat with image
    chat = Chat().user("What's in this image?", images=[image_data])

    response = await llm(chat)
    print(response)

asyncio.run(analyze_image())
```

## Document Handling

For models supporting documents (like PDFs):

```python
# example: document_chat.py
from think import LLM
from think.llm.chat import Chat
import asyncio

llm = LLM.from_url("google:///gemini-1.5-pro")  # Use a document-capable model

async def analyze_document():
    # Load PDF data
    with open("document.pdf", "rb") as f:
        pdf_data = f.read()

    # Create chat with document
    chat = Chat().user("Summarize this document", documents=[pdf_data])

    response = await llm(chat)
    print(response)

asyncio.run(analyze_document())
```

See also:
- [Basic LLM Use](#basic-llm-use) for more about using Chat with LLMs
- [Vision and Document Handling](#vision-and-document-handling) for advanced usage
- [Tool Use](#tool-use) for using Chat with tools



#### Role

Message role (sender identity) in an LLM chat.



#### ContentType

Content type of a part of an LLM chat message.



#### ContentPart

Part of an LLM chat message with a specific type:

* `text`: Textual content
* `image`: Image (PNG or JPG) as a data URL or HTTP/HTTPS URL
    (HTTP/HTTPS supported only by OpenAI)
* `document`: Document in PDF format, as a data URL or HTTP/HTTPS URL
    (HTTP/HTTPS supported only by OpenAI)
* `tool_call`: Tool call made by the assistant
* `tool_response`: Tool response (provided by the client)

Image/document can be provided as either a data URL, raw data (bytes),
or an HTTP(S) URL. If provided as raw data in supported format
(PNG or JPEG for images, PDF for documents), it will be
automatically converted to a data URL.

Note: not all content types are supported by all AI models.


##### `ContentPart.validate_image(cls, v)`

Pydantic validator/converter for the image field.


##### `ContentPart.is_image_url(self) -> bool`

Return True if the image is an HTTP(S) URL.

:return: True if the image is an HTTP(S) URL


##### `ContentPart.image_data(self) -> str | None`

Return base64-encoded image data if possible.

For images provided as HTTP(S) URLs, this will return None.

:return: Base64-encoded image data or None


##### `ContentPart.image_bytes(self) -> bytes | None`

Return raw image data if possible.

For images provided as data URLs, this will return None.

:return: Raw image data (as byte string), or None


##### `ContentPart.image_mime_type(self) -> str | None`

Return the MIME type of the image if possible.

:return: MIME type of the image or None


##### `ContentPart.validate_document(cls, v)`

Pydantic validator/converter for the document field.


##### `ContentPart.is_document_url(self) -> bool`

Return True if the document is an HTTP(S) URL.

:return: True if the document is an HTTP(S) URL


##### `ContentPart.document_data(self) -> str | None`

Return base64-encoded document data if possible.

For documents provided as HTTP(S) URLs, this will return None.

:return: Base64-encoded document data or None


##### `ContentPart.document_bytes(self) -> bytes | None`

Return raw document data if possible.

For documents provided as HTTP(S) URLs, this will return None.

:return: Raw document data (as byte string), or None


##### `ContentPart.document_mime_type(self) -> str | None`

Return the MIME type of the document if possible.

:return: MIME type of the document or None



#### Message

A message in an LLM chat.

Provider-independent representation of a message,
to be converted from/to provider-specific format
by the appropriate adapter.

Note: not all roles are supported by all AI models.

If the LLM call specified a parser and the AI reply
was successfully parsed, the `parsed` field will contain
the parsed output, otherwise it will be None.


##### `Message.create(cls, role: Role) -> 'Message'`

Helper method to create a message with the given role.

When providing tool responses, the dictionary should map tool call IDs
to response content.

:param role: Role of the message
:param text: Text content, if any
:param images: Image(s) attached to the message, if any
:param documents: Document(s) attached to the message, if any
:param tool_calls: Tool calls, if this is an assistant message
:param tool_responses: Tool responses, if this is a tool response
    message.
:return: Message instance


##### `Message.system(cls, text: str, images: list[str | bytes] | None, documents: list[str | bytes] | None) -> 'Message'`

Creates a system message with the given text and optional images.

:param text: The text content of the system message.
:param images: Optional, a list of images associated with the message,
    which can be either strings or bytes.
:param documents: Optional, a list of documents associated with the message,
    which can be either strings or bytes.
:return: A new Message instance.


##### `Message.user(cls, text: str | None, images: list[str | bytes] | None, documents: list[str | bytes] | None) -> 'Message'`

Creates a user message with the given text and optional images.

:param text: The text content of the message.
:param images: Optional, a list of images associated with the message,
    which can be either strings or bytes.
:param documents: Optional, a list of documents associated with the message,
    which can be either strings or bytes.
:return: A new Message instance.


##### `Message.assistant(cls, text: str | None, tool_calls: list[ToolCall] | None) -> 'Message'`

Creates an assistant message with the specified text and tool calls.

:param text: The text content of the assistant message (optional).
:param tool_calls: Optional, list of tool calls that the assistant makes.
:return: A new Message instance.


##### `Message.tool(cls, tool_responses: dict[str, str]) -> 'Message'`

Creates a tool response message with the specified tool responses.

:param tool_responses: Dictionary mapping tool call IDs to response content.
:return: A new Message instance.



#### Chat

A conversation with an LLM.

Provider-independent representation of messages exchanged with
the LLM, to be converted from/to provider-specific format
by the appropriate adapter.


##### `Chat.__init__(self, system_message: str | None)`

Initialize a new chat with an optional system message.

:param system_message: Optional system message to include
    in the chat.


##### `Chat.system(self, text: str, images: list[str | bytes] | None, documents: list[str | bytes] | None) -> 'Chat'`

Add a system message to the chat.

:param text: The text content of the system message.
:param images: Optional, a list of images associated with
    the message, which can be either strings or bytes.
:param documents: Optional, a list of documents associated with
    the message, which can be either strings or bytes.
:return: The chat instance, for chaining.


##### `Chat.user(self, text: str | None, images: list[str | bytes] | None, documents: list[str | bytes] | None) -> 'Chat'`

Add a user message to the chat.

:param text: The text content of the system message.
:param images: Optional, a list of images associated with
    the message, which can be either strings or bytes.
:param documents: Optional, a list of documents associated with
    the message, which can be either strings or bytes.
:return: The chat instance, for chaining.


##### `Chat.assistant(self, text: str | None, tool_calls: list[ToolCall] | None) -> 'Chat'`

Add an assistant message to the chat.

:param text: The text content of the assistant message.
:param tool_calls: Optional, list of tool calls that the assistant makes.
:return: The chat instance, for chaining.


##### `Chat.tool(self, tool_responses: dict[str, str]) -> 'Chat'`

Add a tool response message to the chat.

:param tool_responses: Dictionary mapping tool call IDs to response content.
:return: The chat instance, for chaining.


##### `Chat.dump(self) -> list[dict[str, Any]]`

Dump the chat to a JSON-serializable format.

Note that the format is provider-independent. It's useful for
storing and loading chats, but must be converted to the
provider-specific format by the appropriate adapter.

:return: JSON-serializable representation of the chat.


##### `Chat.load(cls, data: list[dict[str, Any]]) -> 'Chat'`

Load a chat from a JSON-serializable format.

Loads a chat saved using the `dump` method.

:param data: JSON-serializable representation of the chat.
:return: Chat instance.


##### `Chat.clone(self) -> 'Chat'`

Return a copy of the chat.

Performs a deep-copy, so there's no shared state between the
original and the cloned chat.

:return: A copy of the chat.





#### `image_url(value: Any) -> str | None`

Converts raw image data to a data URL.

:param value: The raw image data or URL to be converted.
:return: A data URL representing the image.


#### `document_url(value: Any) -> str | None`

Converts raw document data to a data URL.

:param value: The raw document data or URL to be converted.
:return: A data URL representing the document.



### think.llm.google




#### GoogleAdapter

Adapter for Google Gemini API request/response format.

See `BaseAdapter` for more details on the adapter interface
and https://ai.google.dev/gemini-api/docs/text-generation#rest
for the Gemini API documentation.


##### `GoogleAdapter.get_tool_spec(self, tool: ToolDefinition) -> dict`




##### `GoogleAdapter.spec(self) -> dict | None`




##### `GoogleAdapter.dump_message(self, message: Message) -> list[dict]`




##### `GoogleAdapter.dump_chat(self, chat: Chat) -> tuple[str, list[dict]]`




##### `GoogleAdapter.parse_message(self, message: dict) -> Message`





#### GoogleClient

LLM client for Google Gemini API.

See `LLM` for more details.


##### `GoogleClient.__init__(self, model: str, **kwargs)`








### think.llm.groq




#### GroqAdapter

Adapter for Groq API request/response format.

See `BaseAdapter` for more details on the adapter interface
and https://console.groq.com/docs/api-reference#chat-create-request-body
for the Groq API reference.


##### `GroqAdapter.get_tool_spec(self, tool: ToolDefinition) -> dict`




##### `GroqAdapter.dump_message(self, message: Message) -> dict`




##### `GroqAdapter.dump_chat(self, chat: Chat) -> tuple[str, list[dict]]`




##### `GroqAdapter.parse_message(self, message: dict) -> Message`





#### GroqClient




##### `GroqClient.__init__(self, model: str, **kwargs)`








### think.llm.litellm




#### LiteLLMAdapter

Adapter for LiteLLM that converts think's internal formats to OpenAI-compatible format.

LiteLLM uses OpenAI's API format as the standard, so we follow the same patterns
as the OpenAI adapter but route through litellm for multi-provider support.


##### `LiteLLMAdapter.get_tool_spec(self, tool: ToolDefinition) -> dict`

Convert think tool definition to OpenAI function format.


##### `LiteLLMAdapter.dump_message(self, message: Message) -> list[dict]`

Convert think Message to OpenAI message format.


##### `LiteLLMAdapter.text_content(content: str | list[dict[str, str]] | None) -> str | None`

Extract text content from OpenAI message content.


##### `LiteLLMAdapter.parse_tool_call(self, message: dict[str, Any]) -> Message`

Parse OpenAI tool call message to think Message.


##### `LiteLLMAdapter.parse_assistant_message(self, message: dict[str, Any]) -> Message`

Parse OpenAI assistant message to think Message.


##### `LiteLLMAdapter.parse_message(self, message: dict[str, Any]) -> Message`

Parse OpenAI message format to think Message.


##### `LiteLLMAdapter.dump_chat(self, chat: Chat) -> tuple[str, list[dict]]`

Convert think Chat to OpenAI messages format.


##### `LiteLLMAdapter.load_chat(self, messages: list[dict]) -> Chat`

Convert OpenAI messages to think Chat.



#### LiteLLMClient

LiteLLM client that provides access to 100+ LLM providers through a unified interface.

LiteLLM acts as a universal adapter for various AI providers, normalizing their APIs
into a consistent OpenAI-compatible interface.


##### `LiteLLMClient.__init__(self, model: str, **kwargs)`








### think.llm.ollama




#### OllamaAdapter

Adapter for Ollama API request/response format.

See `BaseAdapter` for more details on the adapter interface
and https://github.com/ollama/ollama/blob/main/docs/api.md for the Ollama API documentation.


##### `OllamaAdapter.get_tool_spec(self, tool: ToolDefinition) -> dict`




##### `OllamaAdapter.dump_message(self, message: Message) -> list[dict[str, str]]`




##### `OllamaAdapter.parse_message(self, message: dict[str, Any]) -> Message`




##### `OllamaAdapter.dump_chat(self, chat: Chat) -> tuple[str, list[dict]]`




##### `OllamaAdapter.load_chat(self, messages: list[dict]) -> Chat`





#### OllamaClient

LLM client for Ollama API server.

See `LLM` for more details.


##### `OllamaClient.__init__(self, model: str, **kwargs)`








### think.llm.openai




#### OpenAIAdapter

Adapter for OpenAI API request/response format.

See `BaseAdapte` for more details on the adapter interface
and https://platform.openai.com/docs/api-reference/chat/create
for the OpenAI API reference.


##### `OpenAIAdapter.get_tool_spec(self, tool: ToolDefinition) -> dict`




##### `OpenAIAdapter.dump_message(self, message: Message) -> list[dict]`




##### `OpenAIAdapter.text_content(content: str | list[dict[str, str]] | None) -> str | None`




##### `OpenAIAdapter.parse_tool_call(self, message: dict[str, Any]) -> Message`




##### `OpenAIAdapter.parse_assistant_message(self, message: dict[str, Any]) -> Message`




##### `OpenAIAdapter.parse_message(self, message: dict[str, Any]) -> Message`




##### `OpenAIAdapter.dump_chat(self, chat: Chat) -> tuple[str, list[dict]]`




##### `OpenAIAdapter.load_chat(self, messages: list[dict]) -> Chat`





#### OpenAIClient

LLM client for OpenAI API.

See `LLM` for more details.


##### `OpenAIClient.__init__(self, model: str, **kwargs)`








### think.llm.tool


# Tool Integration

The `llm.tool` module provides functionality for creating and using tools with LLMs.
Tools are functions that LLMs can call to perform actions or retrieve information
during a conversation, enabling more interactive and capable AI assistants.

## Basic Tool Usage

```python
# example: basic_tools.py
import asyncio
from think import LLM
from think.llm.chat import Chat

llm = LLM.from_url("openai:///gpt-5-nano")

def get_weather(location: str) -> str:
    '''
    Get the current weather for a location.

    :param location: The city name or location to get weather for
    :return: Current weather information
    '''
    # In a real app, this would call a weather API
    return f"It's currently sunny and 22°C in {location}"

async def travel_assistant():
    chat = Chat("You are a helpful travel assistant.")
    chat.user("What's the weather like in Paris?")

    # Pass the tool to the LLM
    response = await llm(chat, tools=[get_weather])
    print(response)

asyncio.run(travel_assistant())
```

## Multiple Tools

You can provide multiple tools for the LLM to choose from:

```python
# example: multiple_tools.py
import asyncio
from datetime import datetime
from think import LLM
from think.llm.chat import Chat

llm = LLM.from_url("openai:///gpt-5-nano")

def get_time() -> str:
    '''Get the current time.'''
    return datetime.now().strftime("%H:%M:%S")

def calculate_age(birth_year: int) -> int:
    '''
    Calculate a person's age.

    :param birth_year: The year of birth
    :return: The calculated age
    '''
    current_year = datetime.now().year
    return current_year - birth_year

async def assistant_with_tools():
    chat = Chat("You are a helpful assistant.")
    chat.user("What time is it now? Also, how old is someone born in 1990?")

    response = await llm(chat, tools=[get_time, calculate_age])
    print(response)

asyncio.run(assistant_with_tools())
```

## Tool Kits

For organizing related tools:

```python
# example: tool_kit.py
import asyncio
from think import LLM
from think.llm.chat import Chat
from think.llm.tool import ToolKit

llm = LLM.from_url("openai:///gpt-5-nano")

# Create a toolkit for math operations
math_tools = ToolKit("math")

@math_tools.tool
def add(a: float, b: float) -> float:
    '''Add two numbers.'''
    return a + b

@math_tools.tool
def multiply(a: float, b: float) -> float:
    '''Multiply two numbers.'''
    return a * b

async def math_assistant():
    chat = Chat("You are a math assistant.")
    chat.user("What is 25 + 17, and what is 8 * 9?")

    response = await llm(chat, tools=math_tools)
    print(response)

asyncio.run(math_assistant())
```

See also:
- [Agents](#agents) for building more complex tool-using systems
- [Basic LLM Use](#basic-llm-use) for general LLM interaction



#### ToolDefinition

A tool available to the LLM.

A tool is a function that can be called by the LLM to perform some
operation. The tool definition includes the function itself, a name
for the tool, a Pydantic model for the function's arguments, and a
description of the tool.


##### `ToolDefinition.__init__(self, func: Callable, name: str | None)`

Define a new tool that runs a function.

The function should have a Sphinx-style docstring with :param: and
:return: lines to describe the parameters and return value. The
docstring should describe the function in detail so that the LLM
can decide when to use it and to know how.

:param func: The function to run.
:param name: The name of the tool, exposed to the LLM. Defaults to the
    function name.


##### `ToolDefinition.parse_docstring(docstring: str) -> dict[str, str]`

Parse the Sphinx-style docstring and extract parameter descriptions.

:param docstring: The docstring to parse.
:return: A dictionary mapping parameter names to descriptions.


##### `ToolDefinition.create_model_from_function(cls, func: Callable) -> type[BaseModel]`

Creates a Pydantic model for agiven function.

This method extracts the function's signature and docstring,
parses the docstring for parameter descriptions, and constructs
a Pydantic model with fields corresponding to the function's
parameters.

:param func: The function from which to create the model.
:return: A Pydantic model class with fields derived from the
    function's parameters and their annotations.



#### ToolCall

A call to a tool.

Parsed assistant/AI tool call.
Contains the tool's ID, name, and arguments.



#### ToolResponse

A response from a tool call.

Contains the reference to the tool call, the response string from
the tool or an error message if the tool call failed.



#### ToolError

Tool error that should be passed back to the LLM.

This exception should be raised by tools when the cause of
the error is on the LLM's side. The LLM will be prompted
to fix their invocation/arguments and try again.



#### ToolKit

A collection of tools available to the LLM.

The toolkit is a collection of functions that the LLM can use to
perform various operations.

Both synchronous and asynchronous functions are supported. Async
functions will be automatically awaited.


##### `ToolKit.__init__(self, functions: list[Callable] | None)`

Initialize the toolkit with a list of functions.

Each function will be introspected to create a tool definition
to be used by LLM to decide which tool to use (if any).

The function should have type annotation for arguments and
return value, and a Sphinx-style docstring with :param: and
:return: lines to describe the parameters and return value.

See `ToolDefinition` for more information on the tool definition.

:param functions: A list of functions to add to the toolkit.


##### `ToolKit.tool_names(self) -> list[str]`

Return a list of tool names.


##### `ToolKit.add_tool(self, func: Callable, name: str | None) -> None`

Add a single tool to the toolkit.

:param func: The function to add as a tool
:param name: Optional custom name for the tool


##### `ToolKit.generate_tool_spec(self, formatter: Callable[[ToolDefinition], dict]) -> list[dict]`

Generate tool specifications to pass to the LLM.






### think.parser


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

llm = LLM.from_url("openai:///gpt-5-nano")
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

llm = LLM.from_url("openai:///gpt-5-nano")
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

llm = LLM.from_url("openai:///gpt-5-nano")
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

llm = LLM.from_url("openai:///gpt-5-nano")

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



#### MultiCodeBlockParser

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
    >>> assert parser(text) == ["first block", "more
code"]

    If no code blocks are found, an empty list is returned:
    


##### `MultiCodeBlockParser.__init__(self)`

Initialize the parser with regex pattern for code blocks.



#### CodeBlockParser

    Parse a Markdown code block from a string.

    Expects exactly one code block, and ignores
    any text before or after it.

    Usage:
    >>> parser = CodeBlockParser()
    >>> text = "text
```py
codeblock
'''
more text"
    >>> assert parser(text) == "codeblock"

    This is a special case of MultiCodeBlockParser,
    checking that there's exactly one block.
    



#### JSONParser

Parse a JSON string into a Python structure or Pydantic model

If the model is provided, the JSON will be parsed
and validated against the model. If the model is
not provided, the JSON will be returned as a dict.

If the JSON is not valid and strict is True (default),
a ValueError is raised. If strict is False,
None is returned instead.

The JSON can be provided as a string or inside a
Markdown code block.


##### `JSONParser.__init__(self, spec: Optional[Type[BaseModel]], strict: bool)`

Initialize the JSON parser.

:param spec: Optional Pydantic model class for validation
:param strict: Whether to raise errors on invalid JSON (default True)


##### `JSONParser.schema(self)`

Get the JSON schema for the Pydantic model if one is specified.

:return: JSON schema dict or None if no spec provided



#### EnumParser

Parse text into one of possible Enum values.

If ignore_case is True (default), the text is
converted to lowercase before parsing.

Raises a ValueError if the text does not match
any of the Enum values.


##### `EnumParser.__init__(self, spec: Type[Enum], ignore_case: bool)`

Initialize the enum parser.

:param spec: The Enum class to parse values into
:param ignore_case: Whether to ignore case when matching (default True)






### think.prompt


# Prompt Templates

The `prompt` module provides functionality for creating and managing templates
for prompts to be sent to LLMs. It's built on top of Jinja2 and offers both
string-based and file-based templating.

## String Templates

The simplest way to use templates is with `JinjaStringTemplate`:

```python
# example: string_template.py
from think.prompt import JinjaStringTemplate

template = JinjaStringTemplate()
prompt = template("Hello, my name is {{ name }} and I'm {{ age }} years old.",
                  name="Alice", age=30)
print(prompt)  # Outputs: Hello, my name is Alice and I'm 30 years old.
```

## File Templates

For more complex prompts, you can use file-based templates:

```python
# example: file_template.py
from pathlib import Path
from think.prompt import JinjaFileTemplate

# Create a template file
template_path = Path("my_template.txt")
template_path.write_text("Hello, my name is {{ name }} and I'm {{ age }} years old.")

# Use the template
template = JinjaFileTemplate(template_path.parent)
prompt = template("my_template.txt", name="Bob", age=25)
print(prompt)  # Outputs: Hello, my name is Bob and I'm 25 years old.
```

## Multi-line Templates

When working with multi-line templates, the `strip_block` function helps
preserve the relative indentation while removing the overall indentation:

```python
# example: multiline_template.py
from think.prompt import JinjaStringTemplate, strip_block

template = JinjaStringTemplate()
prompt_text = strip_block('''
    System:
        You are a helpful assistant.

    User:
        {{ question }}
''')

prompt = template(prompt_text, question="How does photosynthesis work?")
print(prompt)
```

## Using with LLMs

Templates integrate seamlessly with the Think LLM interface:

```python
# example: template_with_llm.py
import asyncio
from think import LLM, ask
from think.prompt import JinjaStringTemplate

llm = LLM.from_url("openai:///gpt-3.5-turbo")
template = JinjaStringTemplate()

async def main():
    prompt = template("Write a {{ length }} poem about {{ topic }}.",
                     length="short", topic="artificial intelligence")
    response = await ask(llm, prompt)
    print(response)

asyncio.run(main())
```

See also:
- [Basic LLM Use](#basic-llm-use) for using templates with LLMs
- [Structured Outputs and Parsing](#structured-outputs-and-parsing) for combining templates with structured outputs



#### FormatTemplate

Template renderer using str.format

Instances of this class, when called with a template string
and keyword arguments, will render and return the template.

:param template: The template string to render.
:param kwargs: Keyword arguments to pass to str.format.
:return: The rendered template string.



#### BaseJinjaTemplate

Base class for Jinja2 template renderers.


##### `BaseJinjaTemplate.__init__(self, loader: Optional[BaseLoader])`

Initialize the Jinja2 template environment.

:param loader: Optional Jinja2 loader for template loading



#### JinjaStringTemplate

String template renderer using Jinja2

Instances of this class, when called with a template string
and keyword arguments, will render and return the template.

:param template: The template string to render.
:param kwargs: Keyword arguments to pass to the template.
:return: The rendered template string.


##### `JinjaStringTemplate.__init__(self)`

Initialize the string template renderer with no loader.



#### JinjaFileTemplate

File template renderer using Jinja2

Instances of this class, when called with a template filename
and keyword arguments, will render and return the template.

Since this class uses the FileSystemLoader, the template
may reference other templates using the Jinja2 include or
extends statements.

:param template: The template filename to render.
:param kwargs: Keyword arguments to pass to the template.
:return: The rendered template string.


##### `JinjaFileTemplate.__init__(self, template_dir: str)`

Initialize the file template renderer with a template directory.

:param template_dir: Path to the directory containing template files
:raises ValueError: If the template directory doesn't exist





#### `strip_block(txt: str) -> str`

Strip a multiline block

Strips whitespace from each line in the block so that the indentation
(if any) within the block is preserved, but the block itself is not
indented. Also strips any trailing whitespace.

:param txt: The block of text to strip.
:return: The stripped block of text.



### think.rag.base


# RAG Base Functionality

Retrieval-Augmented Generation (RAG) enhances LLM responses by incorporating relevant information
from external sources. The `rag.base` module provides the core abstractions for building
RAG systems with Think.

## Basic RAG Usage

```python
# example: basic_rag.py
import asyncio
from think import LLM
from think.rag.base import RAG, RagDocument

llm = LLM.from_url("openai:///gpt-5-nano")
rag = RAG.for_provider("txtai")(llm)

async def index_and_query():
    # Step 1: Add documents to the RAG system
    documents = [
        RagDocument(id="doc1", text="Paris is the capital of France and known for the Eiffel Tower."),
        RagDocument(id="doc2", text="London is the capital of the United Kingdom."),
        RagDocument(id="doc3", text="Rome is the capital of Italy and home to the Colosseum.")
    ]
    await rag.add_documents(documents)

    # Step 2: Query the RAG system
    result = await rag("What are some European capitals and their landmarks?")
    print(result)

asyncio.run(index_and_query())
```

## Available RAG Providers

Think supports multiple vector database backends:

- **TxtAI**: Simple in-memory vector database (`"txtai"`)
- **ChromaDB**: Persistent document storage (`"chroma"`)
- **Pinecone**: Scalable cloud vector database (`"pinecone"`)

## Customizing RAG Behavior

You can customize the retrieval process by extending the base RAG classes:

```python
# example: custom_rag.py
import asyncio
from think import LLM
from think.rag.base import RAG, RagDocument
from think.rag.txtai_rag import TxtAIRag

llm = LLM.from_url("openai:///gpt-5-nano")

class CustomRag(TxtAIRag):
    '''Custom RAG implementation with specialized prompting.'''

    async def query_prompt(self, query: str, context: str) -> str:
        '''Override the default prompt template.'''
        return f'''
        Based on the following context:

        {context}

        Please answer this question: {query}

        If the context doesn't contain relevant information, please say so.
        '''

async def custom_rag_demo():
    rag = CustomRag(llm)

    # Add documents
    documents = [
        RagDocument(id="doc1", text="Neural networks are a class of machine learning models."),
        RagDocument(id="doc2", text="Transformers revolutionized natural language processing."),
    ]
    await rag.add_documents(documents)

    # Query
    result = await rag("How do neural networks work?")
    print(result)

asyncio.run(custom_rag_demo())
```

See also:
- [RAG Evaluation](#rag-retrieval-augmented-generation) for benchmarking RAG systems
- [Tool Use](#tool-use) for integrating RAG with other tools



#### RagDocument

Document structure for RAG systems.

A typed dictionary representing a document in the RAG index,
containing a unique identifier and the document text content.

Attributes:
    id: Unique identifier for the document
    text: The textual content of the document



#### RagResult

Result from a RAG document retrieval operation.

Represents a single document retrieved from the RAG system
along with its relevance score for the given query.

Attributes:
    doc: The retrieved document with id and text
    score: Relevance score (typically 0.0 to 1.0, higher is more relevant)



#### RAG

Abstract base class for Retrieval-Augmented Generation (RAG) systems.

This class defines the common interface for RAG implementations that can
index documents, perform semantic search, and generate answers based on
retrieved context. Different providers can be plugged in by subclassing
this class and implementing the abstract methods.

The RAG pipeline consists of several stages:
1. Document indexing via add_documents()
2. Query preparation via prepare_query()
3. Document retrieval via fetch_results()
4. Optional result reranking via rerank()
5. Answer generation via get_answer()

Supported providers include TxtAI, ChromaDB, and Pinecone, each with
their own specific implementations and capabilities.

Class Attributes:
    PROVIDERS: List of supported RAG provider names
    QUERY_PROMPT: Optional template for query enhancement
    ANSWER_PROMPT: Jinja2 template for answer generation

Example usage:
    rag = RAG.for_provider("txtai")(llm)
    await rag.add_documents([{"id": "1", "text": "content"}])
    answer = await rag("What is the content about?")


##### `RAG.__init__(self, llm: LLM, **kwargs: Any)`

Initialize the RAG instance.

:param llm: The LLM instance to use for generating answers.
:param kwargs: Additional arguments for the specific RAG implementation.


##### `RAG.for_provider(cls, provider: str) -> type['RAG']`

Get the RAG class for the specified provider/engine.

:param provider: The provider name
:return The RAG class for the provider

Raises a ValueError if the provider is not supported.
The list of supported providers is available in the
PROVIDERS class attribute.






### think.rag.chroma_rag




#### ChromaRag




##### `ChromaRag.__init__(self, llm: LLM)`

Initialize a RAG instance using ChromaDB engine.

:param llm: The LLM instance to use for generating answers.
:param collection: The name of the ChromaDB collection to use.
:param path: The path to the directory where ChromaDB will store its data.
    If not specified, ChromaDB will use an in-memory store.






### think.rag.eval


# RAG Evaluation

The `rag.eval` module provides tools for evaluating the performance of RAG systems.
It includes metrics for measuring different aspects of RAG quality and functionality.

## Basic Evaluation

```python
# example: rag_eval_basic.py
import asyncio
from think import LLM
from think.rag.base import RAG, RagDocument
from think.rag.eval import RagEval

# Set up the LLM and RAG system
llm = LLM.from_url("openai:///gpt-5-nano")
rag = RAG.for_provider("txtai")(llm)

# Set up the evaluator
evaluator = RagEval(llm)

async def evaluate_rag():
    # Add test documents
    documents = [
        RagDocument(id="doc1", text="The Eiffel Tower is 330 meters tall and located in Paris, France."),
        RagDocument(id="doc2", text="The Great Wall of China is over 21,000 kilometers long."),
        RagDocument(id="doc3", text="The Grand Canyon is 446 km long and up to 29 km wide.")
    ]
    await rag.add_documents(documents)

    # Generate answer
    query = "How tall is the Eiffel Tower?"
    answer = await rag(query)

    # Evaluate answer
    precision = await evaluator.context_precision(query, rag.last_context, answer)
    relevance = await evaluator.answer_relevance(query, answer)

    print(f"Answer: {answer}")
    print(f"Context Precision: {precision}")
    print(f"Answer Relevance: {relevance}")

asyncio.run(evaluate_rag())
```

## Available Metrics

The RagEval class provides several metrics:

1. **Context Precision**: Measures if retrieved documents are relevant to the query
2. **Context Recall**: Measures if all relevant information is retrieved
3. **Faithfulness**: Evaluates if the answer is supported by the retrieved context
4. **Answer Relevance**: Assesses if the answer addresses the query

## Comprehensive Evaluation

```python
# example: rag_eval_comprehensive.py
import asyncio
from think import LLM
from think.rag.base import RAG, RagDocument
from think.rag.eval import RagEval

llm = LLM.from_url("openai:///gpt-5-nano")
rag = RAG.for_provider("txtai")(llm)
evaluator = RagEval(llm)

async def comprehensive_eval():
    # Add documents (assume already done)

    # Define test cases
    test_cases = [
        {"query": "What are the dimensions of the Grand Canyon?", "ground_truth": "The Grand Canyon is 446 km long and up to 29 km wide."},
        {"query": "How tall is the Eiffel Tower?", "ground_truth": "The Eiffel Tower is 330 meters tall."}
    ]

    results = {}
    for tc in test_cases:
        query = tc["query"]
        ground_truth = tc["ground_truth"]

        # Get RAG answer
        answer = await rag(query)

        # Evaluate all metrics
        metrics = {
            "precision": await evaluator.context_precision(query, rag.last_context, answer),
            "recall": await evaluator.context_recall(query, rag.last_context, ground_truth),
            "faithfulness": await evaluator.faithfulness(rag.last_context, answer),
            "relevance": await evaluator.answer_relevance(query, answer)
        }

        results[query] = {"answer": answer, "metrics": metrics}

    # Print results
    for query, result in results.items():
        print(f"Query: {query}")
        print(f"Answer: {result['answer']}")
        print(f"Metrics: {result['metrics']}")
        print()

asyncio.run(comprehensive_eval())
```

See also:
- [RAG Base Functionality](#rag-base-functionality) for RAG implementation details
- [Basic LLM Use](#basic-llm-use) for general LLM interaction



#### RagEval

Evaluation system for RAG (Retrieval-Augmented Generation) systems.

This class provides comprehensive evaluation metrics for assessing the quality
of RAG systems, including context precision, context recall, faithfulness,
and answer relevance. It uses an LLM to evaluate various aspects of the
retrieval and generation process.

The evaluation metrics are based on established RAG evaluation frameworks
and provide quantitative measures of system performance.

Key metrics:
- Context Precision: How relevant are the retrieved documents?
- Context Recall: How well does retrieval cover ground truth?
- Faithfulness: Are answers supported by retrieved context?
- Answer Relevance: How relevant are answers to queries?

Example usage:
    evaluator = RagEval(rag_system, llm)
    precision = await evaluator.context_precision("query", n_results=10)
    recall = await evaluator.context_recall("query", reference_text)


##### `RagEval.__init__(self, rag: RAG, llm: LLM)`

Initialize the RAG evaluation class.

:param rag: The RAG system to evaluate.
:param llm: The LLM used for evaluation.






### think.rag.pinecone_rag




#### PineconeRag




##### `PineconeRag.__init__(self, llm: LLM)`

Initialize a RAG instance using Pinecone as the vector database.

:param llm: The LLM instance to use for generating answers.
:param index_name: The name of the Pinecone index to use.
:param api_key: Pinecone API key. If not provided, it will be read from the
    PINECONE_API_KEY environment variable.
:param embedding_model: The model to use for generating embeddings.
:param embed_batch_size: The batch size for embedding generation.






### think.rag.txtai_rag




#### TxtAiRag




##### `TxtAiRag.__init__(self, llm: LLM)`

Initialize a RAG instance using the TxtAI engine.

:param llm: The LLM instance to use for generating answers.
:param model: The embeddings model to use. Default is `DEFAULT_EMBEDDINGS_MODEL`.
:param path: The path to the directory where the embeddings will be stored.
    If not specified, the embeddings will not be saved to disk.






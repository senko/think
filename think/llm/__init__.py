"""
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
llm = LLM.from_url("ollama://localhost:11434/llama3")

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

See also:
- [Basic LLM Use](#basic-llm-use) for more detailed usage
- [Model URL](#model-url) format documentation
"""

# Import all providers to make them available
from .base import LLM
from .chat import Chat, ContentPart, ContentType, Message, Role

__all__ = ["LLM", "Chat", "Message", "Role", "ContentPart", "ContentType"]

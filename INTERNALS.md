# Think Library Internals

This document provides a comprehensive overview of the Think library's internal
architecture, components, and implementation details. It serves as a guide for
contributors and maintainers to understand the codebase structure and design
patterns.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core API Layer](#core-api-layer)
3. [LLM Provider System](#llm-provider-system)
4. [Chat and Message System](#chat-and-message-system)
5. [Tool System](#tool-system)
6. [Agent Framework](#agent-framework)
7. [RAG System](#rag-system)
8. [Parsing System](#parsing-system)
9. [Template System](#template-system)
10. [Data Flow](#data-flow)
11. [Design Patterns](#design-patterns)
12. [Provider Integration](#provider-integration)

## Architecture Overview

Think is designed as a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────┐
│             User API Layer              │  think.ask(), LLMQuery
├─────────────────────────────────────────┤
│           Agent Framework               │  BaseAgent, tools, RAG
├─────────────────────────────────────────┤
│         Provider-Agnostic LLM           │  LLM base class, Chat
├─────────────────────────────────────────┤
│           Provider Adapters             │  OpenAI, Anthropic, etc.
├─────────────────────────────────────────┤
│      Supporting Systems (RAG/Parse)     │  RAG, Parsers, Templates
└─────────────────────────────────────────┘
```

### Key Design Principles

1. **Provider Agnostic**: Common interface across all LLM providers
2. **Composable**: Mix and match components (tools, RAG, parsers)
3. **Type Safe**: Extensive use of Pydantic for validation
4. **Async First**: All operations are asynchronous
5. **Extensible**: Plugin architecture for new providers and RAG backends

## Core API Layer

### Location: `think/__init__.py`, `think/ai.py`

The core API provides two main entry points for users:

#### `ask(llm, prompt, **kwargs) -> str`
- Simple text-based queries
- Jinja2 template support in prompts
- Returns raw string response

#### `LLMQuery` Class
- Pydantic-based structured queries
- JSON schema generation from class definition
- Automatic parsing and validation of responses
- Class docstring used as prompt template

```python
class LLMQuery(BaseModel):
    @classmethod
    async def run(cls, llm: LLM, **kwargs) -> "LLMQuery":
        """Core method that handles template rendering, LLM calling, and parsing"""
```

**Key Components:**
- Template rendering via Jinja2
- JSON schema generation from Pydantic model
- Response parsing and validation
- Error handling for malformed responses

## LLM Provider System

### Location: `think/llm/base.py`

The LLM system is built around an abstract base class that defines a common interface for all providers.

#### `LLM` Abstract Base Class

**Key Methods:**
- `__call__()`: Main entry point with overloads for different return types
- `from_url()`: Factory method for creating LLM instances from URLs
- `for_provider()`: Factory method for getting provider-specific classes
- `stream()`: Streaming response support
- `_internal_call()`: Provider-specific implementation (abstract)
- `_internal_stream()`: Provider-specific streaming (abstract)

**Supported Providers:**
- OpenAI (`think/llm/openai.py`)
- Anthropic (`think/llm/anthropic.py`)
- Google Gemini (`think/llm/google.py`)
- Groq (`think/llm/groq.py`)
- Ollama (`think/llm/ollama.py`)
- AWS Bedrock (`think/llm/bedrock.py`)

#### `BaseAdapter` Abstract Base Class

Adapters handle the conversion between Think's internal message format and provider-specific APIs.

**Key Methods:**
- `dump_chat()`: Convert Chat to provider format
- `parse_message()`: Convert provider response to Message
- `dump_message()`: Convert Message to provider format
- `get_tool_spec()`: Generate provider-specific tool specifications

**URL Format:**
`provider://[api_key@][host[:port]]/model[?query]`

Examples:
- `openai:///gpt-4o-mini`
- `anthropic://key@/claude-3-haiku-20240307`
- `openai://localhost:1234/v1?model=llama-3.2-8b`

## Chat and Message System

### Location: `think/llm/chat.py`

The chat system provides a provider-agnostic way to represent conversations.

#### `Chat` Class

**Key Methods:**
- `system()`, `user()`, `assistant()`, `tool()`: Add messages with specific roles
- `dump()`: Serialize to JSON
- `load()`: Deserialize from JSON
- `clone()`: Deep copy conversation

#### `Message` Class

**Key Fields:**
- `role`: Role enum (system, user, assistant, tool)
- `content`: List of ContentPart objects
- `parsed`: Cached parsed response (if applicable)

**Factory Methods:**
- `Message.system()`, `Message.user()`, `Message.assistant()`, `Message.tool()`

#### `ContentPart` Class

Represents different types of content within a message:

**Content Types:**
- `text`: Plain text content
- `image`: Images (PNG/JPEG) as data URLs or HTTP(S) URLs
- `document`: PDF documents as data URLs or HTTP(S) URLs
- `tool_call`: Function calls from assistant
- `tool_response`: Function call responses

**Key Features:**
- Automatic data URL conversion for images/documents
- MIME type detection
- Base64 encoding/decoding utilities

#### `Role` Enum
- `system`: System instructions
- `user`: User messages
- `assistant`: AI responses
- `tool`: Tool/function responses

## Tool System

### Location: `think/llm/tool.py`

The tool system enables LLMs to call functions during conversation.

#### `ToolDefinition` Class

**Key Methods:**
- `__init__()`: Create tool from function with docstring parsing
- `create_model_from_function()`: Generate Pydantic model from function signature
- `parse_docstring()`: Extract parameter descriptions from Sphinx-style docstrings

**Features:**
- Automatic schema generation from function signatures
- Sphinx-style docstring parsing for descriptions
- Type annotation support

#### `ToolKit` Class

**Key Methods:**
- `execute_tool_call()`: Execute a tool call with error handling
- `add_tool()`: Add a function to the toolkit
- `generate_tool_spec()`: Generate provider-specific tool specifications

**Features:**
- Async/sync function support
- Argument validation via Pydantic
- Error handling with `ToolError`

#### Data Classes
- `ToolCall`: Represents a function call (id, name, arguments)
- `ToolResponse`: Represents function response (call reference, response/error)
- `ToolError`: Exception for LLM-side errors

## Agent Framework

### Location: `think/agent.py`

The agent framework provides higher-level abstractions for building AI agents.

#### `BaseAgent` Class

**Key Features:**
- Tool integration via `@tool` decorator
- Conversation management
- System prompt templating with Jinja2
- Interaction loop support

**Key Methods:**
- `invoke()`: Single request/response interaction
- `run()`: Continuous interaction loop
- `interact()`: Override for custom interaction handling
- `add_tool()`: Programmatically add tools

#### `@tool` Decorator

Marks agent methods as tools available to the LLM:

```python
@tool
def my_tool(self, param: str) -> str:
    """Tool description for LLM"""
    return f"Result: {param}"
```

#### `RAGMixin` Class

Provides RAG integration for agents:

**Key Methods:**
- `rag_init()`: Initialize RAG sources
- Automatic tool generation for each RAG source

#### `SimpleRAGAgent` Class

Pre-built agent with single RAG source integration.

## RAG System

### Location: `think/rag/`

The RAG system provides retrieval-augmented generation capabilities with multiple backend support.

#### `RAG` Abstract Base Class

**Key Methods:**
- `add_documents()`: Add documents to index
- `remove_documents()`: Remove documents by ID
- `prepare_query()`: Process user query for search
- `fetch_results()`: Perform semantic search
- `get_answer()`: Generate answer from results
- `rerank()`: Reorder search results
- `calculate_similarity()`: Compute similarity scores
- `__call__()`: End-to-end RAG pipeline

**Supported Providers:**
- TxtAI (`think/rag/txtai_rag.py`)
- ChromaDB (`think/rag/chroma_rag.py`)
- Pinecone (`think/rag/pinecone_rag.py`)

#### `RagDocument` TypedDict
- `id`: Document identifier
- `text`: Document content

#### `RagResult` Dataclass
- `doc`: RagDocument reference
- `score`: Relevance score

#### `RagEval` Class

Evaluation metrics for RAG systems:

**Metrics:**
- `context_precision()`: Precision@k for retrieved documents
- `context_recall()`: Coverage of ground truth in retrieved docs
- `faithfulness()`: Answer support by retrieved documents
- `answer_relevance()`: Answer relevance to query

## Parsing System

### Location: `think/parser.py`

Utilities for parsing LLM outputs into structured formats.

#### `CodeBlockParser` Class
- Extracts single code block from markdown
- Ignores language specifier
- Raises error if not exactly one block found

#### `MultiCodeBlockParser` Class
- Extracts multiple code blocks from markdown
- Returns list of code strings
- Base class for `CodeBlockParser`

#### `JSONParser` Class
- Parses JSON strings with optional Pydantic validation
- Supports JSON within code blocks
- Configurable strict/lenient modes

#### `EnumParser` Class
- Parses strings into enum values
- Case-insensitive option
- Clear error messages with valid options

## Template System

### Location: `think/prompt.py`

Template rendering system supporting multiple engines.

#### `JinjaStringTemplate` Class
- Renders string templates with Jinja2
- Block stripping for clean formatting
- Strict undefined variable handling

#### `JinjaFileTemplate` Class
- Renders file-based templates
- Support for template inheritance and includes
- Directory-based template loading

#### `FormatTemplate` Class
- Simple string.format()-based templating
- Fallback option for basic use cases

#### Utility Functions
- `strip_block()`: Clean indentation from multiline strings

## Data Flow

### Typical Request Flow

1. **User Input**: `ask()` or `LLMQuery.run()` called
2. **Template Rendering**: Jinja2 processes prompt with variables
3. **Chat Creation**: Input converted to Chat/Message objects
4. **Provider Adaptation**: Adapter converts to provider format
5. **LLM API Call**: HTTP request to provider API
6. **Response Processing**: Provider response converted back to Message
7. **Tool Execution**: Any tool calls executed and responses added
8. **Parsing**: Optional parsing of response text
9. **Return**: Final result returned to user

### Tool Execution Flow

1. **Tool Call Detection**: LLM response contains tool calls
2. **Argument Validation**: Pydantic validates call arguments
3. **Function Execution**: Tool function executed (async if needed)
4. **Response Creation**: Results wrapped in ToolResponse
5. **Chat Update**: Tool response added to conversation
6. **LLM Continuation**: Updated chat sent back to LLM

### RAG Flow

1. **Query Processing**: User query optionally enhanced for search
2. **Semantic Search**: Query embedded and matched against index
3. **Result Retrieval**: Top-k documents retrieved with scores
4. **Optional Reranking**: Results reordered by relevance
5. **Answer Generation**: LLM generates answer from context
6. **Response Return**: Final answer returned to user

## Design Patterns

### Factory Pattern
- `LLM.for_provider()`: Create provider-specific instances
- `RAG.for_provider()`: Create RAG backend instances

### Adapter Pattern
- `BaseAdapter`: Convert between internal and provider formats
- Provider-specific adapters handle API differences

### Strategy Pattern
- Different parsing strategies via parser classes
- Different RAG backends with common interface

### Template Method Pattern
- `BaseAgent.run()`: Define interaction loop structure
- Subclasses override `interact()` for custom behavior

### Decorator Pattern
- `@tool`: Add tool functionality to methods
- Preserves original method while adding metadata

### Builder Pattern
- `Chat` class builds conversations incrementally
- Method chaining for fluent interface

## Provider Integration

### Adding New LLM Providers

1. **Create Provider Module**: `think/llm/newprovider.py`
2. **Implement Adapter**: Subclass `BaseAdapter`
3. **Implement Client**: Subclass `LLM`
4. **Register Provider**: Add to `LLM.for_provider()`
5. **Handle Provider Specifics**: Error handling, streaming, tool formats

### Adding New RAG Backends

1. **Create RAG Module**: `think/rag/newrag_rag.py`
2. **Implement RAG Class**: Subclass `RAG`
3. **Implement Required Methods**: All abstract methods
4. **Register Backend**: Add to `RAG.for_provider()`
5. **Handle Backend Specifics**: Connection, indexing, search

### Error Handling

**Exception Hierarchy:**
- `ConfigError`: Configuration issues (API keys, models)
- `BadRequestError`: Invalid request parameters
- `ToolError`: Tool execution errors (LLM-side)

**Error Patterns:**
- Provider errors mapped to common exception types
- Retry logic for parsing failures
- Graceful degradation for optional features

### Testing Considerations

**Key Test Areas:**
- Provider adapter round-trip tests
- Tool execution with various argument types
- Chat serialization/deserialization
- Template rendering edge cases
- RAG evaluation metrics
- Error handling scenarios

**Mocking Strategies:**
- Mock provider HTTP clients
- Use in-memory RAG backends for tests
- Deterministic tool functions
- Fixed LLM responses for parsing tests

## Documentation Coverage

As part of creating this internal documentation, we have ensured comprehensive docstring coverage across the codebase:

### Completed Documentation Areas

**Core API Layer:**
- `LLMQuery.run()` - Structured query execution
- `ask()` - Simple text-based queries

**Agent Framework:**
- `BaseAgent._add_class_tools()` - Tool introspection method
- `@tool` decorator functionality
- All agent initialization and interaction methods

**Parser System:**
- `MultiCodeBlockParser.__init__()` and `__call__()` - Code block extraction
- `JSONParser.__init__()`, `schema` property, and `__call__()` - JSON parsing
- `EnumParser.__init__()` and `__call__()` - Enum value parsing

**Template System:**
- `FormatTemplate.__call__()` - String formatting
- `BaseJinjaTemplate.__init__()` - Environment setup
- `JinjaStringTemplate.__init__()` and `__call__()` - String template rendering
- `JinjaFileTemplate.__init__()` and `__call__()` - File template rendering

**RAG System:**
- `RAG.__init__()` and `__call__()` - Core RAG pipeline
- `RagDocument` and `RagResult` - Data structure documentation
- `BASE_ANSWER_PROMPT` - Template constant documentation
- `RagEval.__init__()` - Evaluation system initialization
- Comprehensive `RagEval` class documentation

### Documentation Standards

All public methods and classes now include:
- Purpose and functionality description
- Parameter documentation with types
- Return value documentation
- Usage examples where appropriate
- Error conditions and exceptions
- Integration points with other components

### Maintenance Guidelines

**For Contributors:**
- Add docstrings to all new public methods and classes
- Follow the established Sphinx-style documentation format
- Include parameter types and return value descriptions
- Provide usage examples for complex functionality
- Document any side effects or state changes

**For Maintainers:**
- Review docstring completeness in pull requests
- Update this internal documentation when architectural changes occur
- Ensure consistency in documentation style across modules
- Validate that examples in docstrings remain functional

This internal documentation should be updated as new features are added and the
architecture evolves. Contributors should refer to this document when making
changes to understand the impact on other components.

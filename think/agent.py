from pathlib import Path
from typing import Callable, Any, Optional, TypeVar
from logging import getLogger
import inspect

from think.llm.base import LLM, CustomParserResultT, PydanticResultT
from think.llm.chat import Chat
from think.llm.tool import ToolKit, ToolDefinition
from think.prompt import JinjaStringTemplate, JinjaFileTemplate
from think.rag.base import RAG


F = TypeVar("F", bound=Callable[..., Any])


def tool(func: Optional[F] = None, *, name: Optional[str] = None) -> Callable[[F], F]:
    """
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
    """

    def decorator(f: F) -> F:
        # Mark the function as a tool
        setattr(f, "_is_tool", True)
        # Store custom name if provided
        if name is not None:
            setattr(f, "_tool_name", name)
        return f

    # Handle both @tool and @tool(name="something") usage
    if func is None:
        return decorator
    return decorator(func)


log = getLogger(__name__)


class BaseAgent:
    """
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
    """

    llm: LLM
    tools: list[Callable] | None = None
    toolkit: ToolKit
    chat: Chat

    def __init__(
        self,
        llm: LLM,
        system: str | Path | None = None,
        tools: ToolKit | list[Callable] | None = None,
        **kwargs: Any,
    ):
        """
        Construct a new Agent instance.

        :param llm: The LLM instance to use for generating responses.
        :param system: System prompt to use for the agent (if provided, overrides the docstring).
        :param tools: List of tools to add to the agent (adds to @tool/tools).
        :param kwargs: Additional keyword arguments for the system template.
        """
        self.llm = llm

        # Initialize toolkit
        if tools is None:
            self.toolkit = ToolKit([])
        elif isinstance(tools, ToolKit):
            self.toolkit = tools
        else:
            self.toolkit = ToolKit(tools)

        # Add class methods marked with @tool decorator to toolkit
        self._add_class_tools()

        # Add explicitly listed tools to toolkit
        if self.tools:
            for tool in self.tools:
                self.add_tool(tool.__name__, tool)

        # Prepare the system message
        if isinstance(system, Path):
            if not system.is_file():
                raise ValueError(
                    f"System prompt file {system} does not exist or is not a file."
                )
            tpl = JinjaFileTemplate(system.parent)
            system_msg = tpl(system.name, **kwargs)
        elif isinstance(system, str):
            system_msg = system
        elif system is None and self.__doc__.strip():
            tpl = JinjaStringTemplate()
            system_msg = tpl(self.__doc__, **kwargs)
        else:
            log.debug(
                f"{self.__class__.__name__}: System prompt is not a string or file: {system}"
            )
            system_msg = None

        self.chat = Chat(system_msg)

    def add_tool(self, name: str, tool: Callable) -> None:
        """
        Add a tool to the toolkit.

        See `think.llm.tool.ToolDefinition` for more details on how the docstring
        is used for the tool's description and parameters.

        :param name: Name of the tool
        :param tool: Tool function
        """
        if name in self.toolkit.tools:
            raise ValueError(f"Tool with name {name} already added to the agent.")
        if not callable(tool):
            raise ValueError("Tool must be a callable function.")
        self.toolkit.tools[name] = ToolDefinition(tool, name=name)
        log.debug(f"{self.__class__.__name__}: Added tool {name}")

    def _add_class_tools(self) -> None:
        # Scan the class for methods marked with @tool decorator and add them to the toolkit.
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, "_is_tool") and getattr(method, "_is_tool", False):
                tool_name = getattr(method, "_tool_name", None) or name
                self.add_tool(tool_name, method)

    async def invoke(
        self,
        query: str | None = None,
        images: list[str | bytes] | None = None,
        documents: list[str | bytes] | None = None,
        parser: type[PydanticResultT]
        | Callable[[str], CustomParserResultT]
        | None = None,
    ) -> str | PydanticResultT | CustomParserResultT:
        """
        Invoke the agent with a query and optional images/documents.

        This runs a single request/response interaction with the agent.
        While the agent may use multiple tools, this method is not intended
        to be used for long-running interactions. For that, use the `run`
        method.

        :param query: The query to send to the agent (optional).
        :param images: List of images to send to the agent (optional).
        :param documents: List of documents to send to the agent (optional).
        :param parser: Optional parser to process the response.
        :return: The response from the agent.
        """
        if query or images or documents:
            self.chat.user(query, images=images)
        response = await self.llm(self.chat, tools=self.toolkit)
        return response

    async def interact(
        self,
        response: str | PydanticResultT | CustomParserResultT,
    ) -> str | None:
        """
        Interact with the user based on the agent's response.

        This method is called after the agent has generated a response.
        It can be used to ask follow-up questions or provide additional
        information to the user. The method should return a new query
        to continue the interaction or None to end it.

        The default implementation returns None, indicating that the
        interaction should end. Subclasses can override this method
        to provide custom behavior.

        :param response: The response from the agent.
        :return: A new query to continue the interaction or None to end it
        """
        return None

    async def run(self, query: str | None = None) -> None:
        """
        Run the agent in a loop, allowing for continuous interaction.

        This method will keep the agent running until `BaseAgent.interact()` returns
        None. This allows for a continuous interaction with the agent, where the agent
        can ask follow-up questions or provide additional information. The loop will
        continue until the agent has no more questions to ask or until
        the user decides to stop the interaction.

        :param query: The initial query to start the interaction (optional).
        :return: None
        """
        while True:
            response = await self.invoke(query)
            query = await self.interact(response)
            if query is None:
                break


class RAGMixin:
    """
    Agent mixin for integrating RAG (Retrieval-Augmented Generation) sources.

    This mixin allows the agent to use multiple RAG sources for
    document retrieval and generation. It provides a method to
    initialize the RAG sources and adds lookup functions for each
    source to the agent's toolkit.
    """

    rag_sources: dict[str, RAG]

    def rag_init(self, rag_sources: dict[str, RAG]):
        """
        Initialize the RAG mixin.

        The provided dictionary of RAG sources is name → RAG instance, where
        "name" should be a single word describing the thing to look up
        (for example "movie", "person", etc.)

        :param rag_sources: Dictionary of RAG sources to initialize.
        """
        self.rag_sources = rag_sources

        for name, rag in rag_sources.items():

            def lookup_func(query):
                f"""
                Look up {name} in the database.

                :param query: The query to look up
                :return: Results matching the provided query
                """
                return rag(
                    f"Look up {query} and give me all the info you have about it",
                )

            lookup_func.__name__ = "lookup_" + name

            self.__doc__ += "\n\nWhen asked about {name}, use the provided tool `lookup_{name}` to look it up."
            self.add_tool(lookup_func.__name__, lookup_func)


class SimpleRAGAgent(RAGMixin, BaseAgent):
    """
    Simple RAG agent that uses a single RAG source for document retrieval.

    This agent is designed to work with a single RAG source and provides
    a simple interface for querying the source and generating responses.

    The `rag_name` attribute must be set to the name of the source/object to look up
    (for example "movie"; see `RAGMixin` for details).

    See `BaseAgent` for more details on how to use the agent.
    """

    rag_name: str

    def __init__(self, llm: LLM, rag: RAG, **kwargs: Any):
        """
        Construct a new SimpleRAGAgent instance.

        :param llm: The LLM instance to use for generating responses.
        :param rag: The RAG instance to use for document retrieval.
        :param kwargs: Additional keyword arguments for the system template
        """
        super().__init__(llm, **kwargs)

        name = getattr(self, "rag_name", None)
        if not name:
            raise ValueError(
                f"{self.__class__.__name__}.rag_name must be set to the name of the source/object to look up"
            )
        self.rag_init({name: rag})

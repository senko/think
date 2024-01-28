from logging import getLogger
from os import getenv
import time
from typing import Callable, Optional

from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletion

from ..chat import Chat

log = getLogger(__name__)


class ToolError(Exception):
    def __init__(self, message: str):
        self.message = message


class ChatGPT:
    """
    Client for OpenAI ChatGPT API.
    """

    MODELS = [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-4",
        "gpt-4-32k",
        "gpt-4-1106-preview",
        "gpt-4-vision-preview",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-1106-preview",
        temperature: float = 0.7,
        timeout: int = 120.0,
        max_retries: int = 3,
    ):
        """
        Create a new ChatGPT client instance.

        :param api_key: OpenAI API key (default: use from OPENAI_API_KEY env var).
        :param model: Model to use (default: gpt-3.5-turbo).
        :param temperature: Temperature parameter (default: 0.7).
        :param timeout: Timeout for API requests (default: 120.0).
        :param max_retries: Maximum number of retries for API requests (default: 3).
        """
        if api_key is None:
            api_key = getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OpenAI API key is not set")

        if model not in self.MODELS:
            raise ValueError(
                f"Unsupported model: {model} (supported: {','.join(self.MODELS)})"
            )

        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def _run_tool(
        self,
        function_call,
        tools,
    ):
        assert tools, "Tools should not be empty/missing at this point"

        function_name = function_call.name

        available_tools = {t.__name__: t for t in tools}
        available_tool_list = ",".join(available_tools.keys())
        if function_name not in available_tools:
            log.warning(
                f"GPT requested unknown tool: {function_name}; available tools: {available_tool_list}"
            )
            return f"ERROR: Unknown tool: {function_name}; available tools: {available_tool_list}"

        t = available_tools[function_name]
        try:
            args = t._validate_arguments(function_call.arguments)
        except TypeError as err:
            log.warning(
                f"Invalid arguments for GPT tool {function_name}: {err}", exc_info=True
            )
            return f"ERROR: {err}"

        try:
            log.debug(f"Running GPT tool {function_name} with args: {args}")
            return t(**args)
        except Exception as err:
            log.warning(f"Error running GPT tool {function_name}: {err}", exc_info=True)
            raise ToolError(f"Error running GPT tool {function_name}: {err}") from err

    def _call_chatgpt(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list] = None,
    ) -> ChatCompletion:
        try:
            log.debug(f"Calling ChatGPT with messages: {messages})")
            api_kwargs = dict(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
            )
            if tools:
                functions = [t._get_json_schema() for t in tools]
                api_kwargs["functions"] = functions
                api_kwargs["function_call"] = "auto"

            t0 = time.time()
            api_result = self.client.chat.completions.create(**api_kwargs)
            t1 = time.time()
            log.debug(
                f"ChatGPT request completed in {t1 - t0:.2f}s, {api_result.usage.total_tokens} tokens used"
            )
            return api_result.choices[0].message
        except OpenAIError as err:
            log.warning(f"Error calling ChatGPT: {err}", exc_info=True)
            raise

    def __call__(
        self,
        chat: Chat,
        tools: Optional[list] = None,
        parser: Optional[Callable] = None,
        max_iterations: int = 5,
    ) -> Optional[str]:
        chat = chat.fork()

        for i in range(max_iterations):
            response = self._call_chatgpt(list(chat), tools)
            if response.function_call:
                chat.assistant(f"Using tool '{response.function_call.name}'")
                try:
                    result = self._run_tool(response.function_call, tools)
                except ToolError as err:
                    print(repr(err.__cause__), type(err.__cause__))
                    raise err.__cause__
                chat.function(result, name=response.function_call.name)
                continue

            if parser:
                try:
                    content = parser(response.content)
                except ValueError as err:
                    log.debug(f"Error parsing GPT response: {err}", exc_info=True)
                    chat.assistant(response.content)
                    chat.user(
                        f"Error parsing response: {err}. Please output your response EXACTLY as requested."
                    )
                    continue
            else:
                content = response.content

            return content

        return None

from json import dumps

from pydantic import BaseModel

from .llm.base import LLM
from .llm.chat import Chat
from .prompt import JinjaStringTemplate


class LLMQuery(BaseModel):
    @classmethod
    async def run(cls, llm: LLM, **kwargs) -> "LLMQuery":
        """
        Run an LLM query using the provided LLM instance.

        The LLM will be prompted with the docstring of this class, which
        should contain a Jinja2 template that will be rendered with the
        provided keyword arguments. The LLM output will be parsed as JSON
        and validated against the JSON schema of this class.

        >>> class Joke(LLMQuery):
        ...    "Tell me a joke about {{ topic }} in Q / A format."
        ...    question: str
        ...    answer: str
        ...
        >>> joke = await Joke.run(llm, topic="chickens")

        :param llm: The LLM instance to use for the query.
        :param kwargs: Keyword arguments to render the docstring template.
        :return: An instance of this class with the LLM output as attributes.

        Note: use this only when requiring the LLM output to be parsed into
        a structure. If you only need the string output, use the `ask` function
        instead.

        Note: some models may require an explicit example of the output
        format in the prompt. If so, include one output example in the docstring.

        """
        template = cls.__doc__
        if not template:
            raise ValueError("LLMQuery must have a docstring")

        schema = cls.model_json_schema()
        schema.pop("description", None)

        tpl = JinjaStringTemplate()
        prompt = tpl(cls.__doc__, **kwargs)
        prompt += (
            "\n\nIMPORTANT: You must respond with a JSON object conforming to this JSON schema: "
            + dumps(schema)
            + "\nDo not add any additional text to the response - only respond with the JSON object."
        )
        c = Chat(prompt)
        return await llm(c, parser=cls)


async def ask(llm: LLM, prompt: str, **kwargs) -> str:
    """
    Ask a question using the provided LLM instance.

    The LLM will be prompted with the provided question, which may contain
    Jinja2 template variables that will be rendered with the provided keyword
    arguments. The LLM output will be returned as a string.

    :param llm: The LLM instance to use for the query.
    :param prompt: The question to ask the LLM.
    :param kwargs: Keyword arguments to render the prompt template.
    :return: The LLM output as a string.

    Note: use this when you only need the string output of the LLM. If you
    need the output to be parsed into a structure, use `LLMQuery.run()` instead.
    """
    tpl = JinjaStringTemplate()
    c = Chat().user(tpl(prompt, **kwargs))
    return await llm(c)

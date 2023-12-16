from pydantic import BaseModel
from decimal import Decimal, InvalidOperation

from .prompt import JinjaStringTemplate
from .parser import JSONParser
from .chat import Chat


def ai(fn):
    """
    Decorator for creating LLM-powered functions.

    The function docstring will be used as a Jinja2 template, rendered with
    the function arguments as context, and used as a prompt for the LLM.

    The function arguments and return value must be annotated with type hints.
    The return value may be a Pydantic model, in which case the LLM output
    will be parsed as JSON and validated against the model's schema. It may
    also be one of (int, float, bool, dict, Decimal), in which case the LLM
    output will be parsed as that type. If the return value is annotated as
    a string or some other type, no parsing will occur.

    The function body may be empty or contain "alignment code" that will be
    executed after the LLM output is received. The alignment code may raise
    an assertion error, which will be used to prompt the LLM to fix its output
    and try again.

    The function body is only executed if a special `_retval` argument is
    present in the function definition. This argument will be set to the parsed
    LLM output. If the argument is present and the function code is executed,
    it should return a vaule that will be used as the function's return value,
    which may or may not be what the LLM returned.

    If the argument is not present and the function code is not executed,
    the parsed LLM output will be returned automatically.

    :param fn: The function to decorate.
    :return: A decorated function.
    """

    if not fn.__doc__ or not fn.__doc__.strip():
        raise ValueError("The function must have a docstring")

    parser = None
    return_type = fn.__annotations__.get("return")
    return_type_prompt = ""

    # If the return type is something we know how to parse, set up the parser
    # and instructions for the LLM so it knows how it should format its output.
    if return_type:
        if issubclass(return_type, BaseModel):
            parser = JSONParser(spec=return_type)
            return_type_prompt = f"Your output must obey the following JSON schema: {return_type.model_json_schema()}"
        elif return_type in [int, float, bool, Decimal]:

            def parser(output):
                try:
                    return return_type(output)
                except (ValueError, TypeError, InvalidOperation) as err:
                    raise ValueError(
                        f"Error parsing '{output}' as {return_type.__name__}: {err}"
                    )

            return_type_prompt = (
                f"Your output must be parsable as Python type {return_type.__name__}"
            )

    # Combined parser that runs the LLM output through the parser and then
    # passes it to the function, with both the parser and the function
    # being optional. Using the combined parser allows us to use the built-in
    # LLM parsing logic for both the parser and the alignhment code.
    def align_parser(response, **kwargs):
        if parser:
            response = parser(response)

        if "_retval" not in fn.__annotations__:
            return response

        try:
            return fn(_retval=response, **kwargs)
        except AssertionError as e:
            raise ValueError(
                f"There's a problem with your output: {e}; please try again."
            )

    tpl = JinjaStringTemplate()

    def llm_wrapper(llm, **kwargs) -> str:
        prompt = tpl(fn.__doc__, **kwargs)
        if return_type_prompt:
            prompt += (
                "\n\nOutput ONLY the requested output in a single Markdown code block, without commentary or explanation. "
                + return_type_prompt
            )

        return llm(Chat(prompt), parser=lambda output: align_parser(output, **kwargs))

    return llm_wrapper

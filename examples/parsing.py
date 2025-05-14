# example: parsing.py
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

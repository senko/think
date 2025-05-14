# example: ask.py
from asyncio import run

from think import LLM, ask

llm = LLM.from_url("anthropic:///claude-3-haiku-20240307")


async def haiku(topic):
    return await ask(llm, "Write a haiku about {{ topic }}", topic=topic)


print(run(haiku("computers")))

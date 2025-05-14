# example: tools.py
from asyncio import run
from datetime import date

from think import LLM, Chat

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


print(run(days_to_xmas()))

# example: agent.py
from asyncio import run
from datetime import datetime

from think import LLM
from think.agent import BaseAgent, tool

llm = LLM.from_url("openai:///gpt-4o-mini")


class Chatbot(BaseAgent):
    """You are a helpful assistant. Today is {{today}}."""

    @tool
    def get_time(self) -> str:
        """Get the current time."""
        return datetime.now().strftime("%H:%M")

    async def interact(self, response: str) -> str:
        print(response)
        return input("> ").strip()


agent = Chatbot(llm, today=datetime.today())
run(agent.run())

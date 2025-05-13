#!/usr/bin/env python3
from asyncio import run
import sys
import os
from typing import Optional

import click
from dotenv import load_dotenv

# Add parent directory to path for easy local development
sys.path.append(".")

from think import LLM
from think.agent import BaseAgent

load_dotenv()


class Chatbot(BaseAgent):
    """You are a helpful assistant. Be concise and friendly in your responses."""

    async def interact(self, response: str) -> str:
        print("AI: ", response, "\n")

        # Wait for user input
        while True:
            user_input = input("> ").strip()
            if user_input:
                break

        # Check for exit command
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Goodbye!")
            return None

        return user_input


@click.command()
@click.option(
    "--model-url",
    "-m",
    default=None,
    help="LLM URL (e.g., 'openai:///gpt-4o-mini'). Defaults to LLM_URL env variable.",
)
@click.option(
    "--system",
    "-s",
    default="You are a helpful assistant. Be concise and friendly in your responses.",
    help="System prompt to initialize the chat.",
)
def main(model_url: Optional[str], system: str):
    """
    Interactive chatbot using the Think library.

    Start a conversation with an LLM in your terminal. Type your messages
    and receive AI responses. Use Ctrl+C or type 'exit' to end the conversation.
    """
    # Get model URL from argument or environment
    model_url = model_url or os.environ.get("LLM_URL")
    if not model_url:
        print(
            "Error: Model URL not provided. Use --model-url option or set LLM_URL environment variable."
        )
        sys.exit(1)

    try:
        # Initialize LLM from URL
        llm = LLM.from_url(model_url)
        print(f"Connected to {model_url}")
        print("Type your messages (type 'exit' to quit)")
        print("-" * 50)

        agent = Chatbot(llm=llm, system=system)
        run(agent.run("Hello!"))

    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

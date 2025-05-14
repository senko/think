# example: vision.py
from asyncio import run

from think import LLM, Chat

llm = LLM.from_url("openai:///gpt-4o-mini")


async def describe_image(path):
    image_data = open(path, "rb").read()
    chat = Chat().user("Describe the image in detail", images=[image_data])
    return await llm(chat)


print(run(describe_image("path/to/image.jpg")))

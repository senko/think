# example: pdf.py
from asyncio import run

from think import LLM, Chat

llm = LLM.from_url("google:///gemini-2.0-flash")


async def read_pdf(path):
    pdf_data = open(path, "rb").read()
    chat = Chat().user("Read the document", documents=[pdf_data])
    return await llm(chat)


print(run(read_pdf("path/to/document.pdf")))

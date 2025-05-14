# example: rag.py
from asyncio import run

from think import LLM
from think.rag.base import RAG, RagDocument

llm = LLM.from_url("openai:///gpt-4o-mini")
rag = RAG.for_provider("txtai")(llm)


async def index_documents():
    data = [
        RagDocument(id="a", text="Titanic: A sweeping romantic epic"),
        RagDocument(id="b", text="The Godfather: A gripping mafia saga"),
        RagDocument(id="c", text="Forrest Gump: A heartwarming tale of a simple man"),
    ]
    await rag.add_documents(data)


run(index_documents())
query = "A movie about a ship that sinks"
result = run(rag(query))
print(result)

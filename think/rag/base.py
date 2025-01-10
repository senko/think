from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypedDict, TypeVar

from ..ai import ask
from ..llm.base import LLM

PreparedQueryT = TypeVar("PreparedQueryT")


class RagDocument(TypedDict):
    id: str
    text: str


@dataclass
class RagResult:
    doc: RagDocument
    score: float


BASE_ANSWER_PROMPT = """Based ONLY on the provided context:

{% for item in results %}
{{ item.doc.text }}
Score: {{ item.score | round(3) }}
{% if not loop.last %}
---
{% endif %}
{% endfor %}

Answer the question:

{{query}}

(Note: don't say "based on provided context" in the output, it's confusing for the reader.)
"""


class RAG(ABC):
    PROVIDERS = ["txtai", "chroma", "pinecone"]
    QUERY_PROMPT: str | None = None
    ANSWER_PROMPT: str = BASE_ANSWER_PROMPT

    def __init__(
        self,
        llm: LLM,
    ):
        self.llm = llm

    @abstractmethod
    async def add_documents(self, documents: list[RagDocument]):
        """
        Add documents to the RAG index.

        :param documents: Documents to add.
        """

    @abstractmethod
    async def remove_documents(self, ids: list[str]):
        """
        Remove documents from the RAG index.

        :param ids: Document IDs to remove.
        """

    async def prepare_query(self, query: str) -> PreparedQueryT:
        """
        Process user input into query suitable for semantic search.

        :param query: User input.
        :return: Query suitable for semantic search.
        """
        if self.QUERY_PROMPT is None:
            return query
        return await ask(self.llm, self.QUERY_PROMPT, query=query)

    @abstractmethod
    async def fetch_results(
        self,
        user_query: str,
        prepared_query: PreparedQueryT,
        limit: int,
    ) -> list[RagResult]:
        """
        Use the provided and processed query to search for relevant context.

        :param user_query: Unprocessed user input.
        :param prepared_query: Processed user input.
        :param limit: Maximum number of search results to return.
        :return: Search results to include in the context.
        """

    async def get_answer(self, query: str, results: list[RagResult]) -> str:
        """
        Ask the LLM to provide the answer based on the retrieved context
        and the user's original query.

        :param query: User input.
        :param results: Search results.
        :return: Answer to the user query.
        """
        return await ask(self.llm, self.ANSWER_PROMPT, results=results, query=query)

    async def __call__(self, query: str, limit: int = 10) -> str:
        prepared_query = await self.prepare_query(query)
        results = await self.fetch_results(query, prepared_query, limit)
        return await self.get_answer(query, results)

    @abstractmethod
    async def count(self) -> int:
        """
        Get the number of documents in the RAG index.

        :return: The number of documents in the RAG index.
        """

    @classmethod
    def for_provider(cls, provider: str) -> type["RAG"]:
        """
        Get the RAG class for the specified provider/engine.

        :param provider: The provider name
        :return The RAG class for the provider

        Raises a ValueError if the provider is not supported.
        The list of supported providers is available in the
        PROVIDERS class attribute.
        """
        if provider == "txtai":
            from .txtai_rag import TxtAiRag

            return TxtAiRag

        elif provider == "chroma":
            from .chroma_rag import ChromaRag

            return ChromaRag

        elif provider == "pinecone":
            from .pinecone_rag import PineconeRag

            return PineconeRag

        else:
            raise ValueError(f"Unknown provider: {provider}")

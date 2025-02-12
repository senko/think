import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypedDict, Any

from think.ai import ask
from think.llm.base import LLM


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
        **kwargs: Any,
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

    async def prepare_query(self, query: str) -> str:
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
        prepared_query: str,
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

    async def rerank(self, results: list[RagResult]) -> list[RagResult]:
        """
        Rerank the search results based on the user query.

        :param results: Search results.
        :return: Reranked search results
        """
        return results

    async def __call__(self, query: str, limit: int = 10) -> str:
        prepared_query = await self.prepare_query(query)
        results = await self.fetch_results(query, prepared_query, limit)
        reranked_results = await self.rerank(results)
        return await self.get_answer(query, reranked_results)

    @abstractmethod
    async def calculate_similarity(self, query: str, docs: list[str]) -> list[float]:
        """
        Calculate the similarity between the query and the text.

        :param query: User input.
        :param text: Text to compare.
        :return: Similarity score.
        """

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

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]):
        """
        Compute cosine similarity between two vectors of equal length.

        :param a: First vector
        :param b: Second vector
        :return: Cosine similarity between the two vectors
        """
        if len(a) != len(b):
            raise ValueError("Vectors must be of equal length")

        # Compute dot product
        dot_product = sum(x * y for x, y in zip(a, b))

        # Compute magnitudes
        magnitude1 = math.sqrt(sum(x * x for x in a))
        magnitude2 = math.sqrt(sum(x * x for x in b))

        # Prevent division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        # Compute cosine similarity
        return dot_product / (magnitude1 * magnitude2)

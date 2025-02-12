from os import getenv
from typing import List, Optional

from ..llm.base import LLM
from .base import RAG, RagDocument, RagResult

try:
    from pinecone.grpc import PineconeGRPC as Pinecone
except ImportError as err:
    raise ImportError(
        "Pinecone requires the pinecone-client library: pip install pinecone-client"
    ) from err


class PineconeRag(RAG):
    DEFAULT_EMBEDDING_MODEL = "multilingual-e5-large"
    DEFAULT_EMBED_BATCH_SIZE = 96

    def __init__(
        self,
        llm: LLM,
        *,
        index_name: str,
        api_key: Optional[str] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    ):
        super().__init__(llm)
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.embed_batch_size = embed_batch_size

        # Initialize Pinecone client
        self.api_key = api_key or getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Pinecone API key must be provided either through constructor "
                "or PINECONE_API_KEY environment variable"
            )

        try:
            self.client = Pinecone(api_key=self.api_key)
            self.index = self.client.Index(self.index_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinecone client: {e}") from e

    async def _embed_texts(
        self, texts: List[str], is_query: bool = False
    ) -> List[dict]:
        """Helper method to embed texts using Pinecone's inference service."""
        try:
            input_type = "query" if is_query else "passage"

            batches = [
                texts[i : i + self.embed_batch_size]
                for i in range(0, len(texts), self.embed_batch_size)
            ]

            embeddings = []
            for batch in batches:
                # TODO: how to handle errors here?
                batch_result = self.client.inference.embed(
                    model=self.embedding_model,
                    inputs=batch,
                    parameters={"input_type": input_type, "truncate": "END"},
                )
                embeddings.extend(batch_result)

            return embeddings
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {e}") from e

    async def add_documents(self, documents: list[RagDocument]):
        try:
            # Generate embeddings for all documents
            texts = [doc["text"] for doc in documents]
            embeddings = await self._embed_texts(texts)

            # Prepare records for Pinecone
            records = []
            for doc, embedding in zip(documents, embeddings):
                records.append(
                    {
                        "id": doc["id"],
                        "values": embedding["values"],
                        "metadata": {"text": doc["text"]},
                    }
                )

            self.index.upsert(vectors=records)
        except Exception as e:
            raise RuntimeError(f"Failed to add documents to Pinecone: {e}") from e

    async def remove_documents(self, ids: list[str]):
        try:
            self.index.delete(ids=ids)
        except Exception as e:
            raise RuntimeError(f"Failed to remove documents from Pinecone: {e}") from e

    async def fetch_results(
        self,
        user_query: str,
        prepared_query: str,
        limit: int,
    ) -> list[RagResult]:
        try:
            embedding = await self._embed_texts([prepared_query], is_query=True)
            vector = embedding[0]["values"]
        except Exception as e:
            raise RuntimeError(f"Failed to prepare query: {e}") from e

        try:
            response = self.index.query(
                vector=vector,
                top_k=limit,
                include_metadata=True,
            )

            results = []
            for match in response["matches"]:
                results.append(
                    RagResult(
                        doc={"id": match["id"], "text": match["metadata"]["text"]},
                        score=match["score"],
                    )
                )

            return results
        except Exception as e:
            raise RuntimeError(f"Failed to fetch results from Pinecone: {e}") from e

    async def count(self) -> int:
        try:
            return self.index.describe_index_stats()["total_vector_count"]
        except Exception as e:
            raise RuntimeError(f"Failed to count records in Pinecone: {e}") from e

    async def calculate_similarity(self, query: str, docs: list[str]) -> list[float]:
        vectors = await self._embed_texts([query] + docs)
        query_vector, *doc_vectors = vectors
        similarities = []
        for doc_vector in doc_vectors:
            similarities.append(
                self._cosine_similarity(
                    query_vector["values"],
                    doc_vector["values"],
                )
            )
        return similarities

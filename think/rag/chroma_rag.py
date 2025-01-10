from pathlib import Path

from ..llm.base import LLM
from .base import RAG, RagDocument, RagResult

try:
    import chromadb
except ImportError as err:
    raise ImportError(
        "ChromaDB embeddings require the chromadb library: pip install chromadb"
    ) from err


class ChromaRag(RAG):
    def __init__(
        self,
        llm: LLM,
        *,
        collection: str,
        path: Path | str | None = None,
    ):
        super().__init__(llm)
        self.collection_name = collection
        self.path = None if path is None else Path(path)

        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.path))
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    async def add_documents(self, documents: list[RagDocument]):
        # Extract document data
        ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]

        self.collection.add(
            documents=texts,
            ids=ids,
        )

    async def remove_documents(self, ids: list[str]):
        self.collection.delete(ids=ids)

    async def fetch_results(
        self, user_query: str, prepared_query: str, limit: int
    ) -> list[RagResult]:
        results = self.collection.query(
            query_texts=[prepared_query],
            n_results=limit,
        )

        documents = []
        for doc_id, text, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["distances"][0],
        ):
            score = 1.0 - distance

            documents.append(
                RagResult(
                    doc={"id": doc_id, "text": text},
                    score=score,
                ),
            )

        return documents

    async def count(self) -> int:
        return self.collection.count()

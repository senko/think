from pathlib import Path

from ..llm.base import LLM
from .base import RAG, RagDocument, RagResult

try:
    from txtai import Embeddings
except ImportError as err:
    raise ImportError(
        "Txtai embeddings require the txtai library: pip install txtai"
    ) from err


class TxtAiRag(RAG):
    DEFAULT_EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        llm: LLM,
        *,
        model: str = DEFAULT_EMBEDDINGS_MODEL,
        path: Path | str | None = None,
    ):
        super().__init__(llm)
        self.model = model
        self.path = None if path is None else Path(path)

        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)

        self.embeddings = Embeddings(
            {
                "path": self.model,
                "content": True,
            }
        )
        if self.path:
            if (self.path / "embeddings").exists():
                print("Loading from", self.path)
                self.embeddings.load(str(self.path))
            else:
                self.embeddings.save(str(self.path))

    async def add_documents(self, documents: list[RagDocument]):
        data = [(doc["id"], doc["text"]) for doc in documents]
        self.embeddings.upsert(data)
        if self.path:
            self.embeddings.save(str(self.path))

    async def remove_documents(self, ids: list[str]):
        self.embeddings.delete(ids)
        if self.path:
            self.embeddings.save(str(self.path))

    async def count(self) -> int:
        return self.embeddings.count()

    async def fetch_results(
        self, user_query: str, prepared_query: str, limit: int
    ) -> list[RagResult]:
        results = self.embeddings.search(prepared_query, limit=limit)
        return [
            RagResult(
                doc={"id": result["id"], "text": result["text"]},
                score=result["score"],
            )
            for result in results
        ]

    async def calculate_similarity(self, query: str, docs: list[str]) -> list[float]:
        return self.embeddings.similarity(query, docs)

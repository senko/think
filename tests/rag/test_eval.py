import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from think.rag.base import RagResult, RagDocument
from think.rag.eval import RagEval


@pytest.mark.asyncio
@patch("think.rag.eval.ask", new_callable=AsyncMock)
async def test_context_precision(ask):
    llm = MagicMock()
    rag = AsyncMock()
    rag.fetch_results.return_value = [
        RagResult(doc=RagDocument(id="A", text="A"), score=0.5),
        RagResult(doc=RagDocument(id="B", text="B"), score=0.5),
        RagResult(doc=RagDocument(id="C", text="C"), score=0.5),
        RagResult(doc=RagDocument(id="D", text="D"), score=0.5),
    ]
    ask.side_effect = ["yes", "no", "yes", "no"]

    eval = RagEval(rag, llm)

    query = "A movie about a ship that sinks"
    ctx_precision = await eval.context_precision(query, 2)

    assert ctx_precision == pytest.approx(1.33, rel=1e-2)


@pytest.mark.asyncio
@patch("think.rag.eval.ask", new_callable=AsyncMock)
async def test_context_recall(ask):
    llm = MagicMock()
    rag = AsyncMock()
    rag.fetch_results.return_value = [
        RagResult(doc=RagDocument(id="A", text="A"), score=0.5),
        RagResult(doc=RagDocument(id="B", text="B"), score=0.5),
        RagResult(doc=RagDocument(id="C", text="C"), score=0.5),
        RagResult(doc=RagDocument(id="D", text="D"), score=0.5),
    ]
    ask.side_effect = ["yes", "no", "yes", "no"]

    eval = RagEval(rag, llm)

    query = "A movie about a ship that sinks"
    reference = [
        "A ship sinks",
        "A love story",
        "A historical event",
        "A tragedy",
    ]
    ctx_recall = await eval.context_recall(query, reference, 4)

    assert ctx_recall == pytest.approx(0.5, rel=1e-2)


@pytest.mark.asyncio
@patch("think.rag.eval.ask", new_callable=AsyncMock)
async def test_faithfulness(ask):
    llm = MagicMock()
    rag = AsyncMock()
    rag.fetch_results.return_value = [
        RagResult(doc=RagDocument(id="A", text="A"), score=0.5),
        RagResult(doc=RagDocument(id="B", text="B"), score=0.5),
        RagResult(doc=RagDocument(id="C", text="C"), score=0.5),
        RagResult(doc=RagDocument(id="D", text="D"), score=0.5),
    ]
    ask.side_effect = [
        "The Titanic sank after hitting an iceberg.\n"
        + "Many people died.\n"
        + "It was a tragic event\n",
        "yes",
        "no",
        "yes",
    ]

    eval = RagEval(rag, llm)

    query = "What happened to the Titanic?"
    answer = "The Titanic sank after hitting an iceberg. Many people died. It was a tragic event."
    faithfulness_score = await eval.faithfulness(query, answer, 4)

    assert faithfulness_score == pytest.approx(0.67, rel=1e-2)


@pytest.mark.asyncio
@patch("think.rag.eval.ask", new_callable=AsyncMock)
async def test_answer_relevance(ask):
    llm = MagicMock()
    rag = AsyncMock()
    rag.calculate_similarity.return_value = [0.8, 0.75, 0.85]
    ask.side_effect = [
        "What is the fate of the Titanic?\n"
        + "How did the Titanic sink?\n"
        + "What happened to the Titanic?"
    ]

    eval = RagEval(rag, llm)

    query = "What happened to the Titanic?"
    answer = "The Titanic sank after hitting an iceberg. Many people died. It was a tragic event."
    relevance_score = await eval.answer_relevance(query, answer, 3)

    assert relevance_score == pytest.approx(0.8, rel=1e-2)

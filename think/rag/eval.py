"""
# RAG Evaluation

The `rag.eval` module provides tools for evaluating the performance of RAG systems.
It includes metrics for measuring different aspects of RAG quality and functionality.

## Basic Evaluation

```python
# example: rag_eval_basic.py
import asyncio
from think import LLM
from think.rag.base import RAG, RagDocument
from think.rag.eval import RagEval

# Set up the LLM and RAG system
llm = LLM.from_url("openai:///gpt-4o-mini")
rag = RAG.for_provider("txtai")(llm)

# Set up the evaluator
evaluator = RagEval(llm)

async def evaluate_rag():
    # Add test documents
    documents = [
        RagDocument(id="doc1", text="The Eiffel Tower is 330 meters tall and located in Paris, France."),
        RagDocument(id="doc2", text="The Great Wall of China is over 21,000 kilometers long."),
        RagDocument(id="doc3", text="The Grand Canyon is 446 km long and up to 29 km wide.")
    ]
    await rag.add_documents(documents)

    # Generate answer
    query = "How tall is the Eiffel Tower?"
    answer = await rag(query)

    # Evaluate answer
    precision = await evaluator.context_precision(query, rag.last_context, answer)
    relevance = await evaluator.answer_relevance(query, answer)

    print(f"Answer: {answer}")
    print(f"Context Precision: {precision}")
    print(f"Answer Relevance: {relevance}")

asyncio.run(evaluate_rag())
```

## Available Metrics

The RagEval class provides several metrics:

1. **Context Precision**: Measures if retrieved documents are relevant to the query
2. **Context Recall**: Measures if all relevant information is retrieved
3. **Faithfulness**: Evaluates if the answer is supported by the retrieved context
4. **Answer Relevance**: Assesses if the answer addresses the query

## Comprehensive Evaluation

```python
# example: rag_eval_comprehensive.py
import asyncio
from think import LLM
from think.rag.base import RAG, RagDocument
from think.rag.eval import RagEval

llm = LLM.from_url("openai:///gpt-4o-mini")
rag = RAG.for_provider("txtai")(llm)
evaluator = RagEval(llm)

async def comprehensive_eval():
    # Add documents (assume already done)

    # Define test cases
    test_cases = [
        {"query": "What are the dimensions of the Grand Canyon?", "ground_truth": "The Grand Canyon is 446 km long and up to 29 km wide."},
        {"query": "How tall is the Eiffel Tower?", "ground_truth": "The Eiffel Tower is 330 meters tall."}
    ]

    results = {}
    for tc in test_cases:
        query = tc["query"]
        ground_truth = tc["ground_truth"]

        # Get RAG answer
        answer = await rag(query)

        # Evaluate all metrics
        metrics = {
            "precision": await evaluator.context_precision(query, rag.last_context, answer),
            "recall": await evaluator.context_recall(query, rag.last_context, ground_truth),
            "faithfulness": await evaluator.faithfulness(rag.last_context, answer),
            "relevance": await evaluator.answer_relevance(query, answer)
        }

        results[query] = {"answer": answer, "metrics": metrics}

    # Print results
    for query, result in results.items():
        print(f"Query: {query}")
        print(f"Answer: {result['answer']}")
        print(f"Metrics: {result['metrics']}")
        print()

asyncio.run(comprehensive_eval())
```

See also:
- [RAG Base Functionality](#rag-base-functionality) for RAG implementation details
- [Basic LLM Use](#basic-llm-use) for general LLM interaction
"""

from ..ai import ask
from ..llm.base import LLM
from .base import RAG


class RagEval:
    """
    Evaluation system for RAG (Retrieval-Augmented Generation) systems.

    This class provides comprehensive evaluation metrics for assessing the quality
    of RAG systems, including context precision, context recall, faithfulness,
    and answer relevance. It uses an LLM to evaluate various aspects of the
    retrieval and generation process.

    The evaluation metrics are based on established RAG evaluation frameworks
    and provide quantitative measures of system performance.

    Key metrics:
    - Context Precision: How relevant are the retrieved documents?
    - Context Recall: How well does retrieval cover ground truth?
    - Faithfulness: Are answers supported by retrieved context?
    - Answer Relevance: How relevant are answers to queries?

    Example usage:
        evaluator = RagEval(rag_system, llm)
        precision = await evaluator.context_precision("query", n_results=10)
        recall = await evaluator.context_recall("query", reference_text)
    """

    CONTEXT_PRECISION_PROMPT = """
    You're tasked with evaluating a knowledge retrieval system. For a user query, you're
    given a document retrieved by the system. Based on the document alone, you need to
    determine if it's relevant to the query.

    User query: {{ query }}

    Document: {{ result }}

    Answer with "yes" if the document is relevant to the query, or "no" otherwise.
    """

    CLAIM_SPLIT_PROMPT = """
    You're tasked with evaluating a knowledge retrieval system. Given a
    {% if is_reference %}ground truth (reference){% else %}system output (answer){% endif %} text,
    your task is to split it into individual claims.

    Example:
    > Text: "The quick brown fox jumps over the lazy dog."
    > Claims:
    > The fox is quick and brown.
    > The fox jumps over the dog.
    > The dog is lazy.

    Note: do not extract trivial claims like "the fox exists".

    Here's the {% if is_reference %}ground truth (reference){% else %}system output (answer){% endif %} text:
    {{ text }}

    Please split it into individual claims. Separate each claim with a newline. Do not include
    any comments, explanations, or additional information - you must only output the claims themselves.
    """

    CONTEXT_RECALL_PROMPT = """
    You're tasked with evaluating a knowledge retrieval system. For a specific fact or claim,
    you're given a set of documents retrieved by the system. Based on the documents alone,
    you need to determine if this claim is supported by the documents.

    Claim: {{ claim }}

    Supporting documents:

    {% for result in results %}
    * {{ result.doc.text }}
    {% endfor %}

    Answer with "yes" if the claim is supported by the documents, or "no" otherwise.
    """

    GENERATE_QUESTIONS_PROMPT = """
    You're tasked with evaluating a knowledge retrieval system. Given a system output (answer),
    your task is to generate a set of {{ n_questions }} questions
    that the answer is a suitable response for.

    Example:
    > Answer: "Paris is the capital of France."
    > Questions:
    > Which city is the capital of France?
    > Paris is the capital of which country?
    > ...

    Here is the system output (answer): {{ answer }}

    Please generate {{ n_questions }} questions that this answer is a suitable response for.

    Please output each question on a separate line (ie. separate them by newlines). Do not
    include any comments, explanations, or additional information - you must only output the
    generated questions.
    """

    def __init__(self, rag: RAG, llm: LLM):
        """
        Initialize the RAG evaluation class.

        :param rag: The RAG system to evaluate.
        :param llm: The LLM used for evaluation.
        """
        self.rag = rag
        self.llm = llm

    async def context_precision(
        self,
        query: str,
        n_results: int = 10,
    ) -> float:
        """
        Calculate Precision@k and average context precision @k.

        This metric indicates how well the system is able to retrieve relevant documents
        for a given query.

        Precision at rank k is the number of relevant documents retrieved in the top k
        results divided by k. Context precision (or average precision) at rank k is
        the average precision at each rank k.

        More info: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/

        :param query: The query to evaluate.
        :param n_results: The number of results to evaluate.
        :return: The average context precision at rank k, in the range [0, 1].
        """
        prepared_query = await self.rag.prepare_query(query)
        results = await self.rag.fetch_results(query, prepared_query, n_results)

        n_relevant = 0
        ctx_precision = 0

        for i, result in enumerate(results):
            r = await ask(
                self.llm,
                self.CONTEXT_PRECISION_PROMPT,
                query=query,
                result=result.doc["text"],
            )
            if "yes" in r.lower():
                n_relevant += 1

            k = i + 1  # rank
            precision_k = n_relevant / k
            ctx_precision += precision_k

        return ctx_precision / n_results

    async def split_into_claims(
        self,
        text: str,
        is_reference: bool = False,
    ) -> list[str]:
        """
        Split the text into individual constituent claims.

        The text can be reference text (ground truth) or system output (answer).

        :param text: The text to split into claims.
        :param is_reference: Whether the text is reference text (ground truth).
        :return: A list of claims.
        """
        r = await ask(
            self.llm,
            self.CLAIM_SPLIT_PROMPT,
            text=text,
            is_reference=is_reference,
        )

        return [line.strip() for line in r.split("\n") if line.strip()]

    async def _supported_by_claims(
        self,
        query: str,
        reference: str | list[str],
        is_reference: bool,
        n_results: int,
    ) -> float:
        if isinstance(reference, str):
            reference_claims = await self.split_into_claims(
                reference,
                is_reference=is_reference,
            )
        elif isinstance(reference, list) and len(reference) > 0:
            reference_claims = reference
        else:
            raise ValueError(
                "Reference must be a string or a non-empty list of strings."
            )

        prepared_query = await self.rag.prepare_query(query)
        results = await self.rag.fetch_results(query, prepared_query, n_results)

        n_supported = 0

        for claim in reference_claims:
            r = await ask(
                self.llm,
                self.CONTEXT_RECALL_PROMPT,
                claim=claim,
                results=results,
            )
            if "yes" in r.lower():
                n_supported += 1

        return n_supported / len(reference_claims)

    async def context_recall(
        self,
        query: str,
        reference: str | list[str],
        n_results: int = 10,
    ) -> float:
        """
        Calculate Context Recall

        This metric indicates how well the system is able to retrieve documents that
        support the ground truth (reference).

        This is estimated by checking if all the ground truth claims (list of reference claims)
        is supported by the retrieved documents.

        More info: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/

        :param query: The query to evaluate.
        :param reference: The ground truth claims.
        :param n_results: The number of results to use in the context.
        :return: The context recall score, in the range [0, 1].
        """
        return await self._supported_by_claims(
            query,
            reference,
            is_reference=True,
            n_results=n_results,
        )

    async def faithfulness(
        self,
        query: str,
        answer: str,
        n_results: int = 10,
    ) -> float:
        """
        Calculate answer Faithfulness

        This metric indicates how well the system is able to provide answers that are
        supported by the retrieved documents.

        The answer is split into constituent claims, and each claim is checked to see if
        it is supported by the retrieved documents.

        More info: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/

        :param query: The query to evaluate.
        :param answer: The system output / answer.
        :param n_results: The number of results to use in the context.
        :return: The faithfulness score, in the range [0, 1].
        """
        return await self._supported_by_claims(
            query,
            answer,
            is_reference=False,
            n_results=n_results,
        )

    async def answer_relevance(
        self,
        query: str,
        answer: str,
        n_questions: int = 3,
    ) -> float:
        """
        Calculate Answer (Response) Relevance

        This metrics indicates how relevant the answer is to the user query.

        The answer is used to generate a set of artificial questions the answer
        is a suitable response for. The questions are embedded and the cosine
        similarity between the query and each question is calculated.

        The score is the average similarity across all the generated questions.

        More info: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/

        :param query: The user query.
        :param answer: The system output / answer.
        :param n_questions: The number of questions to generate.
        """

        r = await ask(
            self.llm,
            self.GENERATE_QUESTIONS_PROMPT,
            answer=answer,
            n_questions=n_questions,
        )
        questions = [line.strip() for line in r.split("\n") if line.strip()]

        similarities = await self.rag.calculate_similarity(query, questions)
        return sum(similarities) / n_questions

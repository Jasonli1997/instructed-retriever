"""Instruction-aware reranking module for Instructed Retriever."""

import abc
import json
import logging
import time
from typing import Optional

import dspy
import mlflow
import requests  # type: ignore[import-untyped]
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from pydantic import BaseModel, Field

from instructed_retriever.agent.dspy.instructions import SystemSpecifications
from instructed_retriever.agent.dspy.schemas import CustomDocument

logger = logging.getLogger(__name__)

BATCH_SIZE = 5
STATUS_CODE_OK = 200


class BatchDocumentScore(BaseModel):
    """Score and reasoning for a single document within a batch."""

    index: int = Field(..., description="1-based index of the document in the batch")
    score: float = Field(..., description="Relevance score from 0.0 to 1.0")
    reasoning: str = Field(..., description="Brief explanation of the relevance score")


class BatchRerankerSignature(dspy.Signature):
    """Signature for instruction-aware batch document reranking."""

    query: str = dspy.InputField(description="The user query")
    documents: str = dspy.InputField(
        description="Numbered list of documents to score, formatted as 'Document N: <text>'"
    )
    anchor_reference: str = dspy.InputField(
        description=(
            "Calibration reference for scoring. If provided, the first document in the list "
            "has already been scored — use its score as a baseline when scoring the rest. "
            "Empty string means no calibration reference (first batch)."
        ),
        default="",
    )
    system_specifications: str = dspy.InputField(
        description="System specifications and instructions to consider",
        default="",
    )
    document_scores: list[BatchDocumentScore] = dspy.OutputField(
        description=(
            "A relevance score and reasoning for each document that needs scoring "
            "(all documents when no anchor_reference is given, or all except the first "
            "when anchor_reference is provided). Scores 0.0–1.0."
        )
    )


class Reranker(abc.ABC):
    """Abstract base class for rerankers."""

    @abc.abstractmethod
    def rerank_documents(
        self,
        query: str,
        documents: list[CustomDocument],
        system_specifications: Optional[SystemSpecifications],
    ) -> list[CustomDocument]:
        """Rerank the given documents and return the sorted list."""


class InstructedReranker(Reranker):
    """LLM-based instruction-aware reranker.

    Implements a sliding-window batch scoring strategy:
    - Documents are processed in fixed-size batches (``batch_size``).
    - The last document of each batch becomes the first document of the next batch,
      acting as a calibration anchor whose known score is passed to the LLM so that
      scores remain comparable across batches.
    - Only new (non-anchor) documents are scored in each batch after the first.

    Example with batch_size=4 and 8 documents:
        Batch 1: [d0, d1, d2, d3]             → scores s0..s3
        Batch 2: [d3(anchor, s3), d4, d5, d6] → LLM told "d3 = s3"; scores d4, d5, d6
        Batch 3: [d6(anchor, s6), d7]         → LLM told "d6 = s6"; scores d7
    """

    def __init__(self, model: str, batch_size: int = BATCH_SIZE) -> None:
        if batch_size < 2:
            raise ValueError(f"batch_size must be >= 2, got {batch_size}")
        self.model = model
        self.batch_size = batch_size
        self.reranker = dspy.ChainOfThought(BatchRerankerSignature)

    @mlflow.trace(span_type=SpanType.RERANKER, name="llm_rerank_retrievals")
    def rerank_documents(
        self,
        query: str,
        documents: list[CustomDocument],
        system_specifications: Optional[SystemSpecifications],
    ) -> list[CustomDocument]:
        if not documents:
            return documents

        if system_specifications and system_specifications.has_instructions():
            specs_context = system_specifications.to_retrieval_context()
        else:
            logger.warning(
                "No system specifications provided; reranking will be based on query relevance alone."
            )
            specs_context = ""

        anchor_doc: CustomDocument | None = None
        anchor_score: float | None = None

        new_doc_index = 0
        while new_doc_index < len(documents):
            max_new = self.batch_size - (1 if anchor_doc is not None else 0)
            batch_new = documents[new_doc_index : new_doc_index + max_new]
            batch = [anchor_doc] + batch_new if anchor_doc is not None else batch_new

            try:
                new_scores = self._score_batch(
                    query=query,
                    batch=batch,
                    specs_context=specs_context,
                    anchor_score=anchor_score,
                )
            except Exception as e:
                logger.warning("Batch scoring failed: %s. Assigning default scores.", e)
                new_scores = [(0.5, "Reranking failed, using default score")] * len(batch_new)

            for doc, (score, reasoning) in zip(batch_new, new_scores):
                doc.rerank_score = score
                doc.rerank_reasoning = reasoning

            if not batch_new:
                logger.warning(
                    "batch_new is unexpectedly empty at index=%d; stopping early.", new_doc_index
                )
                break
            anchor_doc = batch_new[-1]
            anchor_score = anchor_doc.rerank_score
            new_doc_index += len(batch_new)

        documents.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
        return documents

    @mlflow.trace(span_type=SpanType.TASK)
    def _score_batch(
        self,
        query: str,
        batch: list[CustomDocument],
        specs_context: str,
        anchor_score: float | None,
    ) -> list[tuple[float, str]]:
        formatted = "\n".join(
            f"Document {i + 1}: {doc.page_content}" for i, doc in enumerate(batch)
        )
        anchor_reference = (
            f"Document 1 has already been scored: {anchor_score:.3f}. "
            "Use this as a calibration reference."
            if anchor_score is not None
            else ""
        )

        with dspy.context(lm=dspy.LM(model=self.model)):
            pred = self.reranker(
                query=query,
                documents=formatted,
                anchor_reference=anchor_reference,
                system_specifications=specs_context,
            )

        score_map: dict[int, tuple[float, str]] = {}
        for item in pred.document_scores:
            clipped = max(0.0, min(1.0, item.score))
            score_map[item.index] = (clipped, item.reasoning)

        start_idx = 2 if anchor_score is not None else 1
        new_docs = batch[start_idx - 1 :]
        results: list[tuple[float, str]] = []
        for offset, _ in enumerate(new_docs):
            doc_idx = start_idx + offset
            if doc_idx in score_map:
                results.append(score_map[doc_idx])
            else:
                logger.warning("Missing score for document %s; using default 0.5", doc_idx)
                results.append((0.5, "Score not returned by reranker"))
        return results


class DatabricksReranker(Reranker):
    """Reranker that calls a Databricks-hosted reranker model endpoint.

    Authenticates via OAuth client-credentials flow using a Databricks service principal
    (``DATABRICKS_CLIENT_ID`` / ``DATABRICKS_CLIENT_SECRET``). The access token is cached
    in-memory and automatically refreshed when it expires.
    """

    _TOKEN_EXPIRY_BUFFER_SECS = 60

    def __init__(self, model: str, workspace_client: WorkspaceClient) -> None:
        host = workspace_client.config.host.rstrip("/")
        self.model_endpoint = f"{host}/serving-endpoints/{model}/invocations"
        self.token_endpoint = f"{host}/oidc/v1/token"
        self.client_id = workspace_client.config.client_id
        self.client_secret = workspace_client.config.client_secret
        self._access_token: str = ""
        self._token_expiry: float = 0.0

    def _fetch_token(self) -> str:
        response = requests.post(
            url=self.token_endpoint,
            auth=(self.client_id, self.client_secret),
            data={"grant_type": "client_credentials", "scope": "all-apis"},
            timeout=30,
        )
        if response.status_code != STATUS_CODE_OK:
            raise requests.exceptions.RequestException(
                f"Token request failed with status {response.status_code}: {response.text}"
            )
        payload = response.json()
        self._access_token = payload["access_token"]
        expires_in = int(payload.get("expires_in", 3600))
        self._token_expiry = time.monotonic() + expires_in - self._TOKEN_EXPIRY_BUFFER_SECS
        return self._access_token

    def _get_token(self) -> str:
        if not self._access_token or time.monotonic() >= self._token_expiry:
            self._fetch_token()
        return self._access_token

    @mlflow.trace(span_type=SpanType.RERANKER, name="databricks_rerank_retrievals")
    def rerank_documents(
        self,
        query: str,
        documents: list[CustomDocument],
        system_specifications: Optional[SystemSpecifications],
    ) -> list[CustomDocument]:
        queries = [query] * len(documents)
        data = {"inputs": {"query": queries, "context": [doc.page_content for doc in documents]}}

        headers = {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json",
        }
        response = requests.request(
            method="POST",
            headers=headers,
            url=self.model_endpoint,
            data=json.dumps(data),
            timeout=600,
        )

        if response.status_code != STATUS_CODE_OK:
            raise requests.exceptions.RequestException(
                f"Request failed with status {response.status_code}: {response.text}"
            )

        scores = [e["0"] for e in response.json()["predictions"]]
        for doc, score in zip(documents, scores):
            doc.rerank_score = score

        documents.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
        return documents

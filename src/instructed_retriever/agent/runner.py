import asyncio
import logging

import dspy
import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import DatabricksEmbeddings, DatabricksVectorSearch
from langchain_core.documents import Document as LangchainDocument
from mlflow.entities import Document as MLflowDocument
from mlflow.entities import SpanType

from instructed_retriever.agent.config import InstructedRetrieverConfiguration
from instructed_retriever.agent.context import RunContext
from instructed_retriever.agent.dspy.instructions import load_system_specifications
from instructed_retriever.agent.dspy.prompts import (
    ANSWER_GENERATION_PROMPT,
    CATEGORY_CLASSIFICATION_PROMPT,
    QUERY_REWRITE_PROMPT,
)
from instructed_retriever.agent.dspy.reranker import DatabricksReranker, Reranker
from instructed_retriever.agent.dspy.schemas import CustomDocument, StructuredQuery
from instructed_retriever.agent.dspy.signatures import (
    AnswerGeneratorSignature,
    CategoryClassifierSignature,
    QueryRewriterSignature,
)

# TODO: Make these configurable via config
TEXT_COLUMN = "content"
TITLE_COLUMN, CATEGORY_COLUMN = "title", "category"
LOCAL_TOP_K, GLOBAL_TOP_K = 5, 3
QUERY_TYPE = "HYBRID"
MAX_DECOMPOSED_QUERIES = 5
CACHE_CONTROL_INJECTION_POINTS = [{"location": "message", "role": "system"}]

dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)

logger = logging.getLogger(__name__)


class InstructedRetrieverRunner(dspy.Module):
    def __init__(self, config: InstructedRetrieverConfiguration) -> None:
        self.config = config

        self.system_specs = load_system_specifications(config.system_specs_path)
        other_columns = [f.name for f in self.system_specs.index_schema.fields]

        category_classifier_sig = CategoryClassifierSignature.with_instructions(
            CATEGORY_CLASSIFICATION_PROMPT
        )
        query_rewriter_sig = QueryRewriterSignature.with_instructions(QUERY_REWRITE_PROMPT)
        answer_generator_sig = AnswerGeneratorSignature.with_instructions(ANSWER_GENERATION_PROMPT)

        self.category_classifier = dspy.ChainOfThought(category_classifier_sig)
        self.query_rewriter = dspy.Predict(query_rewriter_sig)
        self.answer_generator = dspy.ChainOfThought(answer_generator_sig)

        workspace_client = WorkspaceClient(
            host=self.config.databricks_host,
            client_id=self.config.databricks_client_id,
            client_secret=self.config.databricks_client_secret.get_secret_value(),
        )
        embedding = DatabricksEmbeddings(endpoint=self.config.embedding_model)

        self.reranker: Reranker | None = None
        if config.enable_llm_reranking:
            from instructed_retriever.agent.dspy.reranker import InstructedReranker

            self.reranker = InstructedReranker(model=config.answer_generator_model)

            if config.reranker_model:
                logger.warning(
                    "Both LLM-based reranking and a separate reranker model are configured. "
                    "Set enable_llm_reranking=false to use the Databricks-hosted reranker, "
                    "or clear reranker_model to use LLM-based reranking exclusively."
                )

        elif config.reranker_model:
            self.reranker = DatabricksReranker(
                model=config.reranker_model, workspace_client=workspace_client
            )

        else:
            logger.info("No reranker configured; results will be ranked by vector similarity score.")

        self.vector_store = DatabricksVectorSearch(
            endpoint=self.config.vs_endpoint,
            index_name=self.config.vs_index_name,
            embedding=embedding,
            text_column=TEXT_COLUMN,
            columns=other_columns,
            workspace_client=workspace_client,
        )

    async def aforward(self, query: str, context: RunContext) -> dspy.Prediction:
        task_lm = dspy.LM(
            model=self.config.query_rewriter_model,
            cache_control_injection_points=CACHE_CONTROL_INJECTION_POINTS,
        )
        main_lm = dspy.LM(
            model=self.config.answer_generator_model,
            cache_control_injection_points=CACHE_CONTROL_INJECTION_POINTS,
        )

        specs_query_rewriter_context = specs_generation_context = specs_categorization_context = (
            None
        )

        if self.config.enable_instructed_retrieval:
            specs_query_rewriter_context = self.system_specs.to_query_rewriter_context()
            specs_generation_context = self.system_specs.to_generation_context()
            specs_categorization_context = self.system_specs.to_categorization_context()

        async def _rewrite() -> dspy.Prediction:
            with dspy.context(lm=task_lm):
                return await self.query_rewriter.acall(
                    system_specifications=specs_query_rewriter_context,
                    query=query,
                    history=context.chat_history,
                )

        if self.config.enable_instructed_retrieval:

            async def _classify() -> dspy.Prediction:
                with dspy.context(lm=task_lm):
                    return await self.category_classifier.acall(
                        system_specifications=specs_categorization_context,
                        query=query,
                        history=context.chat_history,
                    )

            rewrite_result, classify_result = await asyncio.gather(_rewrite(), _classify())
            category = classify_result.category or ""
        else:
            rewrite_result = await _rewrite()
            category = ""

        structured_queries: list[StructuredQuery] = rewrite_result.structured_queries[
            :MAX_DECOMPOSED_QUERIES
        ]

        formatted_docs = await self.retrieve_context(
            original_query=query,
            structured_queries=structured_queries,
            priority_data_source=category,
            use_filters=self.config.enable_instructed_retrieval,
        )

        prioritized_docs = [doc for doc in formatted_docs if doc.metadata.get("prioritized")]
        other_docs = [doc for doc in formatted_docs if not doc.metadata.get("prioritized")]

        formatted_prioritized = "\n\n".join(
            f"**Article #{i + 1}: {doc.metadata['doc_uri']}**\n{doc.page_content}"
            for i, doc in enumerate(prioritized_docs)
        )
        formatted_others = "\n\n".join(
            f"**Article #{i + 1}: {doc.metadata['doc_uri']}**\n{doc.page_content}"
            for i, doc in enumerate(other_docs)
        )

        with dspy.context(lm=main_lm):
            pred = await self.answer_generator.acall(
                system_specifications=specs_generation_context,
                prioritized_context=formatted_prioritized,
                other_context=formatted_others,
                history=context.chat_history,
                query=query,
            )
        pred.category = category
        return pred

    @mlflow.trace(span_type=SpanType.RETRIEVER)
    async def retrieve_context(
        self,
        original_query: str,
        structured_queries: list[StructuredQuery],
        priority_data_source: str,
        use_filters: bool = True,
    ) -> list[MLflowDocument]:
        retrieval_tasks = [
            asyncio.create_task(
                self.query_retriever(
                    sq.query_text,
                    filters=sq.to_databricks_filter() if use_filters else None,
                    priority_data_source=priority_data_source,
                )
            )
            for sq in structured_queries
        ]
        retrieval_results = await asyncio.gather(*retrieval_tasks)

        all_scored: list[tuple[CustomDocument, float]] = []
        for docs in retrieval_results:
            all_scored.extend(docs)

        # Deduplicate by content, accumulating scores for repeated chunks
        best: dict[str, tuple[CustomDocument, float]] = {}
        for doc, score in all_scored:
            if doc.page_content not in best:
                best[doc.page_content] = (doc, score)
            else:
                existing_doc, existing_score = best[doc.page_content]
                best[doc.page_content] = (existing_doc, existing_score + score)
        all_scored = list(best.values())

        if self.reranker:
            all_documents = self.reranker.rerank_documents(
                query=original_query,
                documents=[doc for doc, _ in all_scored],
                system_specifications=(
                    self.system_specs if self.config.enable_instructed_retrieval else None
                ),
            )
        else:
            all_scored.sort(key=lambda x: x[1], reverse=True)
            all_documents = [doc for doc, _ in all_scored]

        return self.aggregate_chunks(all_documents[:GLOBAL_TOP_K])

    @mlflow.trace(span_type=SpanType.TASK)
    async def query_retriever(
        self,
        query: str,
        priority_data_source: str,
        filters: dict | None = None,
    ) -> list[tuple[CustomDocument, float]]:
        try:
            results: list[
                tuple[LangchainDocument, float]
            ] = await self.vector_store.asimilarity_search_with_score(
                query=query,
                k=LOCAL_TOP_K,
                query_type=QUERY_TYPE,
                filter=filters,
            )
            return self._parse_langchain_docs(results, priority_data_source)

        except Exception as e:
            if filters is None:
                logger.error("Unfiltered retrieval failed for query %r: %s", query, e)
                raise

            logger.warning(
                "Filtered retrieval failed for query %r with filters %r: %s. "
                "Falling back to unfiltered retrieval.",
                query,
                filters,
                e,
            )
            try:
                results = await self.vector_store.asimilarity_search_with_score(
                    query=query,
                    k=LOCAL_TOP_K,
                    query_type=QUERY_TYPE,
                )
            except Exception as fallback_error:
                logger.error(
                    "Unfiltered fallback also failed for query %r: %s", query, fallback_error
                )
                raise
            return self._parse_langchain_docs(results, priority_data_source)

    @mlflow.trace(span_type=SpanType.TASK)
    def aggregate_chunks(self, documents: list[CustomDocument]) -> list[MLflowDocument]:
        doc_groups: dict[str, dict] = {}

        for doc in documents:
            doc_uri = doc.title
            if doc_uri not in doc_groups:
                doc_groups[doc_uri] = {
                    "doc_uri": doc_uri,
                    "chunks": [],
                    "prioritized": doc.prioritized,
                }
            doc_groups[doc_uri]["chunks"].append(doc.page_content)

        formatted_docs: list[MLflowDocument] = []
        for _, doc_data in doc_groups.items():
            aggregated_content = "\n\n".join(
                f"{'=' * 10}\nChunk #{i + 1}:\n{'=' * 10}\n{chunk}"
                for i, chunk in enumerate(doc_data["chunks"])
            )
            formatted_docs.append(
                MLflowDocument(
                    page_content=aggregated_content,
                    metadata={
                        "doc_uri": doc_data["doc_uri"],
                        "prioritized": doc_data["prioritized"],
                    },
                )
            )
        return formatted_docs

    def _parse_langchain_docs(
        self,
        results: list[tuple[LangchainDocument, float]],
        priority_data_source: str,
    ) -> list[tuple[CustomDocument, float]]:
        return [
            (
                CustomDocument(
                    title=doc.metadata[TITLE_COLUMN],
                    page_content=doc.page_content,
                    prioritized=doc.metadata.get(CATEGORY_COLUMN) == priority_data_source,
                ),
                score,
            )
            for doc, score in results
        ]

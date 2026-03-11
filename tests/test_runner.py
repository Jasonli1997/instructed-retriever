"""Tests for InstructedRetrieverRunner."""

from unittest.mock import AsyncMock, MagicMock, patch

import dspy
import pytest
from langchain_core.documents import Document as LangchainDocument
from mlflow.entities import Document as MLflowDocument
from pydantic import SecretStr

from instructed_retriever.agent.config import InstructedRetrieverConfiguration
from instructed_retriever.agent.context import RunContext
from instructed_retriever.agent.dspy.instructions import (
    FieldType,
    FilterCondition,
    FilterOperator,
    IndexField,
    IndexSchema,
    SystemSpecifications,
)
from instructed_retriever.agent.dspy.schemas import CustomDocument, StructuredQuery
from instructed_retriever.agent.runner import (
    CATEGORY_COLUMN,
    TITLE_COLUMN,
    InstructedRetrieverRunner,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config() -> InstructedRetrieverConfiguration:
    return InstructedRetrieverConfiguration(
        databricks_host="https://test.databricks.com",
        databricks_client_id="test-client-id",
        databricks_client_secret=SecretStr("test-secret"),
        embedding_model="test-embedding-model",
        query_rewriter_model="test-query-rewriter",
        answer_generator_model="test-answer-generator",
        vs_endpoint="test-vs-endpoint",
        vs_index_name="test-vs-index",
        enable_instructed_retrieval=True,
    )


@pytest.fixture
def mock_system_specs() -> SystemSpecifications:
    schema = IndexSchema(
        fields=[
            IndexField(
                name="data_source",
                field_type=FieldType.STRING,
                description="Source of the document",
                filterable=True,
            ),
        ]
    )
    return SystemSpecifications(
        index_schema=schema,
        user_instructions=[],
        examples=[],
        response_constraints={},
        expected_categories={},
    )


@pytest.fixture
def runner(
    mock_config: InstructedRetrieverConfiguration, mock_system_specs: SystemSpecifications
) -> InstructedRetrieverRunner:
    """Create an InstructedRetrieverRunner with all external dependencies mocked."""
    with (
        patch("instructed_retriever.agent.runner.WorkspaceClient"),
        patch("instructed_retriever.agent.runner.DatabricksEmbeddings"),
        patch("instructed_retriever.agent.runner.DatabricksVectorSearch") as mock_vs_cls,
        patch(
            "instructed_retriever.agent.runner.load_system_specifications",
            return_value=mock_system_specs,
        ),
        patch("instructed_retriever.agent.runner.dspy.ChainOfThought"),
        patch("instructed_retriever.agent.runner.dspy.Predict"),
    ):
        from instructed_retriever.agent.runner import InstructedRetrieverRunner

        mock_vs_instance = MagicMock()
        mock_vs_cls.return_value = mock_vs_instance

        r = InstructedRetrieverRunner(config=mock_config)
        r.vector_store = mock_vs_instance
        return r


@pytest.fixture
def run_context() -> RunContext:
    return RunContext(chat_history=dspy.History(messages=[]))


# ---------------------------------------------------------------------------
# Tests: _parse_langchain_docs
# ---------------------------------------------------------------------------


class TestParseLangchainDocs:
    def _make_lc_doc(
        self, content: str, title: str, data_source: str = "other"
    ) -> LangchainDocument:
        return LangchainDocument(
            page_content=content,
            metadata={TITLE_COLUMN: title, CATEGORY_COLUMN: data_source},
        )

    def test_parses_basic_documents(self, runner: InstructedRetrieverRunner) -> None:
        docs = [
            (self._make_lc_doc("content A", "doc-a"), 0.9),
            (self._make_lc_doc("content B", "doc-b"), 0.7),
        ]
        result = runner._parse_langchain_docs(docs, "login_docs")
        parsed_doc, score = result[0]
        assert parsed_doc.title == "doc-a"
        assert parsed_doc.page_content == "content A"
        assert score == 0.9

    def test_marks_priority_data_source(self, runner: InstructedRetrieverRunner) -> None:
        docs = [
            (self._make_lc_doc("priority content", "doc-p", data_source="login_docs"), 0.95),
            (self._make_lc_doc("other content", "doc-o", data_source="other_docs"), 0.8),
        ]
        result = runner._parse_langchain_docs(docs, "login_docs")
        priority_doc, _ = result[0]
        other_doc, _ = result[1]
        assert priority_doc.prioritized is True
        assert other_doc.prioritized is False

    def test_returns_empty_list_for_no_docs(self, runner: InstructedRetrieverRunner) -> None:
        assert runner._parse_langchain_docs([], "login_docs") == []

    def test_missing_data_source_defaults_to_not_prioritized(
        self, runner: InstructedRetrieverRunner
    ) -> None:
        lc_doc = LangchainDocument(page_content="no source", metadata={"title": "doc-x"})
        result = runner._parse_langchain_docs([(lc_doc, 0.5)], "login_docs")
        doc, _ = result[0]
        assert doc.prioritized is False


# ---------------------------------------------------------------------------
# Tests: aggregate_chunks
# ---------------------------------------------------------------------------


class TestAggregateChunks:
    def _make_doc(self, title: str, content: str, prioritized: bool = False) -> CustomDocument:
        return CustomDocument(title=title, page_content=content, prioritized=prioritized)

    def test_aggregates_chunks_by_title(self, runner: InstructedRetrieverRunner) -> None:
        docs = [self._make_doc("doc-a", "chunk 1"), self._make_doc("doc-a", "chunk 2")]
        result = runner.aggregate_chunks(docs)

        assert len(result) == 1
        assert "chunk 1" in result[0].page_content
        assert "chunk 2" in result[0].page_content
        assert result[0].metadata["doc_uri"] == "doc-a"

    def test_separates_different_documents(self, runner: InstructedRetrieverRunner) -> None:
        docs = [self._make_doc("doc-a", "content a"), self._make_doc("doc-b", "content b")]
        result = runner.aggregate_chunks(docs)

        assert len(result) == 2
        uris = {r.metadata["doc_uri"] for r in result}
        assert uris == {"doc-a", "doc-b"}

    def test_preserves_prioritized_flag(self, runner: InstructedRetrieverRunner) -> None:
        docs = [
            self._make_doc("doc-p", "priority chunk", prioritized=True),
            self._make_doc("doc-o", "other chunk", prioritized=False),
        ]
        result = runner.aggregate_chunks(docs)

        by_uri = {r.metadata["doc_uri"]: r for r in result}
        assert by_uri["doc-p"].metadata["prioritized"] is True
        assert by_uri["doc-o"].metadata["prioritized"] is False

    def test_returns_empty_for_no_documents(self, runner: InstructedRetrieverRunner) -> None:
        assert runner.aggregate_chunks([]) == []

    def test_chunk_format_includes_separators(self, runner: InstructedRetrieverRunner) -> None:
        docs = [self._make_doc("doc-a", "chunk 1"), self._make_doc("doc-a", "chunk 2")]
        result = runner.aggregate_chunks(docs)

        content = result[0].page_content
        assert "Chunk #1" in content
        assert "Chunk #2" in content
        assert "==========" in content

    def test_returns_mlflow_document_type(self, runner: InstructedRetrieverRunner) -> None:
        docs = [self._make_doc("doc-a", "content")]
        result = runner.aggregate_chunks(docs)
        assert isinstance(result[0], MLflowDocument)


# ---------------------------------------------------------------------------
# Tests: query_retriever
# ---------------------------------------------------------------------------


class TestQueryRetriever:
    @pytest.mark.asyncio
    async def test_returns_parsed_results(self, runner: InstructedRetrieverRunner) -> None:
        lc_doc = LangchainDocument(
            page_content="result content",
            metadata={TITLE_COLUMN: "result-doc", CATEGORY_COLUMN: "login_docs"},
        )
        runner.vector_store.asimilarity_search_with_score = AsyncMock(return_value=[(lc_doc, 0.88)])

        result = await runner.query_retriever("test query", "login_docs")

        assert len(result) == 1
        doc, score = result[0]
        assert doc.title == "result-doc"
        assert doc.page_content == "result content"
        assert doc.prioritized is True
        assert score == 0.88

    @pytest.mark.asyncio
    async def test_passes_filters_to_vector_store(self, runner: InstructedRetrieverRunner) -> None:
        runner.vector_store.asimilarity_search_with_score = AsyncMock(return_value=[])

        await runner.query_retriever("query", "login_docs", filters={"data_source": "login_docs"})

        call_kwargs = runner.vector_store.asimilarity_search_with_score.call_args.kwargs
        assert call_kwargs.get("filter") == {"data_source": "login_docs"}

    @pytest.mark.asyncio
    async def test_falls_back_to_unfiltered_on_error(
        self, runner: InstructedRetrieverRunner
    ) -> None:
        lc_doc = LangchainDocument(
            page_content="fallback content",
            metadata={"title": "fallback-doc", "data_source": "other"},
        )
        runner.vector_store.asimilarity_search_with_score = AsyncMock(
            side_effect=[Exception("filter error"), [(lc_doc, 0.5)]]
        )

        result = await runner.query_retriever("query", "login_docs", filters={"bad_field": "value"})

        assert len(result) == 1
        assert runner.vector_store.asimilarity_search_with_score.call_count == 2
        second_call_kwargs = runner.vector_store.asimilarity_search_with_score.call_args_list[
            1
        ].kwargs
        assert "filter" not in second_call_kwargs

    @pytest.mark.asyncio
    async def test_raises_on_network_error_without_filters(
        self, runner: InstructedRetrieverRunner
    ) -> None:
        runner.vector_store.asimilarity_search_with_score = AsyncMock(
            side_effect=Exception("network error")
        )

        with pytest.raises(Exception, match="network error"):
            await runner.query_retriever("query", "login_docs", filters=None)

        assert runner.vector_store.asimilarity_search_with_score.call_count == 1


# ---------------------------------------------------------------------------
# Tests: retrieve_context
# ---------------------------------------------------------------------------


class TestRetrieveContext:
    def _make_structured_query(self, query_text: str) -> StructuredQuery:
        return StructuredQuery(query_text=query_text, probability=0.9)

    @pytest.mark.asyncio
    async def test_deduplicates_chunks_by_content(self, runner: InstructedRetrieverRunner) -> None:
        lc_doc = LangchainDocument(
            page_content="duplicate content",
            metadata={"title": "doc-dup", "data_source": "other"},
        )
        runner.vector_store.asimilarity_search_with_score = AsyncMock(return_value=[(lc_doc, 0.8)])

        queries = [
            self._make_structured_query("query 1"),
            self._make_structured_query("query 2"),
        ]
        result = await runner.retrieve_context("original", queries, "login_docs")
        all_content = [doc.page_content for doc in result]
        combined = "\n".join(all_content)
        assert combined.count("duplicate content") == 1

    @pytest.mark.asyncio
    async def test_limits_to_global_top_k(self, runner: InstructedRetrieverRunner) -> None:
        def make_doc(i: int) -> LangchainDocument:
            return LangchainDocument(
                page_content=f"content {i}",
                metadata={"title": f"doc-{i}", "data_source": "other"},
            )

        runner.vector_store.asimilarity_search_with_score = AsyncMock(
            return_value=[(make_doc(i), float(5 - i)) for i in range(5)]
        )

        queries = [self._make_structured_query("query")]
        result = await runner.retrieve_context("original", queries, "login_docs")
        uris = [doc.metadata["doc_uri"] for doc in result]
        assert len(set(uris)) <= 3

    @pytest.mark.asyncio
    async def test_accumulates_scores_for_repeated_docs(
        self, runner: InstructedRetrieverRunner
    ) -> None:
        repeated_doc = LangchainDocument(
            page_content="repeated content",
            metadata={"title": "repeated-doc", "data_source": "other"},
        )
        unique_doc = LangchainDocument(
            page_content="unique content",
            metadata={"title": "unique-doc", "data_source": "other"},
        )

        runner.vector_store.asimilarity_search_with_score = AsyncMock(
            side_effect=[
                [(repeated_doc, 0.5), (unique_doc, 0.9)],
                [(repeated_doc, 0.5)],
            ]
        )

        queries = [
            self._make_structured_query("q1"),
            self._make_structured_query("q2"),
        ]
        runner.reranker = None
        result = await runner.retrieve_context("original", queries, "login_docs")

        uris = [doc.metadata["doc_uri"] for doc in result]
        assert uris[0] == "repeated-doc"

    @pytest.mark.asyncio
    async def test_uses_reranker_when_configured(self, runner: InstructedRetrieverRunner) -> None:
        lc_doc = LangchainDocument(
            page_content="some content",
            metadata={"title": "doc-r", "data_source": "other"},
        )
        runner.vector_store.asimilarity_search_with_score = AsyncMock(return_value=[(lc_doc, 0.7)])

        mock_reranker = MagicMock()
        reranked_docs = [CustomDocument(title="doc-r", page_content="some content")]
        mock_reranker.rerank_documents.return_value = reranked_docs
        runner.reranker = mock_reranker

        queries = [self._make_structured_query("query")]
        await runner.retrieve_context("original", queries, "login_docs")

        mock_reranker.rerank_documents.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: aforward
# ---------------------------------------------------------------------------


class TestAforward:
    def _make_structured_queries(self, n: int = 1) -> list[StructuredQuery]:
        return [StructuredQuery(query_text=f"query {i}", probability=0.9) for i in range(n)]

    def _make_rewrite_prediction(self, queries: list[StructuredQuery]) -> MagicMock:
        pred = MagicMock()
        pred.structured_queries = queries
        return pred

    def _make_classify_prediction(self, category: str | None = "support") -> MagicMock:
        pred = MagicMock()
        pred.category = category
        return pred

    def _make_answer_prediction(self, answer: str = "The answer.") -> MagicMock:
        pred = MagicMock()
        pred.answer = answer
        return pred

    @pytest.mark.asyncio
    async def test_attaches_category_to_prediction(
        self, runner: InstructedRetrieverRunner, run_context: RunContext
    ) -> None:
        structured_queries = self._make_structured_queries()
        runner.query_rewriter = MagicMock()
        runner.query_rewriter.acall = AsyncMock(
            return_value=self._make_rewrite_prediction(structured_queries)
        )
        runner.category_classifier = MagicMock()
        runner.category_classifier.acall = AsyncMock(
            return_value=self._make_classify_prediction("boost")
        )
        runner.answer_generator = MagicMock()
        runner.answer_generator.acall = AsyncMock(
            return_value=self._make_answer_prediction("some answer")
        )
        runner.vector_store.asimilarity_search_with_score = AsyncMock(return_value=[])

        with patch("instructed_retriever.agent.runner.dspy.LM"):
            pred = await runner.aforward("What is boost?", run_context)

        assert pred.category == "boost"

    @pytest.mark.asyncio
    async def test_handles_empty_category(
        self, runner: InstructedRetrieverRunner, run_context: RunContext
    ) -> None:
        structured_queries = self._make_structured_queries()
        runner.query_rewriter = MagicMock()
        runner.query_rewriter.acall = AsyncMock(
            return_value=self._make_rewrite_prediction(structured_queries)
        )
        runner.category_classifier = MagicMock()
        runner.category_classifier.acall = AsyncMock(
            return_value=self._make_classify_prediction(None)
        )
        runner.answer_generator = MagicMock()
        runner.answer_generator.acall = AsyncMock(return_value=self._make_answer_prediction())
        runner.vector_store.asimilarity_search_with_score = AsyncMock(return_value=[])

        with patch("instructed_retriever.agent.runner.dspy.LM"):
            pred = await runner.aforward("test query", run_context)

        assert pred.category == ""

    @pytest.mark.asyncio
    async def test_limits_structured_queries_to_max(
        self, runner: InstructedRetrieverRunner, run_context: RunContext
    ) -> None:
        many_queries = self._make_structured_queries(n=10)
        runner.query_rewriter = MagicMock()
        runner.query_rewriter.acall = AsyncMock(
            return_value=self._make_rewrite_prediction(many_queries)
        )
        runner.category_classifier = MagicMock()
        runner.category_classifier.acall = AsyncMock(return_value=self._make_classify_prediction())
        runner.answer_generator = MagicMock()
        runner.answer_generator.acall = AsyncMock(return_value=self._make_answer_prediction())
        runner.vector_store.asimilarity_search_with_score = AsyncMock(return_value=[])

        with patch("instructed_retriever.agent.runner.dspy.LM"):
            await runner.aforward("big query", run_context)

        assert runner.vector_store.asimilarity_search_with_score.call_count <= 5

    @pytest.mark.asyncio
    async def test_separates_prioritized_and_other_docs(
        self, runner: InstructedRetrieverRunner, run_context: RunContext
    ) -> None:
        structured_queries = self._make_structured_queries()
        runner.query_rewriter = MagicMock()
        runner.query_rewriter.acall = AsyncMock(
            return_value=self._make_rewrite_prediction(structured_queries)
        )
        runner.category_classifier = MagicMock()
        runner.category_classifier.acall = AsyncMock(return_value=self._make_classify_prediction())

        answer_pred = self._make_answer_prediction()
        runner.answer_generator = MagicMock()
        runner.answer_generator.acall = AsyncMock(return_value=answer_pred)

        priority_doc = LangchainDocument(
            page_content="priority content",
            metadata={TITLE_COLUMN: "prio-doc", CATEGORY_COLUMN: "support"},
        )
        regular_doc = LangchainDocument(
            page_content="regular content",
            metadata={TITLE_COLUMN: "reg-doc", CATEGORY_COLUMN: "other"},
        )
        runner.vector_store.asimilarity_search_with_score = AsyncMock(
            return_value=[(priority_doc, 0.9), (regular_doc, 0.7)]
        )

        with patch("instructed_retriever.agent.runner.dspy.LM"):
            await runner.aforward("test query", run_context)

        call_kwargs = runner.answer_generator.acall.call_args.kwargs
        assert "priority content" in call_kwargs["prioritized_context"]
        assert "regular content" in call_kwargs["other_context"]
        assert "priority content" not in call_kwargs["other_context"]


# ---------------------------------------------------------------------------
# Tests: aforward with instructed retrieval disabled
# ---------------------------------------------------------------------------


class TestAforwardInstructedRetrievalDisabled:
    def _make_structured_queries(self, n: int = 1) -> list[StructuredQuery]:
        return [StructuredQuery(query_text=f"query {i}", probability=0.9) for i in range(n)]

    def _make_rewrite_prediction(self, queries: list[StructuredQuery]) -> MagicMock:
        pred = MagicMock()
        pred.structured_queries = queries
        return pred

    def _make_answer_prediction(self, answer: str = "The answer.") -> MagicMock:
        pred = MagicMock()
        pred.answer = answer
        return pred

    @pytest.fixture
    def runner_no_instructed(
        self, mock_config: InstructedRetrieverConfiguration, mock_system_specs: SystemSpecifications
    ) -> InstructedRetrieverRunner:
        mock_config.enable_instructed_retrieval = False
        with (
            patch("instructed_retriever.agent.runner.WorkspaceClient"),
            patch("instructed_retriever.agent.runner.DatabricksEmbeddings"),
            patch("instructed_retriever.agent.runner.DatabricksVectorSearch") as mock_vs_cls,
            patch(
                "instructed_retriever.agent.runner.load_system_specifications",
                return_value=mock_system_specs,
            ),
            patch("instructed_retriever.agent.runner.dspy.ChainOfThought"),
            patch("instructed_retriever.agent.runner.dspy.Predict"),
        ):
            from instructed_retriever.agent.runner import InstructedRetrieverRunner

            mock_vs_instance = MagicMock()
            mock_vs_cls.return_value = mock_vs_instance

            r = InstructedRetrieverRunner(config=mock_config)
            r.vector_store = mock_vs_instance
            return r

    @pytest.mark.asyncio
    async def test_category_classifier_not_called(
        self, runner_no_instructed: InstructedRetrieverRunner, run_context: RunContext
    ) -> None:
        runner_no_instructed.query_rewriter = MagicMock()
        runner_no_instructed.query_rewriter.acall = AsyncMock(
            return_value=self._make_rewrite_prediction(self._make_structured_queries())
        )
        runner_no_instructed.category_classifier = MagicMock()
        runner_no_instructed.category_classifier.acall = AsyncMock()
        runner_no_instructed.answer_generator = MagicMock()
        runner_no_instructed.answer_generator.acall = AsyncMock(
            return_value=self._make_answer_prediction()
        )
        runner_no_instructed.vector_store.asimilarity_search_with_score = AsyncMock(return_value=[])

        with patch("instructed_retriever.agent.runner.dspy.LM"):
            await runner_no_instructed.aforward("test query", run_context)

        runner_no_instructed.category_classifier.acall.assert_not_called()

    @pytest.mark.asyncio
    async def test_category_is_empty_string(
        self, runner_no_instructed: InstructedRetrieverRunner, run_context: RunContext
    ) -> None:
        runner_no_instructed.query_rewriter = MagicMock()
        runner_no_instructed.query_rewriter.acall = AsyncMock(
            return_value=self._make_rewrite_prediction(self._make_structured_queries())
        )
        runner_no_instructed.answer_generator = MagicMock()
        runner_no_instructed.answer_generator.acall = AsyncMock(
            return_value=self._make_answer_prediction()
        )
        runner_no_instructed.vector_store.asimilarity_search_with_score = AsyncMock(return_value=[])

        with patch("instructed_retriever.agent.runner.dspy.LM"):
            pred = await runner_no_instructed.aforward("test query", run_context)

        assert pred.category == ""

    @pytest.mark.asyncio
    async def test_retrieval_uses_no_filters(
        self, runner_no_instructed: InstructedRetrieverRunner, run_context: RunContext
    ) -> None:
        sq = StructuredQuery(
            query_text="query 0",
            probability=0.9,
            filters=[
                FilterCondition(field="data_source", operator=FilterOperator.EQUALS, value="boost")
            ],
        )
        rewrite_pred = MagicMock()
        rewrite_pred.structured_queries = [sq]
        runner_no_instructed.query_rewriter = MagicMock()
        runner_no_instructed.query_rewriter.acall = AsyncMock(return_value=rewrite_pred)
        runner_no_instructed.answer_generator = MagicMock()
        runner_no_instructed.answer_generator.acall = AsyncMock(
            return_value=self._make_answer_prediction()
        )
        runner_no_instructed.vector_store.asimilarity_search_with_score = AsyncMock(return_value=[])

        with patch("instructed_retriever.agent.runner.dspy.LM"):
            await runner_no_instructed.aforward("test query", run_context)

        call_kwargs = (
            runner_no_instructed.vector_store.asimilarity_search_with_score.call_args.kwargs
        )
        assert call_kwargs.get("filter") is None

    @pytest.mark.asyncio
    async def test_all_docs_treated_as_other_context(
        self, runner_no_instructed: InstructedRetrieverRunner, run_context: RunContext
    ) -> None:
        runner_no_instructed.query_rewriter = MagicMock()
        runner_no_instructed.query_rewriter.acall = AsyncMock(
            return_value=self._make_rewrite_prediction(self._make_structured_queries())
        )
        runner_no_instructed.answer_generator = MagicMock()
        runner_no_instructed.answer_generator.acall = AsyncMock(
            return_value=self._make_answer_prediction()
        )

        doc = LangchainDocument(
            page_content="some content",
            metadata={TITLE_COLUMN: "doc-a", CATEGORY_COLUMN: "support"},
        )
        runner_no_instructed.vector_store.asimilarity_search_with_score = AsyncMock(
            return_value=[(doc, 0.9)]
        )

        with patch("instructed_retriever.agent.runner.dspy.LM"):
            await runner_no_instructed.aforward("test query", run_context)

        call_kwargs = runner_no_instructed.answer_generator.acall.call_args.kwargs
        assert call_kwargs["prioritized_context"] == ""
        assert "some content" in call_kwargs["other_context"]

    @pytest.mark.asyncio
    async def test_system_specs_not_passed_to_rewriter(
        self, runner_no_instructed: InstructedRetrieverRunner, run_context: RunContext
    ) -> None:
        runner_no_instructed.query_rewriter = MagicMock()
        runner_no_instructed.query_rewriter.acall = AsyncMock(
            return_value=self._make_rewrite_prediction(self._make_structured_queries())
        )
        runner_no_instructed.answer_generator = MagicMock()
        runner_no_instructed.answer_generator.acall = AsyncMock(
            return_value=self._make_answer_prediction()
        )
        runner_no_instructed.vector_store.asimilarity_search_with_score = AsyncMock(return_value=[])

        with patch("instructed_retriever.agent.runner.dspy.LM"):
            await runner_no_instructed.aforward("test query", run_context)

        rewriter_kwargs = runner_no_instructed.query_rewriter.acall.call_args.kwargs
        assert rewriter_kwargs["system_specifications"] is None

        answer_kwargs = runner_no_instructed.answer_generator.acall.call_args.kwargs
        assert answer_kwargs["system_specifications"] is None

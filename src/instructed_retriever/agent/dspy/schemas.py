from typing import TypedDict

from pydantic import BaseModel, Field

from instructed_retriever.agent.dspy.instructions import FilterCondition


class DocsDict(TypedDict):
    prioritized: str
    others: str


class CustomDocument(BaseModel):
    """A retrieved document with content and metadata."""

    title: str = Field(..., description="Document URI or title")
    page_content: str = Field(..., description="Text content of the document chunk")
    prioritized: bool = Field(
        default=False, description="Whether this document is from a priority data source"
    )
    rerank_score: float | None = Field(
        default=None, description="Relevance score from the reranker"
    )
    rerank_reasoning: str | None = Field(
        default=None, description="Explanation from the reranker"
    )


class StructuredQuery(BaseModel):
    """A structured query with keyword search and optional metadata filters."""

    query_text: str = Field(..., description="The keyword search query")
    probability: float = Field(
        ..., description="Likelihood that this query aligns with the user's intent"
    )
    reasoning: str = Field(
        default="",
        description="Reasoning for how this query was formulated, including filter choices",
    )
    filters: list[FilterCondition] = Field(
        default_factory=list, description="Metadata filters to apply"
    )

    def to_databricks_filter(self) -> dict | None:
        """Convert all filters to a Databricks standard endpoint filter dict.

        Returns None if no filters are specified. Multiple conditions are
        combined with AND logic (multiple keys in one dict).
        """
        if not self.filters:
            return None
        combined: dict = {}
        for f in self.filters:
            combined.update(f.to_databricks_filter())
        return combined

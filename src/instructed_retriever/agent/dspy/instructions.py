"""System specifications and instruction models for Instructed Retriever."""

import logging
from enum import Enum
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class InstructionType(str, Enum):
    """Types of user instructions for retrieval."""

    INCLUSION = "inclusion"
    EXCLUSION = "exclusion"
    RECENCY = "recency"
    CUSTOM = "custom"  # Sent to both query rewriter and answer generator
    CUSTOM_QUERY_REWRITER = "custom_query_rewriter"
    CUSTOM_ANSWER_GENERATOR = "custom_answer_generator"


class FilterOperator(str, Enum):
    """Filter operators supported by Databricks Vector Search standard endpoints."""

    EQUALS = "="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    IN = "IN"
    LIKE = "LIKE"


class FieldType(str, Enum):
    """Supported field types in the index schema."""

    STRING = "string"
    TIMESTAMP = "timestamp"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"


class IndexField(BaseModel):
    """Definition of a single field in the index schema."""

    name: str = Field(..., description="Field name in the index")
    field_type: FieldType = Field(..., description="Data type of the field")
    description: str = Field(default="", description="Human-readable description")
    filterable: bool = Field(default=True, description="Whether this field can be filtered")
    examples: list[str] = Field(default_factory=list, description="Example values")


class IndexSchema(BaseModel):
    """Schema definition for the vector search index."""

    fields: list[IndexField] = Field(..., description="List of available fields")

    def get_field(self, name: str) -> IndexField | None:
        for field in self.fields:
            if field.name == name:
                return field
        return None

    def get_filterable_fields(self) -> list[IndexField]:
        return [field for field in self.fields if field.filterable]

    def to_description(self) -> str:
        lines = ["Available vector index fields:"]
        for field in self.fields:
            filterable = " (filterable)" if field.filterable else " (non-filterable)"
            examples = f" - Examples: {', '.join(field.examples)}" if field.examples else ""
            desc = f" - {field.description}" if field.description else ""
            lines.append(f"- {field.name} ({field.field_type.value}){filterable}{desc}{examples}")
        return "\n".join(lines)


class UserInstruction(BaseModel):
    """A single behavioral instruction for retrieval."""

    instruction_type: InstructionType = Field(..., description="Type of instruction")
    description: str = Field(..., description="Natural language instruction")
    field_name: str | None = Field(default=None, description="Target index field if applicable")
    priority: int = Field(
        default=1, description="Priority level (higher = more important)", ge=1, le=10
    )


class FilterCondition(BaseModel):
    """A single filter condition to apply during retrieval."""

    field: str = Field(..., description="Field name to filter on")
    operator: FilterOperator = Field(..., description="Comparison operator")
    value: Any = Field(..., description="Value to compare against")

    @model_validator(mode="after")
    def coerce_in_values(self) -> "FilterCondition":
        if self.operator == FilterOperator.IN:
            if isinstance(self.value, (list, tuple, set)):
                self.value = list(self.value)
            else:
                self.value = [self.value]
        return self

    def to_databricks_filter(self) -> dict:
        """
        Convert to Databricks Vector Search standard endpoint filter format.

        Examples:
          data_source = 'login'   -> {"data_source": "login"}
          timestamp > '2024-01-01' -> {"timestamp >": "2024-01-01"}
          category IN ('a', 'b')  -> {"category": ["a", "b"]}
        """
        if self.operator == FilterOperator.EQUALS:
            return {self.field: self.value}
        elif self.operator == FilterOperator.NOT_EQUALS:
            return {f"{self.field} NOT": self.value}
        elif self.operator == FilterOperator.GREATER_THAN:
            return {f"{self.field} >": self.value}
        elif self.operator == FilterOperator.LESS_THAN:
            return {f"{self.field} <": self.value}
        elif self.operator == FilterOperator.GREATER_EQUAL:
            return {f"{self.field} >=": self.value}
        elif self.operator == FilterOperator.LESS_EQUAL:
            return {f"{self.field} <=": self.value}
        elif self.operator == FilterOperator.IN:
            return {self.field: self.value}
        elif self.operator == FilterOperator.LIKE:
            return {f"{self.field} LIKE": self.value}
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")


class RelevantExample(BaseModel):
    """A labeled query-document pair used for few-shot guidance."""

    query: str = Field(..., description="Example query")
    document: str = Field(default="", description="Description of a relevant document")
    relevance_reason: str = Field(default="", description="Why this document is relevant")
    subquery: str = Field(default="", description="Expected subquery the rewriter should generate")
    subquery_reasoning: str = Field(default="", description="Why this subquery is appropriate")
    category: str = Field(default="", description="Expected category for this query")
    category_reasoning: str = Field(default="", description="Why this query belongs here")

    @model_validator(mode="after")
    def check_document_or_category(self) -> "RelevantExample":
        if not self.query:
            raise ValueError("query must be non-empty")
        if not self.document and not self.category and not self.subquery:
            raise ValueError("at least one of document, category, or subquery must be filled")
        return self


class SystemSpecifications(BaseModel):
    """Complete system specifications for instructed retrieval."""

    index_schema: IndexSchema = Field(..., description="Schema of the search index")
    user_instructions: list[UserInstruction] = Field(default_factory=list)
    examples: list[RelevantExample] = Field(default_factory=list)
    response_constraints: dict[str, Any] = Field(default_factory=dict)
    expected_categories: dict[str, str | None] = Field(default_factory=dict)

    def has_instructions(self) -> bool:
        return len(self.user_instructions) > 0

    def to_query_rewriter_context(self) -> str:
        query_rewriter_types = {InstructionType.CUSTOM, InstructionType.CUSTOM_QUERY_REWRITER}
        sections: list[str] = []

        sections.append("## Index Schema")
        sections.append(self.index_schema.to_description())

        qr_instructions = [
            inst for inst in self.user_instructions if inst.instruction_type in query_rewriter_types
        ]
        if qr_instructions:
            sections.append("\n## User Instructions")
            sections.append(
                "\nFollow these instructions when formulating queries and filters "
                "(priority=10 is highest):"
            )
            for i, inst in enumerate(qr_instructions, 1):
                sections.append(f"{i}. {inst.description} (priority: {inst.priority})")

        retrieval_examples = [ex for ex in self.examples if ex.document or ex.subquery]
        if retrieval_examples:
            sections.append("\n## Retrieval Examples")
            for i, ex in enumerate(retrieval_examples, 1):
                sections.append(f"{i}. Query: {ex.query}")
                if ex.document:
                    sections.append(f"   Relevant: {ex.document}")
                if ex.relevance_reason:
                    sections.append(f"   Reason: {ex.relevance_reason}")
                if ex.subquery:
                    sections.append(f"   Subquery: {ex.subquery}")
                if ex.subquery_reasoning:
                    sections.append(f"   Subquery Reasoning: {ex.subquery_reasoning}")

        return "\n".join(sections)

    def to_retrieval_context(self) -> str:
        retrieval_types = {
            InstructionType.INCLUSION,
            InstructionType.EXCLUSION,
            InstructionType.RECENCY,
            InstructionType.CUSTOM,
        }
        sections: list[str] = []

        retrieval_instructions = [
            inst for inst in self.user_instructions if inst.instruction_type in retrieval_types
        ]
        if retrieval_instructions:
            sections.append("## User Instructions")
            sections.append(
                "\nFollow these instructions when scoring and ranking documents "
                "(priority=10 is highest):"
            )
            for i, inst in enumerate(retrieval_instructions, 1):
                sections.append(f"{i}. {inst.description} (priority: {inst.priority})")

        retrieval_examples = [ex for ex in self.examples if ex.document]
        if retrieval_examples:
            sections.append("\n## Retrieval Examples")
            for i, ex in enumerate(retrieval_examples, 1):
                sections.append(f"{i}. Query: {ex.query}")
                sections.append(f"   Relevant: {ex.document}")
                if ex.relevance_reason:
                    sections.append(f"   Reason: {ex.relevance_reason}")

        return "\n".join(sections)

    def to_generation_context(self) -> str:
        generation_types = {InstructionType.CUSTOM, InstructionType.CUSTOM_ANSWER_GENERATOR}
        sections: list[str] = []

        generation_instructions = [
            inst for inst in self.user_instructions if inst.instruction_type in generation_types
        ]
        if generation_instructions:
            sections.append("## User Instructions")
            sections.append(
                "\nFollow these instructions when generating the answer (priority=10 is highest):"
            )
            for i, inst in enumerate(generation_instructions, 1):
                sections.append(f"{i}. {inst.description} (priority: {inst.priority})")

        if self.response_constraints:
            sections.append("\n## Response Constraints")
            for key, value in self.response_constraints.items():
                sections.append(f"- {key}: {value}")

        return "\n".join(sections)

    def to_categorization_context(self) -> str:
        sections: list[str] = []

        if self.expected_categories:
            sections.append("## Expected Categories")
            sections.append("Classify the query into one of the following categories:")
            for category, description in self.expected_categories.items():
                line = f"- {category}"
                if description:
                    line += f": {description}"
                sections.append(line)

        categorization_examples = [ex for ex in self.examples if ex.category]
        if categorization_examples:
            sections.append("\n## Categorization Examples")
            for i, ex in enumerate(categorization_examples, 1):
                sections.append(f"{i}. Query: {ex.query}")
                sections.append(f"   Category: {ex.category}")
                if ex.category_reasoning:
                    sections.append(f"   Reasoning: {ex.category_reasoning}")

        return "\n".join(sections)


def create_default_index_schema() -> IndexSchema:
    return IndexSchema(
        fields=[
            IndexField(
                name="title",
                field_type=FieldType.STRING,
                description="Document title or URI",
                filterable=False,
                examples=["product_pages", "blog_articles"],
            ),
            IndexField(
                name="file_updated_at",
                field_type=FieldType.TIMESTAMP,
                description="Document creation or last-modified timestamp",
                filterable=False,
                examples=["2026-01-22T18:46:16.289+00:00"],
            ),
            IndexField(
                name="category",
                field_type=FieldType.STRING,
                description="Document category or classification",
                filterable=True,
                examples=["docs", "blog", "support"],
            ),
        ]
    )


def create_empty_specifications(
    expected_categories: dict[str, str | None] | None = None,
) -> SystemSpecifications:
    return SystemSpecifications(
        index_schema=create_default_index_schema(),
        user_instructions=[],
        examples=[],
        response_constraints={},
        expected_categories=expected_categories or {},
    )


_DEFAULT_SYSTEM_SPECS_PATH = Path(__file__).parents[4] / "artifacts" / "system_specs.yaml"


def load_system_specifications(path: str | Path | None = None) -> SystemSpecifications:
    """Load system specifications from a YAML file.

    Args:
        path: Path to the YAML file. Defaults to artifacts/system_specs.yaml relative to
              the project root.

    Returns:
        Parsed SystemSpecifications, or defaults if the file does not exist.
    """
    yaml_path = Path(path) if path else _DEFAULT_SYSTEM_SPECS_PATH

    if not yaml_path.exists():
        logger.warning("System specs YAML not found at %s; using defaults", yaml_path)
        return create_empty_specifications()

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if not data:
        return create_empty_specifications()

    index_fields = []
    for field_data in data.get("index_schema", []):
        index_fields.append(
            IndexField(
                name=field_data["name"],
                field_type=FieldType(field_data["field_type"]),
                description=field_data.get("description", ""),
                filterable=field_data.get("filterable", True),
                examples=field_data.get("examples", []),
            )
        )
    index_schema = (
        IndexSchema(fields=index_fields) if index_fields else create_default_index_schema()
    )

    user_instructions = []
    for inst_data in data.get("user_instructions", []):
        user_instructions.append(
            UserInstruction(
                instruction_type=InstructionType(inst_data["instruction_type"]),
                description=inst_data["description"],
                field_name=inst_data.get("field_name"),
                priority=inst_data.get("priority", 1),
            )
        )

    response_constraints = data.get("response_constraints", {}) or {}

    raw_categories = data.get("expected_categories") or {}
    if isinstance(raw_categories, list):
        expected_categories: dict[str, str | None] = dict.fromkeys(raw_categories)
    else:
        expected_categories = {k: v or None for k, v in raw_categories.items()}

    examples = []
    for ex_data in data.get("examples") or []:
        examples.append(
            RelevantExample(
                query=ex_data["query"],
                document=ex_data.get("document", ""),
                relevance_reason=ex_data.get("relevance_reason", ""),
                subquery=ex_data.get("subquery", ""),
                subquery_reasoning=ex_data.get("subquery_reasoning", ""),
                category=ex_data.get("category", ""),
                category_reasoning=ex_data.get("category_reasoning", ""),
            )
        )

    return SystemSpecifications(
        index_schema=index_schema,
        user_instructions=user_instructions,
        examples=examples,
        response_constraints=response_constraints,
        expected_categories=expected_categories,
    )

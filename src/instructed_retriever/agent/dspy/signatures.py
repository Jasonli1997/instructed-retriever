import dspy

from instructed_retriever.agent.dspy.schemas import StructuredQuery


class QueryRewriterSignature(dspy.Signature):
    system_specifications: str | None = dspy.InputField(
        description=(
            "Retrieval-focused system specifications including index schema, "
            "retrieval instructions, examples, and category classifications. "
            "May be None if not configured."
        ),
        default=None,
    )
    history: dspy.History = dspy.InputField()
    query: str = dspy.InputField()
    # ========
    structured_queries: list[StructuredQuery] = dspy.OutputField()


class AnswerGeneratorSignature(dspy.Signature):
    system_specifications: str | None = dspy.InputField(
        description=(
            "Generation-focused system specifications including response constraints, "
            "answer generation instructions, and category classifications. "
            "May be None if not configured."
        ),
        default=None,
    )
    prioritized_context: str = dspy.InputField(
        description="Prioritized context documents relevant to the user query."
    )
    other_context: str = dspy.InputField(
        description="Secondary context — use only if prioritized_context is insufficient."
    )
    history: dspy.History = dspy.InputField()
    query: str = dspy.InputField()
    # ========
    answer: str = dspy.OutputField()


class CategoryClassifierSignature(dspy.Signature):
    system_specifications: str | None = dspy.InputField(
        description=(
            "Categorization context including expected categories with descriptions "
            "and labeled examples. May be None if not configured."
        ),
        default=None,
    )
    history: dspy.History = dspy.InputField()
    query: str = dspy.InputField()
    # ========
    category: str = dspy.OutputField(
        description="The category that best matches the user query. Empty string if none applies."
    )

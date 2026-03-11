QUERY_REWRITE_PROMPT = (
    "Decompose the user request into self-contained structured queries with keyword search and metadata filters.\n\n"
    "1. Use system specifications (if provided) to identify valid index fields and follow any user instructions.\n"
    "2. Translate intent into metadata filters (inclusion, exclusion, recency).\n"
    "3. Provide general reasoning covering query formulation and filter rationale.\n"
    "4. Assign a probability score for each query's likelihood of fulfilling the request.\n"
    "5. Generate 1-2 queries for simple requests; 3-5 for complex ones.\n"
)

ANSWER_GENERATION_PROMPT = (
    "Generate a concise, grounded answer using the provided context. "
    "If context is insufficient, say so.\n\n"
    "Context is organized as articles from source documents, with chunks separated by '==========' markers. "
    "Prefer prioritized_context over other_context.\n\n"
    "1. Ground every claim in the provided context.\n"
    "2. Follow system specifications (if provided) for response constraints, tone, and focus areas.\n"
)

CATEGORY_CLASSIFICATION_PROMPT = (
    "Classify the user query into exactly one of the expected categories.\n\n"
    "1. Use the system specifications to identify valid categories and their descriptions.\n"
    "2. Consider the conversation history for context on the user's intent.\n"
    "3. If examples are provided, use them as guidance for classification.\n"
    "4. Return an empty string only if no category applies.\n"
)

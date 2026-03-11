# Instructed Retriever — Advanced RAG with System-Level Reasoning

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DSPy](https://img.shields.io/badge/built%20with-DSPy-blueviolet)](https://dspy.ai)
[![MLflow](https://img.shields.io/badge/traced%20with-MLflow-orange)](https://mlflow.org)

**An open-source Python RAG framework** that propagates full system specifications — index schema, user instructions, and labeled examples — through every stage of the retrieval-augmented generation (RAG) pipeline, enabling reliable instruction-following that standard RAG cannot achieve.

> **Background:** This project implements the architecture described in [Instructed Retriever: Unlocking System-Level Reasoning for Search Agents](https://www.databricks.com/blog/instructed-retriever-unlocking-system-level-reasoning-search-agents). The core insight is that standard RAG ignores system-level context after the initial query embedding — the instructed retriever keeps that context alive through query decomposition, metadata-aware structured filtering, contextual reranking, and grounded response generation.

## What Makes This Different from Standard RAG

| Capability | RAG | Instructed Retriever |
|---|---|---|
| Follows user instructions | ✖ | ✅ |
| Understands index schema | ✖ | ✅ |
| Generates structured filters | ✖ | ✅ |
| Query decomposition | ✖ | ✅ |
| Low latency / small footprint | ✅ | ✅ |

Three retrieval capabilities are layered on top of standard vector search:

1. **Query Decomposition** — breaks a complex multi-part request into a search plan with multiple keyword searches and filter conditions.
2. **Metadata Reasoning** — translates natural language constraints (e.g. *"from last year"*) into precise executable filters (e.g. `doc_timestamp > TO_TIMESTAMP('2024-01-01')`).
3. **Contextual Relevance** — the reranker uses system instructions to boost documents that match user intent, not just keyword similarity.

## Prerequisites

- Python 3.12
- [Poetry](https://python-poetry.org/) for dependency management (`brew install poetry`)
- A [Databricks](https://www.databricks.com/) workspace with a Vector Search index (used for retrieval)
- Any [LiteLLM-compatible](https://docs.litellm.ai/docs/providers) LLM provider (OpenAI, Anthropic, Databricks Models, etc.)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jasonli1997/instructed-retriever.git
   cd instructed-retriever
   ```

2. **Install dependencies:**
   ```bash
   poetry install
   ```
   Include dev dependencies (tests, linting, notebooks):
   ```bash
   poetry install --with dev
   ```

3. **Install pre-commit hooks:**
   ```bash
   poetry run pre-commit install
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials and model choices
   ```

## Project Structure

```
instructed-retriever/
├── deploy_agent.py                   # MLflow model packaging script
├── artifacts/
│   └── system_specs.yaml             # Instructed retrieval configuration
├── eval/
│   ├── run_eval.py                   # MLflow evaluation runner
│   └── scorers.py                    # Custom MLflow scorers
├── src/
│   └── instructed_retriever/         # Main Python package
│       ├── redact.py                 # PII redaction (Presidio-based)
│       ├── responses_agent.py        # MLflow ResponsesAgent wiring
│       └── agent/
│           ├── config.py             # Agent configuration (pydantic-settings)
│           ├── context.py            # Per-request runtime context
│           ├── runner.py             # Core async orchestration
│           └── dspy/
│               ├── instructions.py   # SystemSpecifications Pydantic models
│               ├── prompts.py        # All prompt strings
│               ├── reranker.py       # LLM + Databricks rerankers
│               ├── schemas.py        # Typed I/O schemas
│               └── signatures.py    # DSPy signatures
└── tests/                            # pytest test suite
```

## Configuring System Specifications

`artifacts/system_specs.yaml` is the primary configuration file. It tells the LLM what your vector index looks like and how retrieval should behave — enabling structured filter generation and instruction-following that plain RAG cannot do.

The file ships with a worked example from the Databricks blog post: a **FooBrand product Q&A** assistant.

**To adapt it for your use case, replace:**

| Section | What to provide |
|---|---|
| `index_schema` | The filterable metadata fields in your Databricks Vector Search index |
| `user_instructions` | Behavioral rules (recency preferences, inclusions, exclusions) |
| `response_constraints` | Length, tone, and grounding rules for answer generation |
| `expected_categories` | Categories the query classifier can route queries to |
| `examples` | A few `<query, document, relevance_reason>` pairs as few-shot guidance |

Enable it in `.env`:
```
SYSTEM_SPECS_PATH=artifacts/system_specs.yaml
ENABLE_INSTRUCTED_RETRIEVAL=true
```

## Choosing Your LLM

The agent uses [LiteLLM](https://docs.litellm.ai) under the hood, so any LiteLLM-compatible model string works. Set these in `.env`:

```
# OpenAI
QUERY_REWRITER_MODEL=openai/gpt-4o-mini
ANSWER_GENERATOR_MODEL=openai/gpt-4o

# Anthropic
QUERY_REWRITER_MODEL=anthropic/claude-3-5-haiku-20241022
ANSWER_GENERATOR_MODEL=anthropic/claude-3-5-sonnet-20241022

# Databricks (served models)
QUERY_REWRITER_MODEL=databricks/databricks-meta-llama-3-3-70b-instruct
ANSWER_GENERATOR_MODEL=databricks/databricks-meta-llama-3-3-70b-instruct
```

Make sure the relevant API keys are in your environment (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

## Running Locally

```bash
# Run tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=src --cov-report=term-missing

# Lint (auto-fix)
poetry run ruff check src tests --fix

# Format
poetry run ruff format src tests

# Type-check
poetry run mypy src
```

## MLflow Tracing

The agent emits structured MLflow traces for every request. To enable Unity Catalog trace export set in `.env`:
```
MLFLOW_TRACE_EXPORT_TO_UC=true
OTEL_CATALOG=my_catalog
OTEL_SCHEMA=mlflow_traces
```

Refer to the [MLflow tracing documentation](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog.html) for setup details.

## PII Redaction

Set `REDACT_PII=true` in `.env` to strip personally identifiable information from MLflow trace spans before they are stored. Redaction uses [Presidio](https://microsoft.github.io/presidio/) and covers names, emails, phone numbers, SSNs, credit card numbers, and postal addresses.

## Related Work & Positioning

This project addresses well-known limitations of naive RAG pipelines — sometimes called "advanced RAG" or "agentic RAG" in the literature. Compared to frameworks like LangChain RAG, LlamaIndex, Haystack, or vanilla DSPy RAG pipelines, the instructed retriever is distinguished by:

- **Zero framework lock-in** for the retrieval layer — bring your own vector store
- **Single config file** (`system_specs.yaml`) drives all retrieval behavior — no code changes needed to adapt to a new domain
- **Full async** throughout — designed for low-latency production serving
- **First-class MLflow integration** — every request is traced and exportable to Databricks Unity Catalog

## Topics

`rag` · `retrieval-augmented-generation` · `advanced-rag` · `agentic-rag` · `vector-search` · `semantic-search` · `query-decomposition` · `reranking` · `llm` · `dspy` · `databricks` · `mlflow` · `instruction-following` · `information-retrieval` · `python`

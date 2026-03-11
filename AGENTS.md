# Developer Guide

This document is the primary reference for contributors and coding agents working in this repo. It covers repo layout, coding conventions, tooling, and common extension patterns. For architecture and design rationale, see `README.md` and the [Databricks blog post](https://www.databricks.com/blog/instructed-retriever-unlocking-system-level-reasoning-search-agents).

---

## Repo Layout

```
instructed-retriever/
├── src/instructed_retriever/   # Installable Python package (src layout)
│   ├── redact.py               # PII redaction span processor (Presidio)
│   ├── responses_agent.py      # MLflow ResponsesAgent entry point
│   └── agent/
│       ├── config.py           # Pydantic-settings configuration class
│       ├── context.py          # RunContext — per-request state
│       ├── runner.py           # Core async orchestration (aforward)
│       └── dspy/
│           ├── instructions.py # SystemSpecifications Pydantic models
│           ├── prompts.py      # All prompt strings (one constant per prompt)
│           ├── reranker.py     # InstructedReranker + DatabricksReranker
│           ├── schemas.py      # Typed DSPy I/O schemas
│           └── signatures.py  # DSPy Signatures
├── artifacts/
│   └── system_specs.yaml       # Instructed retrieval config (edit for your use case)
├── eval/
│   ├── run_eval.py             # CLI: run MLflow evaluation against a golden dataset
│   └── scorers.py              # Custom MLflow scorers (category_accuracy)
├── tests/
│   └── test_runner.py          # pytest test suite
├── deploy_agent.py             # MLflow model packaging script
├── pyproject.toml              # Project metadata, deps, and tool config
└── .env.example                # All supported environment variables with comments
```

**Where things live by concern:**

| I want to… | Go to |
|---|---|
| Change retrieval/generation logic | `src/instructed_retriever/agent/runner.py` |
| Add or change a prompt | `src/instructed_retriever/agent/dspy/prompts.py` |
| Add or change a DSPy signature | `src/instructed_retriever/agent/dspy/signatures.py` |
| Add a config field / env var | `src/instructed_retriever/agent/config.py` + `.env.example` |
| Change how the agent streams responses | `src/instructed_retriever/responses_agent.py` |
| Configure retrieval for a use case | `artifacts/system_specs.yaml` |
| Add a test | `tests/test_runner.py` |

---

## Running Locally

```bash
# Install all deps (including dev)
poetry install --with dev

# Install pre-commit hooks (run once after cloning)
poetry run pre-commit install

# Run tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=src --cov-report=term-missing

# Lint (ruff) — auto-fix
poetry run ruff check src tests --fix

# Format
poetry run ruff format src tests

# Type-check (mypy strict)
poetry run mypy src

# Run evaluation (requires a populated golden dataset in Databricks)
poetry run python eval/run_eval.py --dataset <uc_table> --schema <uc_schema>
```

---

## Pre-commit Hooks

Pre-commit runs on every `git commit`. Hooks that **modify files** (end-of-file fixer, ruff) will amend the file and exit with code 1 — simply `git add` and re-commit:

| Hook | What it does |
|---|---|
| `check-yaml` / `check-toml` | Validates YAML and TOML syntax |
| `end-of-file-fixer` | Ensures files end with a single newline |
| `trailing-whitespace` | Strips trailing whitespace |
| `ruff` (lint) | Runs ruff linter; auto-fixes safe issues |
| `ruff-format` | Formats code |
| `mypy` | Full strict type-check of `src/` |

---

## Coding Standards

### Python Version and Typing

- **Python 3.12** only (`python = ">=3.12,<3.13"`)
- **Full type annotations required** on all functions and methods — mypy runs in strict mode
- Use `from __future__ import annotations` when needed for forward refs
- `Optional[X]` → prefer `X | None` (Python 3.10+ union syntax)
- Never use bare `Any` — use `object` or a proper type; if unavoidable, add `# type: ignore[assignment]` with a comment

### Async Patterns

- All I/O-bound operations use `async def` + `await`
- Parallel async calls use `asyncio.gather()` — never sequential `await` in a loop when tasks are independent
- `pytest-asyncio` is configured with `asyncio_mode = "auto"` — mark async tests with `async def test_*` directly

### Pydantic Models

- All config and data models inherit from `pydantic.BaseModel` or `pydantic_settings.BaseSettings`
- Use `model_validator(mode="before")` / `field_validator` for coercion; never override `__init__`
- `InstructedRetrieverConfiguration` inherits `BaseSettings` — env vars are loaded automatically; `.env` file support via `SettingsConfigDict(env_file=".env")`

### DSPy Conventions

- Each signature lives in `signatures.py` as a `dspy.Signature` subclass
- Prompts live exclusively in `prompts.py` — no prompt literals elsewhere
- New signatures follow: define in `signatures.py` → add prompt in `prompts.py` → wire into `runner.py`
- Modules (`dspy.Module` subclasses) are instantiated in `InstructedRetrieverRunner.__init__` and called in `aforward()`

---

## Testing Conventions

All tests live in `tests/test_runner.py`. The suite uses `pytest` with `async def` test methods.

### Mocking Pattern

```python
# LM calls — always patch at the dspy module level
with patch("instructed_retriever.agent.runner.dspy.LM"):
    result = await runner.aforward(query, run_context)

# Vector store similarity search
runner.vector_store.asimilarity_search_with_score = AsyncMock(
    return_value=[(mock_doc, 0.9)]
)

# DSPy module calls — patch acall on the module instance
runner.query_rewriter.acall = AsyncMock(return_value=mock_prediction)
```

### Test Structure

Tests are grouped into classes by feature area:

```python
class TestInstructedRetrieverRunner:
    def setUp(self) -> None:
        self.config = _make_config()
        self.runner = _make_runner(self.config)

    async def test_<feature>_<scenario>(self) -> None:
        ...
```

---

## Adding Features

### New environment variable

1. Add the field to `InstructedRetrieverConfiguration` in `agent/config.py`
2. Add the var (with a comment) to `.env.example`
3. Pass it through `RunContext` or directly to the component that needs it

### New DSPy signature

1. Add the class to `agent/dspy/signatures.py`
2. Add the system prompt string to `agent/dspy/prompts.py`
3. Instantiate the module in `InstructedRetrieverRunner.__init__`
4. Call it in `aforward()`, following the existing `asyncio.gather()` pattern
5. Add tests mocking `module.acall`

### New reranker

1. Implement a class with a `rerank_documents(docs, query, system_specifications)` method in `agent/dspy/reranker.py`
2. Add the selection logic in `InstructedRetrieverRunner.__init__` where `self.reranker` is set
3. Add the corresponding env var to `config.py` and `.env.example`

### Changing system_specs.yaml

The schema is validated against `SystemSpecifications` in `agent/dspy/instructions.py`. When adding a new field:
1. Add it to the relevant Pydantic model in `instructions.py`
2. Update `artifacts/system_specs.yaml` with an example value and comment
3. Expose the new field in one of the `to_*_context()` methods

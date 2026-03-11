import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class InstructedRetrieverConfiguration(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Databricks — required for vector search; also used for hosted reranker if configured
    databricks_host: str
    databricks_client_id: str
    databricks_client_secret: SecretStr

    # MLflow
    mlflow_experiment_id: str | None = None

    # Models
    # Use any LiteLLM-compatible model string, e.g. "openai/gpt-4o" or
    # "databricks/databricks-meta-llama-3-3-70b-instruct".
    embedding_model: str
    query_rewriter_model: str
    answer_generator_model: str
    optimized_prompt_path: str | None = None
    system_specs_path: str | None = None
    enable_instructed_retrieval: bool = False
    enable_llm_reranking: bool = False
    reranker_model: str | None = None

    # Vector Search (Databricks)
    vs_endpoint: str
    vs_index_name: str

    # Model Registry — only needed for deployment
    model_catalog: str | None = None
    model_schema: str | None = None
    model_name: str | None = None
    endpoint_name: str | None = None

    # Observability
    mlflow_trace_export_to_uc: bool = False
    otel_catalog: str | None = None
    otel_schema: str | None = None
    redact_pii: bool = False

    def configure_trace_export_to_uc(self) -> None:
        os.environ["MLFLOW_TRACKING_URI"] = "databricks"
        os.environ["MLFLOW_REGISTRY_URI"] = "databricks-uc"

        if not self.mlflow_trace_export_to_uc:
            return

        if self.otel_catalog is None or self.otel_schema is None:
            raise ValueError(
                "Both otel_catalog and otel_schema must be set when "
                "mlflow_trace_export_to_uc=true."
            )
        os.environ["MLFLOW_TRACING_DESTINATION"] = f"{self.otel_catalog}.{self.otel_schema}"


_config: InstructedRetrieverConfiguration | None = None


def get_config(env_path: str | Path | None = None) -> InstructedRetrieverConfiguration:
    global _config
    if _config is None:
        env_file = Path(env_path) if env_path else Path(".env")
        if env_file.exists():
            logger.info("Loading configuration from %s", env_file.absolute())
            load_dotenv(env_file, override=True)
            _config = InstructedRetrieverConfiguration(
                _env_file=str(env_file),
                _env_file_encoding="utf-8",
                _env_ignore_empty=True,
            )
        else:
            logger.info(
                ".env not found at %s, using environment variables only", env_file.absolute()
            )
            _config = InstructedRetrieverConfiguration(_env_ignore_empty=True)

        _config.configure_trace_export_to_uc()
    else:
        logger.debug("Returning cached configuration")
    return _config

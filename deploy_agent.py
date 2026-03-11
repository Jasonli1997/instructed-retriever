"""
Package the agent with MLflow and optionally register it in a Unity Catalog model registry.

Usage:
    poetry run python deploy_agent.py

Set MODEL_CATALOG, MODEL_SCHEMA, MODEL_NAME in .env to register the model after packaging.
"""

import logging
from pathlib import Path

import mlflow
from dotenv import dotenv_values
from mlflow.models.model import ModelInfo

from instructed_retriever.agent.config import get_config
from instructed_retriever.responses_agent import InstructedRetrieverResponsesAgent

SPACY_MODEL_WHEEL_URL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl"
)
# Path prefix used when the model is loaded inside an MLflow serving environment.
MLFLOW_ENV_PREFIX = "/model/code/"

logger = logging.getLogger("deploy_agent")
logging.basicConfig(level=logging.INFO)

config = get_config()


def deploy_responses_agent(model_info: ModelInfo) -> None:
    if not all([config.model_catalog, config.model_schema, config.model_name]):
        logger.info(
            "MODEL_CATALOG, MODEL_SCHEMA, and MODEL_NAME are not all set — "
            "skipping model registration."
        )
        return

    uc_model_name = f"{config.model_catalog}.{config.model_schema}.{config.model_name}"

    env_path = Path(".env")
    env_vars: dict[str, str] = {}
    if env_path.exists():
        raw = dotenv_values(str(env_path))
        env_vars = {k: v for k, v in raw.items() if v is not None}

        # Remap local paths to in-container paths for optimized prompts and system specs
        if env_vars.get("OPTIMIZED_PROMPT_PATH"):
            env_vars["OPTIMIZED_PROMPT_PATH"] = MLFLOW_ENV_PREFIX + env_vars[
                "OPTIMIZED_PROMPT_PATH"
            ].lstrip("/")
        if env_vars.get("SYSTEM_SPECS_PATH"):
            env_vars["SYSTEM_SPECS_PATH"] = MLFLOW_ENV_PREFIX + env_vars[
                "SYSTEM_SPECS_PATH"
            ].lstrip("/")
        else:
            env_vars["SYSTEM_SPECS_PATH"] = MLFLOW_ENV_PREFIX + "artifacts/system_specs.yaml"

    uc_registered = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=uc_model_name,
    )
    logger.info(
        "Registered model: %s version %s", uc_model_name, uc_registered.version
    )


def main() -> None:
    input_example = {"input": [{"role": "user", "content": "Who are you?"}]}
    test_agent = InstructedRetrieverResponsesAgent(config)
    result = test_agent.predict(input_example)  # type: ignore[arg-type]
    logger.info("Test prediction: %s", result.model_dump(exclude_none=True))

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model="src/instructed_retriever/responses_agent.py",
            name="agent",
            input_example=input_example,
            extra_pip_requirements=[
                f"en-core-web-md @ {SPACY_MODEL_WHEEL_URL}",
            ],
            code_paths=[
                "src/instructed_retriever",
                "artifacts/",
            ],
        )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    logger.info("Model loaded successfully: %s", loaded_model)

    test_result = loaded_model.predict(input_example)
    logger.info("Test prediction from loaded model: %s", test_result)

    deploy_responses_agent(model_info)


if __name__ == "__main__":
    main()

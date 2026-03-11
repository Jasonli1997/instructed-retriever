#!/usr/bin/env python3
"""Evaluation script — runs MLflow evaluation against a golden dataset in Unity Catalog."""

import argparse
from pathlib import Path
from time import strftime

import mlflow
from mlflow.genai.datasets import get_dataset
from mlflow.genai.evaluation.entities import EvaluationResult
from mlflow.genai.scorers import Correctness, RetrievalGroundedness, RetrievalSufficiency
from mlflow.types.responses import ResponsesAgentRequest

from eval.scorers import category_accuracy
from instructed_retriever.agent.config import get_config

config = get_config(Path(__file__).resolve().parents[1] / ".env")

# Import AGENT after config is initialized so responses_agent.py reuses the cached config.
from instructed_retriever.responses_agent import AGENT  # noqa: E402


@mlflow.trace
def predict_fn(request: str) -> str:
    payload = ResponsesAgentRequest(input=[{"role": "user", "content": request}])
    result = AGENT.predict(payload)

    if result.custom_outputs and (category := result.custom_outputs.get("category")):
        mlflow.update_current_trace(tags={"category": category})

    return result.output[0].content[0]["text"]  # type: ignore[attr-defined, no-any-return]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Instructed Retriever evaluation against a golden dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset table name (without catalog/schema prefix).",
    )
    parser.add_argument(
        "--schema",
        default="db_eval",
        help="Unity Catalog schema containing the dataset.",
    )
    return parser.parse_args()


def main() -> EvaluationResult:
    args = parse_args()
    table_name = f"{config.model_catalog}.{args.schema}.{args.dataset}"
    eval_dataset = get_dataset(table_name)

    run_name = f"instructed_retriever_eval_{strftime('%Y_%m_%d_%H_%M')}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("query_rewriter", config.query_rewriter_model)
        mlflow.log_param("answer_generator", config.answer_generator_model)
        mlflow.log_param("vector_index", config.vs_index_name)
        mlflow.log_param("enable_instructed_retrieval", config.enable_instructed_retrieval)
        mlflow.log_param("enable_llm_reranking", config.enable_llm_reranking)
        mlflow.log_param("reranker_model", config.reranker_model)

        results = mlflow.genai.evaluate(
            data=eval_dataset,
            predict_fn=predict_fn,
            scorers=[
                Correctness(),
                RetrievalSufficiency(),
                RetrievalGroundedness(),
                category_accuracy,
            ],
        )

        print("\n✓ Evaluation complete!")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Experiment ID: {run.info.experiment_id}")

    print("\nMetrics:")
    print(results.metrics)
    print(f"\nResults shape: {results.tables['eval_results'].shape}")

    return results


if __name__ == "__main__":
    main()

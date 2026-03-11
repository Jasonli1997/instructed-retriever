"""Custom MLflow scorers for Instructed Retriever evaluation."""

from mlflow.entities import Feedback, Trace
from mlflow.genai.scorers import scorer


@scorer(
    name="Category Accuracy",
    description="Evaluate if the predicted category matches the expected category.",
)
def category_accuracy(trace: Trace, expectations: dict) -> Feedback:
    """Judge whether the agent's predicted category matches the expected_category label.

    The predicted category is read from the trace tag set by ``predict_fn`` via
    ``mlflow.update_current_trace(tags={"category": ...})``.

    Dataset expectations must include an ``expected_category`` key.
    """
    predicted = (trace.info.tags or {}).get("category")
    expected = (expectations or {}).get("expected_category")
    processed_expected = expected.strip().lower().replace(" ", "_") if expected else None

    if processed_expected is None:
        return Feedback(
            value=None,
            rationale="No expected_category provided in expectations.",
        )
    if not predicted:
        return Feedback(
            value="no",
            rationale="Agent did not return a category.",
        )

    match = predicted.strip().lower() == processed_expected
    return Feedback(
        value="yes" if match else "no",
        rationale=f"Predicted '{predicted}', expected '{expected}'.",
    )

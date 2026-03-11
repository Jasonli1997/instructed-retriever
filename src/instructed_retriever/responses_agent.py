import json
import time
from collections.abc import Generator
from typing import Any, TypedDict
from uuid import NAMESPACE_DNS, uuid3, uuid4

import dspy
import mlflow
from dspy.streaming import StatusMessage, StatusMessageProvider, StreamListener, StreamResponse
from mlflow.pyfunc.model import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from instructed_retriever.agent.config import InstructedRetrieverConfiguration, get_config
from instructed_retriever.agent.context import RunContext
from instructed_retriever.agent.runner import InstructedRetrieverRunner
from instructed_retriever.redact import redact_span_pii


class AgentState(TypedDict):
    last_tool_call_id: str | None
    item_id: str | None
    query_ts: str | None


class ToolStatusMessageProvider(StatusMessageProvider):
    """Custom tool-calling status messages for DSPy streaming."""

    def tool_start_status_message(self, instance: dspy.Tool, inputs: dict) -> str:
        return json.dumps({"tool_name": instance.name, "tool_args": inputs["kwargs"]})

    def tool_end_status_message(self, outputs) -> str:  # type: ignore[no-untyped-def]
        return json.dumps({"tool_result": outputs.value})


class InstructedRetrieverResponsesAgent(ResponsesAgent):
    def __init__(self, agent_settings: InstructedRetrieverConfiguration) -> None:
        self.runner = InstructedRetrieverRunner(agent_settings)
        if agent_settings.optimized_prompt_path is not None:
            try:
                self.runner.load(agent_settings.optimized_prompt_path)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Failed to load optimized prompt from "
                    f"'{agent_settings.optimized_prompt_path}': file not found."
                ) from e
            except json.JSONDecodeError as e:
                message = (
                    f"Malformed JSON in optimized prompt file "
                    f"'{agent_settings.optimized_prompt_path}': {e.msg} "
                    f"(line {e.lineno}, column {e.colno})."
                )
                raise json.JSONDecodeError(message, e.doc, e.pos) from e

        self._streamified_runner = dspy.streamify(
            self.runner,
            status_message_provider=ToolStatusMessageProvider(),
            stream_listeners=[
                StreamListener(signature_field_name="answer", allow_reuse=True),
            ],
            is_async_program=True,
            async_streaming=False,
        )

    def _dspy_stream_chunk_to_responses(
        self,
        chunk: StatusMessage | StreamResponse,
        state: AgentState,
    ) -> dict[str, Any] | None:
        """Convert a DSPy streaming chunk to a Responses API output item dict."""
        if isinstance(chunk, StatusMessage):
            message_dict = json.loads(chunk.message)
            if "tool_name" in message_dict:
                state["last_tool_call_id"] = str(uuid4())
                return self.create_function_call_item(
                    id=str(uuid4()),
                    call_id=state["last_tool_call_id"],  # type: ignore[arg-type]
                    name=message_dict["tool_name"],
                    arguments=json.dumps(message_dict["tool_args"]),
                )
            elif "tool_result" in message_dict:
                call_id = state["last_tool_call_id"]
                state["last_tool_call_id"] = None
                return self.create_function_call_output_item(
                    call_id=call_id,  # type: ignore[arg-type]
                    output=message_dict["tool_result"],
                )

        elif isinstance(chunk, StreamResponse):
            stream_chunk = self.create_text_delta(
                delta=chunk.chunk,
                item_id=str(
                    uuid3(
                        NAMESPACE_DNS,
                        f"{chunk.predict_name}.{chunk.signature_field_name}.query_timestamp_{state['query_ts']}",  # noqa: E501
                    )
                ),
            )
            if state["item_id"] is None:
                state["item_id"] = stream_chunk["item_id"]
            return stream_chunk
        return None

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:  # type: ignore[override]
        outputs = [
            event
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        prediction = outputs[-1] if outputs else None
        return ResponsesAgentResponse(
            output=[prediction.item] if prediction else [],  # type: ignore[attr-defined]
            custom_outputs=prediction.custom_outputs if prediction else None,
        )

    def predict_stream(  # type: ignore[override]
        self,
        request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        last_message = request.input[-1]
        if last_message.role != "user":  # type: ignore[union-attr]
            raise ValueError("The last message in the input must have role 'user'.")

        state: AgentState = {
            "last_tool_call_id": None,
            "item_id": None,
            "query_ts": str(time.time()),
        }
        query = last_message.content  # type: ignore[union-attr]
        context = self.prepare_run_context(request)
        mlflow.update_current_trace(metadata=request.custom_inputs or {})

        output = self._streamified_runner(query=query, context=context)
        for chunk in output:
            converted_chunk = self._dspy_stream_chunk_to_responses(chunk, state)
            if isinstance(chunk, StatusMessage):
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=converted_chunk,
                )
            elif isinstance(chunk, StreamResponse):
                yield ResponsesAgentStreamEvent(**converted_chunk)  # type: ignore[arg-type]
            elif isinstance(chunk, dspy.Prediction):
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=self.create_text_output_item(
                        text=chunk.answer,
                        id=state["item_id"] if state["item_id"] is not None else str(uuid4()),
                    ),
                    custom_outputs={"category": chunk.category} if chunk.category else None,
                )

    def prepare_run_context(self, request: ResponsesAgentRequest) -> RunContext:
        messages = []
        for message in request.input[:-1]:
            if message.role == "user":  # type: ignore[union-attr]
                messages.append({"query": message.content})  # type: ignore[union-attr]
            elif message.role == "assistant":  # type: ignore[union-attr]
                if not messages:
                    raise ValueError("Assistant message found without a preceding user message.")
                messages.append({"answer": message.content})  # type: ignore[union-attr]
        return RunContext(chat_history=dspy.History(messages=messages))


config = get_config()

mlflow.dspy.autolog()
mlflow.litellm.autolog()
if config.redact_pii:
    mlflow.tracing.configure(span_processors=[redact_span_pii])

AGENT = InstructedRetrieverResponsesAgent(config)
mlflow.models.set_model(AGENT)

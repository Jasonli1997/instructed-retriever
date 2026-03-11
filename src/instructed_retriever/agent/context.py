import dspy
from pydantic import BaseModel


class RunContext(BaseModel):
    """Runtime context for a single agent request."""

    chat_history: dspy.History = dspy.History(messages=[])

    class Config:
        arbitrary_types_allowed = True

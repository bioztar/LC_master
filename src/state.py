"""Typed graph state. LangGraph reduces dict updates per node return."""
from typing import Annotated, Literal, TypedDict
from operator import add
from langchain_core.documents import Document


Category = Literal["refunds", "shipping", "accounts", "escalate", "unknown"]


class TriageState(TypedDict, total=False):
    # Input
    question: str
    user_id: str

    # Classifier output
    category: Category
    confidence: float
    reasoning: str

    # Retrieval
    docs: list[Document]

    # Final answer
    answer: str

    # HITL approval (set by user via Command.resume after interrupt)
    human_approved: bool
    human_note: str

    # Accumulating trace (append-only)
    trail: Annotated[list[str], add]

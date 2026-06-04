"""Compile LangGraph with SqliteSaver checkpoint.

Swap to PostgresSaver for prod:
    from langgraph.checkpoint.postgres import PostgresSaver
    saver = PostgresSaver.from_conn_string(os.environ["POSTGRES_DSN"]); saver.setup()
"""
from pathlib import Path
from contextlib import contextmanager

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import CHECKPOINT_DB
from src.state import TriageState
from src.nodes import (
    classify_node, rag_node, escalate_node, unknown_node, route_after_classify,
)


def _build_graph():
    g = StateGraph(TriageState)
    g.add_node("classify", classify_node)
    g.add_node("rag", rag_node)
    g.add_node("escalate", escalate_node)
    g.add_node("unknown", unknown_node)

    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify", route_after_classify,
        {"rag": "rag", "escalate": "escalate", "unknown": "unknown"},
    )
    g.add_edge("rag", END)
    g.add_edge("escalate", END)
    g.add_edge("unknown", END)
    return g


@contextmanager
def compiled_graph():
    """Context manager: opens SqliteSaver, yields compiled graph, closes cleanly."""
    Path(CHECKPOINT_DB).parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as saver:
        yield _build_graph().compile(checkpointer=saver)

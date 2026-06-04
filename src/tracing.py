"""Langfuse callback factory. No-op if creds missing — local dev still works."""
from typing import Optional
from src.config import (
    LANGFUSE_ENABLED, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST,
)


def langfuse_callbacks(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> dict:
    """Returns a `config` dict to pass into graph.invoke(..., config=...).

    Wires: LangChain callback handler + LangGraph trace metadata.
    """
    if not LANGFUSE_ENABLED:
        return {"configurable": {"thread_id": session_id or "default"}}

    from langfuse.callback import CallbackHandler
    handler = CallbackHandler(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST,
        session_id=session_id,
        user_id=user_id,
        tags=tags or [],
    )
    return {
        "callbacks": [handler],
        "configurable": {"thread_id": session_id or "default"},
        "metadata": {"langfuse_session_id": session_id, "langfuse_user_id": user_id},
    }


def flush_langfuse() -> None:
    if not LANGFUSE_ENABLED:
        return
    from langfuse import Langfuse
    Langfuse().flush()

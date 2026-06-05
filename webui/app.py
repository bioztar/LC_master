"""Local Flask dashboard for the LangGraph triage agent.

Routes
------
GET  /                  index — graph diagram + list of all threads from checkpoints
GET  /thread/<id>       per-thread timeline: state at every checkpoint, can rewind
GET  /run               form: submit a new question
POST /run               execute graph, redirect to /thread/<new_thread_id>
POST /resume/<id>       resume a paused (HITL) thread with approval decision
POST /rewind/<id>       fork from a chosen checkpoint (time-travel)
GET  /visual_layers     standalone explainer page

Run: python -m webui.app   →   http://localhost:5000
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from flask import Flask, render_template, request, redirect, url_for, abort
from langgraph.types import Command

from src.graph import compiled_graph
from src.tracing import langfuse_callbacks, flush_langfuse
from src.config import LANGFUSE_ENABLED, LANGFUSE_HOST

app = Flask(__name__, template_folder="templates", static_folder="static")


# ---------- Helpers ----------
def _list_threads() -> list[dict[str, Any]]:
    """Return summary of every checkpointed thread (latest snapshot per thread)."""
    counts: dict[str, int] = {}
    with compiled_graph() as g:
        # Pass 1: discover thread IDs + count checkpoints (CheckpointTuple has no .values)
        for ct in g.checkpointer.list(None):
            tid = ct.config["configurable"]["thread_id"]
            counts[tid] = counts.get(tid, 0) + 1
        # Pass 2: fetch latest StateSnapshot per thread for typed access
        results = []
        for tid, n in counts.items():
            snap = g.get_state({"configurable": {"thread_id": tid}})
            v = snap.values or {}
            results.append({
                "thread_id": tid,
                "checkpoints": n,
                "last_question": v.get("question"),
                "last_category": v.get("category"),
                "last_answer": v.get("answer"),
                "paused": bool(snap.next),
                "next_nodes": list(snap.next or []),
                "trail": v.get("trail", []),
            })
    return sorted(results, key=lambda x: x["thread_id"], reverse=True)


def _thread_history(thread_id: str) -> list[dict[str, Any]]:
    """Full ordered checkpoint history for one thread, oldest → newest.

    Adds `node` (which node wrote this checkpoint) extracted from metadata.writes.
    """
    cfg = {"configurable": {"thread_id": thread_id}}
    history = []
    with compiled_graph() as g:
        for snap in g.get_state_history(cfg):
            writes = (snap.metadata or {}).get("writes") or {}
            node_names = [k for k in writes.keys() if k is not None]
            history.append({
                "checkpoint_id": snap.config["configurable"]["checkpoint_id"],
                "node": ", ".join(node_names) if node_names else "(start)",
                "step": (snap.metadata or {}).get("step", 0),
                "source": (snap.metadata or {}).get("source", ""),
                "next": list(snap.next or []),
                "state": dict(snap.values),
                "tasks": [
                    {
                        "name": t.name,
                        "interrupts": [
                            i.value if hasattr(i, "value") else str(i) for i in (t.interrupts or [])
                        ],
                    }
                    for t in (snap.tasks or [])
                ],
                "created_at": snap.created_at,
            })
    history.reverse()  # oldest first
    return history


# ---------- Routes ----------
@app.route("/")
def index():
    threads = _list_threads()
    graph_mmd = Path("webui/static/graph.mmd")
    if not graph_mmd.exists():
        from scripts.render_graph import main as render
        render()
    return render_template(
        "index.html",
        threads=threads,
        mermaid=graph_mmd.read_text(),
        langfuse_enabled=LANGFUSE_ENABLED,
        langfuse_host=LANGFUSE_HOST,
    )


@app.route("/thread/<thread_id>")
def thread(thread_id: str):
    history = _thread_history(thread_id)
    if not history:
        abort(404)
    return render_template("thread.html", thread_id=thread_id, history=history)


@app.route("/run", methods=["GET", "POST"])
def run():
    if request.method == "GET":
        return render_template("run.html")

    question = request.form.get("question", "").strip()
    if not question:
        return redirect(url_for("run"))
    thread_id = f"web-{uuid.uuid4().hex[:8]}"
    cfg = langfuse_callbacks(session_id=thread_id, user_id="webui", tags=["webui"])
    with compiled_graph() as g:
        g.invoke({"question": question, "user_id": "webui"}, config=cfg)
    flush_langfuse()
    return redirect(url_for("thread", thread_id=thread_id))


@app.route("/resume/<thread_id>", methods=["POST"])
def resume(thread_id: str):
    approved = request.form.get("approved") == "yes"
    note = request.form.get("note", "")
    cfg = langfuse_callbacks(session_id=thread_id, user_id="webui", tags=["webui", "resume"])
    with compiled_graph() as g:
        g.invoke(Command(resume={"approved": approved, "note": note}), config=cfg)
    flush_langfuse()
    return redirect(url_for("thread", thread_id=thread_id))


@app.route("/rewind/<thread_id>", methods=["POST"])
def rewind(thread_id: str):
    """Time-travel: fork a new thread starting from the chosen checkpoint."""
    checkpoint_id = request.form["checkpoint_id"]
    new_thread = f"fork-{uuid.uuid4().hex[:6]}-from-{thread_id[-6:]}"
    src_cfg = {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
    with compiled_graph() as g:
        snap = g.get_state(src_cfg)
        new_cfg = {"configurable": {"thread_id": new_thread}}
        g.update_state(new_cfg, snap.values)
    return redirect(url_for("thread", thread_id=new_thread))


@app.route("/visual_layers")
def visual_layers():
    return render_template("visual_layers.html",
                           langfuse_enabled=LANGFUSE_ENABLED,
                           langfuse_host=LANGFUSE_HOST)


if __name__ == "__main__":
    import os
    port = int(os.getenv("FLASK_RUN_PORT", "5050"))  # 5050 default (macOS AirPlay grabs 5000)
    app.run(host="127.0.0.1", port=port, debug=True)

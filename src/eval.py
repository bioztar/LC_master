"""Offline eval harness.

Runs each eval row through the graph (auto-rejects HITL escalations to keep it offline),
scores via:
  - category match (deterministic)
  - must_contain substring check (deterministic)
  - LLM-as-judge groundedness (1 LLM call per row)

Pushes scores to Langfuse if enabled.

Run: python -m src.eval
"""
import json
import uuid
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command

from src.config import LANGFUSE_ENABLED
from src.llm import get_chat
from src.graph import compiled_graph
from src.tracing import langfuse_callbacks, flush_langfuse


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an evaluator. Score the assistant answer on 'groundedness' from 0.0-1.0: "
     "1.0 = fully supported by the knowledge base, 0.0 = hallucinated or off-topic. "
     "Respond JSON only: {{\"score\": float, \"reason\": \"...\"}}"),
    ("human",
     "Question: {q}\n\nAssistant answer: {a}\n\nExpected category: {cat}"),
])


def _judge(question: str, answer: str, expected: str) -> tuple[float, str]:
    from src.nodes import _parse_json
    llm = get_chat(temperature=0.0, json_mode=True)
    chain = JUDGE_PROMPT | llm | (lambda m: _parse_json(m.content))
    out = chain.invoke({"q": question, "a": answer, "cat": expected})
    return float(out["score"]), out["reason"]


def _push_score(handler, name: str, value: float, comment: str = "") -> None:
    if not LANGFUSE_ENABLED or handler is None:
        return
    try:
        trace_id = handler.get_trace_id()
        from langfuse import Langfuse
        Langfuse().score(trace_id=trace_id, name=name, value=value, comment=comment)
    except Exception as e:
        print(f"  [warn] score push failed: {e}")


def run() -> None:
    rows = [json.loads(l) for l in Path("data/eval_dataset.jsonl").read_text().splitlines() if l.strip()]
    results = []

    with compiled_graph() as graph:
        for row in rows:
            session_id = f"eval-{row['id']}-{uuid.uuid4().hex[:6]}"
            cfg = langfuse_callbacks(session_id=session_id, user_id="eval-bot", tags=["eval", row["id"]])
            handler = cfg.get("callbacks", [None])[0] if cfg.get("callbacks") else None

            graph.invoke({"question": row["question"]}, config=cfg)
            snap = graph.get_state(cfg)
            if snap.next:  # paused on HITL → auto-reject for offline eval
                graph.invoke(Command(resume={"approved": False, "note": "eval-bot"}), config=cfg)
                snap = graph.get_state(cfg)

            v = snap.values
            cat_match = float(v.get("category") == row["expected_category"])
            answer = v.get("answer", "")
            contained = all(s.lower() in answer.lower() for s in row.get("must_contain", []))
            contains_score = float(contained)

            judge_score, judge_reason = _judge(row["question"], answer, row["expected_category"])

            _push_score(handler, "category_match", cat_match)
            _push_score(handler, "contains_required", contains_score)
            _push_score(handler, "groundedness", judge_score, judge_reason)

            results.append({
                "id": row["id"],
                "category_match": cat_match,
                "contains_required": contains_score,
                "groundedness": judge_score,
            })
            print(f"{row['id']}: cat={cat_match} contains={contains_score} ground={judge_score:.2f}")

    flush_langfuse()

    n = len(results)
    avg = lambda k: sum(r[k] for r in results) / n if n else 0.0
    print("\n=== SUMMARY ===")
    print(f"category_match    : {avg('category_match'):.2%}")
    print(f"contains_required : {avg('contains_required'):.2%}")
    print(f"groundedness (avg): {avg('groundedness'):.2f}")


if __name__ == "__main__":
    run()

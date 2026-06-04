"""Interactive CLI for the triage agent. Demonstrates HITL resume.

Run:
    python -m src.main                 # interactive
    python -m src.main "where is my refund for ORD-123456?"   # one-shot
"""
import sys
import uuid

from langgraph.types import Command

from src.graph import compiled_graph
from src.tracing import langfuse_callbacks, flush_langfuse


def _print_state(snapshot, label: str = "state") -> None:
    v = snapshot.values
    print(f"\n--- {label} ---")
    print(f"  category   = {v.get('category')}")
    print(f"  confidence = {v.get('confidence')}")
    print(f"  trail      = {v.get('trail')}")
    if v.get("answer"):
        print(f"\nANSWER:\n{v['answer']}\n")


def run_once(question: str, user_id: str = "demo-user") -> None:
    session_id = f"sess-{uuid.uuid4().hex[:8]}"
    cfg = langfuse_callbacks(session_id=session_id, user_id=user_id, tags=["cli"])

    with compiled_graph() as graph:
        result = graph.invoke({"question": question, "user_id": user_id}, config=cfg)

        # Detect interrupt (HITL pause)
        snapshot = graph.get_state(cfg)
        if snapshot.next:  # graph paused on an interrupt
            interrupts = snapshot.tasks[0].interrupts if snapshot.tasks else []
            payload = interrupts[0].value if interrupts else {}
            print("\n*** HUMAN-IN-THE-LOOP PAUSE ***")
            print(f"Reason       : {payload.get('type')}")
            print(f"Question     : {payload.get('question')}")
            print(f"Classifier   : {payload.get('category')} — {payload.get('reasoning')}")
            choice = input("\nApprove escalation? [y/N]: ").strip().lower()
            note = input("Note for case file (optional): ").strip()
            decision = {"approved": choice == "y", "note": note}

            result = graph.invoke(Command(resume=decision), config=cfg)
            snapshot = graph.get_state(cfg)

        _print_state(snapshot, label="final")
        print(f"(session_id={session_id})")

    flush_langfuse()


def interactive() -> None:
    print("Support Triage Agent — type 'quit' to exit.\n")
    while True:
        q = input("you> ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            return
        if not q:
            continue
        run_once(q)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_once(" ".join(sys.argv[1:]))
    else:
        interactive()

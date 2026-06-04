"""Graph nodes. Each is a pure-ish function: state in → state delta out."""
import json
import re
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt

from src.llm import get_chat
from src.retriever import get_retriever
from src.state import TriageState


def _parse_json(content: str) -> dict:
    """Tolerant JSON parser — strips ```json fences some models add."""
    s = content.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return json.loads(s)


# ------------- Classifier (supervisor) -------------
CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a support-ticket classifier. Categories: refunds, shipping, accounts, escalate, unknown.\n"
     "Pick 'escalate' if: legal threat, GDPR deletion, refund > $500, account compromise, or angry tone.\n"
     "Pick 'unknown' if outside support scope.\n"
     "Respond ONLY as JSON: {{\"category\": ..., \"confidence\": 0.0-1.0, \"reasoning\": \"...\"}}"),
    ("human", "{question}"),
])


def classify_node(state: TriageState) -> dict[str, Any]:
    chain = CLASSIFY_PROMPT | get_chat(temperature=0.0, json_mode=True) | (lambda m: _parse_json(m.content))
    out = chain.invoke({"question": state["question"]})
    return {
        "category": out["category"],
        "confidence": float(out["confidence"]),
        "reasoning": out["reasoning"],
        "trail": [f"classified={out['category']} conf={out['confidence']:.2f}"],
    }


# ------------- RAG node -------------
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a support agent. Answer ONLY from the context below. "
     "If insufficient, say 'I don't have that information — escalating.'\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def rag_node(state: TriageState) -> dict[str, Any]:
    retriever = get_retriever(k=4)
    docs = retriever.invoke(state["question"])
    context = "\n\n---\n\n".join(
        f"[{d.metadata.get('source','?')} / {d.metadata.get('h2','')}] {d.page_content}"
        for d in docs
    )
    chain = RAG_PROMPT | get_chat(temperature=0.2)
    msg = chain.invoke({"context": context, "question": state["question"]})
    return {
        "docs": docs,
        "answer": msg.content,
        "trail": [f"rag retrieved={len(docs)} chunks"],
    }


# ------------- Escalate node with HITL -------------
def escalate_node(state: TriageState) -> dict[str, Any]:
    payload = {
        "type": "escalation_approval",
        "question": state["question"],
        "category": state.get("category"),
        "reasoning": state.get("reasoning"),
        "user_id": state.get("user_id"),
    }
    decision = interrupt(payload)

    approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
    note = decision.get("note", "") if isinstance(decision, dict) else ""

    if not approved:
        return {
            "human_approved": False,
            "human_note": note,
            "answer": f"Request reviewed by agent; closed without escalation. Note: {note or 'n/a'}",
            "trail": ["escalation rejected by human"],
        }

    llm = get_chat(temperature=0.3)
    msg = llm.invoke([
        SystemMessage(content="Draft a brief, empathetic escalation acknowledgment. Mention case will be reviewed by a specialist within 1 business day."),
        HumanMessage(content=f"Customer question: {state['question']}\nHuman note: {note}"),
    ])
    return {
        "human_approved": True,
        "human_note": note,
        "answer": msg.content,
        "trail": ["escalation approved + drafted"],
    }


# ------------- Unknown fallback -------------
def unknown_node(state: TriageState) -> dict[str, Any]:
    return {
        "answer": "Your question seems outside our support scope. Try our community forum or rephrase.",
        "trail": ["unknown fallback"],
    }


# ------------- Router -------------
def route_after_classify(state: TriageState) -> str:
    cat = state.get("category", "unknown")
    if cat in ("refunds", "shipping", "accounts") and state.get("confidence", 0.0) < 0.55:
        return "escalate"
    if cat in ("refunds", "shipping", "accounts"):
        return "rag"
    if cat == "escalate":
        return "escalate"
    return "unknown"

# LC_master — Support Triage Agent (LangChain + LangGraph + Langfuse)

End-to-end demo: a **durable, stateful, observable support agent** built on the modern LangChain stack.

> Interview pitch: "I built a supervisor-style LangGraph agent that triages support tickets, routes to a Chroma-backed RAG node or a human-in-the-loop escalation path, persists state with a checkpointer so it survives crashes, traces every step in Langfuse, and ships with an offline LLM-as-judge eval harness."

## Architecture

```
                      ┌──────────────┐
        question ───► │  classify    │   (JSON-mode supervisor LLM)
                      └──────┬───────┘
                             │  conditional edge
        ┌────────────────────┼───────────────────────┐
        ▼                    ▼                       ▼
   ┌─────────┐         ┌──────────┐            ┌──────────┐
   │  rag    │         │ escalate │  ── HITL ─►│ interrupt│
   │ (Chroma │         │ (human   │            │  pause   │
   │  + LLM) │         │  approval│◄── Command(resume=…)
   └────┬────┘         └────┬─────┘            └──────────┘
        │                   │
        └────────► END ◄────┘

State: TypedDict (TriageState), checkpointed every node into SQLite.
Tracing: Langfuse CallbackHandler attached on every invoke.
Eval: deterministic category-match + must-contain + LLM-as-judge groundedness.
```

## Why these choices

| Concern | Choice | Reason |
|---|---|---|
| Orchestration | **LangGraph** | `AgentExecutor` deprecated (EOL Dec 2026). Graph gives explicit branching, retries, HITL. |
| State | TypedDict + `Annotated[list, add]` reducer | Idiomatic LangGraph; reducers avoid merge bugs. |
| Checkpoint | **SqliteSaver** (dev) → swap **PostgresSaver** (prod) | `MemorySaver` loses state on restart; ~60% of prod agent incidents are state mgmt. |
| RAG store | **Chroma** persistent | Lean, local, zero-infra demo. Swap for pgvector / Pinecone trivially. |
| Observability | **Langfuse** (OSS, self-hostable) | Framework-agnostic, OTel-native, fits regulated / EU. LangSmith swap is one import change. |
| Eval | LLM-as-judge + deterministic asserts | Hybrid catches both regressions and hallucinations. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill OPENAI_API_KEY (+ optional LANGFUSE_*)
python -m src.ingest  # embeds docs/*.md → ./data/chroma
```

## Run

```bash
# Interactive
python -m src.main

# One-shot
python -m src.main "Why hasn't my refund of $25 hit my card after 3 days?"

# Trigger HITL escalation
python -m src.main "Delete all my data under GDPR"
# → prompts: Approve escalation? [y/N]

# Run eval suite
python -m src.eval
```

## Hot swaps (no architectural change)

**Postgres checkpointer** (durable, multi-instance):
```python
# src/graph.py
from langgraph.checkpoint.postgres import PostgresSaver
with PostgresSaver.from_conn_string(os.environ["POSTGRES_DSN"]) as saver:
    saver.setup()
    yield _build_graph().compile(checkpointer=saver)
```

**LangSmith instead of Langfuse**:
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=...
# Auto-instruments LangChain/LangGraph. Drop src/tracing.py callbacks.
```

**LlamaIndex retriever instead of Chroma**: replace `src/retriever.py` `get_retriever()` with a `VectorStoreIndex.as_retriever()` wrapped in a LangChain `BaseRetriever` adapter.

## Interview talking points (memorize)

1. **Why LangGraph over AgentExecutor**: durable execution, explicit branching, HITL is first-class via `interrupt()` + `Command(resume=...)`.
2. **State management as #1 prod risk**: chose SqliteSaver for demo with clear Postgres migration path; state is checkpointed *every node transition*, so a crash mid-run resumes exactly where it left off.
3. **Langfuse vs LangSmith**: I picked Langfuse for portability + self-host + OTel; LangSmith would win for a pure-LangChain shop wanting zero-config tracing.
4. **Supervisor pattern**: a JSON-mode classifier acts as router. Low confidence on known categories also escalates — defensive default.
5. **Eval that catches both regressions and hallucinations**: deterministic asserts (category + required substrings) plus LLM-as-judge groundedness, all scored back into Langfuse for trend analysis.
6. **What I'd add next**: streaming intermediate tokens to client, multi-turn memory via thread_id, parallel sub-agent fan-out for complex tickets, DSPy to auto-tune the classifier prompt against the eval set.

## Layout

```
src/
  config.py      # env + paths
  state.py       # TypedDict graph state + reducer
  ingest.py      # docs → Chroma (run once)
  retriever.py   # cached Chroma retriever
  nodes.py       # classify / rag / escalate(HITL) / unknown
  graph.py       # StateGraph + SqliteSaver
  tracing.py     # Langfuse callback factory
  main.py        # CLI w/ HITL resume
  eval.py        # offline eval + LLM judge → Langfuse scores
docs/            # fake support KB
data/            # chroma + sqlite checkpoints (gitignored)
```

## Stack versions
LangChain 0.3 · LangGraph 0.2 · Langfuse 2.60 · Chroma 0.5 · OpenAI gpt-4o-mini + text-embedding-3-small.

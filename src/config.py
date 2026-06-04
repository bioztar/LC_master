"""Centralized config. Reads .env once, exposes typed constants."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"

# Provider switch
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Gemini (AI Studio)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")

# Storage
CHROMA_DIR = os.getenv("CHROMA_DIR", str(ROOT / "data" / "chroma"))
CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", str(ROOT / "data" / "checkpoints.sqlite"))

# Langfuse
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)


def _require(var: str, value: str) -> str:
    if not value:
        raise RuntimeError(
            f"{var} is required for LLM_PROVIDER={LLM_PROVIDER}. Set it in .env"
        )
    return value


def validate() -> None:
    if LLM_PROVIDER == "openai":
        _require("OPENAI_API_KEY", OPENAI_API_KEY)
    elif LLM_PROVIDER == "gemini":
        _require("GOOGLE_API_KEY", GOOGLE_API_KEY)
    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. Use 'openai' or 'gemini'.")

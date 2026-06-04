"""Lazy-loaded Chroma retriever singleton."""
from functools import lru_cache
from langchain_chroma import Chroma

from src.config import CHROMA_DIR
from src.llm import get_embeddings


@lru_cache(maxsize=1)
def get_retriever(k: int = 4):
    store = Chroma(
        collection_name="support_kb",
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )
    return store.as_retriever(search_type="similarity", search_kwargs={"k": k})

"""Lazy-loaded Chroma retriever singleton."""
from functools import lru_cache
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.config import CHROMA_DIR, OPENAI_EMBED_MODEL, OPENAI_API_KEY


@lru_cache(maxsize=1)
def get_retriever(k: int = 4):
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL, api_key=OPENAI_API_KEY)
    store = Chroma(
        collection_name="support_kb",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return store.as_retriever(search_type="similarity", search_kwargs={"k": k})

"""One-shot ingest: load docs/*.md → chunk → embed → persist Chroma store.

Run: python -m src.ingest
"""
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from src.config import DOCS_DIR, CHROMA_DIR
from src.llm import get_embeddings


def load_and_split() -> list:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)

    chunks = []
    for md in sorted(Path(DOCS_DIR).glob("*.md")):
        raw = md.read_text(encoding="utf-8")
        header_chunks = header_splitter.split_text(raw)
        for hc in header_chunks:
            hc.metadata["source"] = md.name
            hc.metadata["topic"] = md.stem
        chunks.extend(char_splitter.split_documents(header_chunks))
    return chunks


def main() -> None:
    chunks = load_and_split()
    print(f"Loaded {len(chunks)} chunks from {DOCS_DIR}")

    embeddings = get_embeddings()
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

    store = Chroma(
        collection_name="support_kb",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    # Idempotent rebuild: wipe collection then add.
    try:
        store.delete_collection()
    except Exception:
        pass
    store = Chroma(
        collection_name="support_kb",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    store.add_documents(chunks)
    print(f"Persisted {len(chunks)} chunks to {CHROMA_DIR}")


if __name__ == "__main__":
    main()

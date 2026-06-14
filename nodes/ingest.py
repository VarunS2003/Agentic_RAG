"""
Ingest node — loads PDF(s), splits into chunks, embeds them, stores in Chroma.

Also builds the in-memory BM25 index from the same chunks.
Returns (vectorstore, bm25_store) so the graph can pass them to the retrieve node.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rich.console import Console

from config.settings import (
    CHUNK_SIZE, CHUNK_OVERLAP,
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION,
)
from utils.bm25_store import BM25Store

console = Console()


def load_documents(file_path: str) -> List[Document]:
    """Load a PDF or plain-text file into LangChain Documents."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        loader = TextLoader(str(path))

    return loader.load()


def ingest_documents(file_path: str) -> Tuple[Chroma, BM25Store]:
    """
    Full ingest pipeline:
    1. Load file
    2. Split into chunks
    3. Embed and store in Chroma
    4. Build BM25 index from the same chunks
    """
    print(f"[ingest_documents] starting...")
    console.print(f"[bold cyan]📄 Loading:[/] {file_path}")
    raw_docs = load_documents(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(raw_docs)
    console.print(f"[green]✓[/] Split into {len(chunks)} chunks")

    from config.models import get_embedding_function
    embeddings = get_embedding_function()

    # Chroma will persist to disk — re-running won't re-embed unless you delete the dir
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION,
    )
    console.print(f"[green]✓[/] Embedded and stored in Chroma ({CHROMA_PERSIST_DIR})")

    bm25 = BM25Store(chunks)
    console.print(f"[green]✓[/] BM25 index built ({len(chunks)} docs)")

    return vectorstore, bm25


def load_existing_vectorstore() -> Chroma:
    """Load an already-persisted Chroma collection (no re-embedding)."""
    from config.models import get_embedding_function
    embeddings = get_embedding_function()
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION,
    )

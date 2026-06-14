"""
Entry point for the agentic RAG pipeline.

Usage:
  # ingest a PDF and ask a question in one shot
  python main.py --pdf docs/paper.pdf --query "What is the main finding?"

  # ask a question using an already-ingested vector store
  python main.py --query "Summarise the methodology"

  # interactive mode — keeps asking until you type 'exit'
  python main.py --pdf docs/paper.pdf --interactive
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from nodes.ingest import ingest_documents, load_existing_vectorstore
from utils.bm25_store import BM25Store
from graph import build_graph
from state import RAGState
from config.settings import CHROMA_PERSIST_DIR

console = Console()


def run_query(pipeline, question: str) -> str:
    """Run a single question through the pipeline and return the answer."""
    initial_state: RAGState = {
        "question": question,
        "rewritten_query": "",
        "documents": [],
        "web_results": [],
        "answer": "",
        "needs_web_search": False,
        "answer_is_grounded": False,
        "retry_count": 0,
    }

    console.print(f"\n[bold]❓ Question:[/] {question}\n")

    final_state = pipeline.invoke(initial_state)

    answer = final_state.get("answer", "No answer generated.")
    rewritten = final_state.get("rewritten_query", "")
    n_docs = len(final_state.get("documents", []))
    used_web = bool(final_state.get("web_results"))

    # show a brief trace
    console.print(f"  [dim]↳ Rewritten query:[/dim] {rewritten}")
    console.print(f"  [dim]↳ Docs used: {n_docs}  |  Web search: {'yes' if used_web else 'no'}[/dim]\n")
    console.print(Panel(Markdown(answer), title="[bold green]Answer", border_style="green"))

    return answer


def main():
    parser = argparse.ArgumentParser(description="Agentic RAG — local LLM pipeline")
    parser.add_argument("--pdf",         type=str, help="Path to a PDF (or .txt) to ingest")
    parser.add_argument("--query",       type=str, help="Single question to answer")
    parser.add_argument("--interactive", action="store_true", help="Interactive Q&A loop")
    args = parser.parse_args()

    # ── Ingest or load ─────────────────────────────────────────────────────────
    if args.pdf:
        vectorstore, bm25_store = ingest_documents(args.pdf)
    else:
        chroma_path = Path(CHROMA_PERSIST_DIR)
        if not chroma_path.exists():
            console.print("[red]No vector store found. Pass --pdf to ingest a document first.[/]")
            sys.exit(1)
        console.print(f"[cyan]Loading existing vector store from {CHROMA_PERSIST_DIR}[/]")
        vectorstore = load_existing_vectorstore()
        # rebuild BM25 from Chroma docs (fast for small collections)
        all_docs = vectorstore.get()
        from langchain_core.documents import Document as LCDoc
        docs = [
            LCDoc(page_content=text, metadata=meta)
            for text, meta in zip(all_docs["documents"], all_docs["metadatas"])
        ]
        bm25_store = BM25Store(docs)
        console.print(f"[green]✓[/] BM25 index rebuilt ({len(docs)} chunks)")

    # ── Build graph ────────────────────────────────────────────────────────────
    pipeline = build_graph(vectorstore, bm25_store)
    console.print("[bold green]✓ Pipeline ready[/]\n")

    # ── Run ────────────────────────────────────────────────────────────────────
    if args.interactive:
        console.print("[bold]Interactive mode — type [cyan]exit[/cyan] to quit[/]\n")
        while True:
            try:
                question = console.input("[bold yellow]You:[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() in ("exit", "quit", "q"):
                break
            if question:
                run_query(pipeline, question)
    elif args.query:
        run_query(pipeline, args.query)
    else:
        console.print("[yellow]Provide --query 'your question' or --interactive[/]")


if __name__ == "__main__":
    main()

"""
Answer generation node.

Formats the graded documents into a context block and asks the LLM
to answer the original question using only that context.

The prompt is intentionally simple — 4b models respond better to
straightforward instructions than to elaborate multi-step prompts.
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from config.models import get_generation_llm
_llm = get_generation_llm()
from state import RAGState

_SYSTEM = """You are a helpful assistant. Answer the question using ONLY the
provided context. If the context doesn't contain enough information to answer,
say so clearly — do not make things up.

Keep answers concise and factual."""


def _format_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "local")
        parts.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n".join(parts)


def generate(state: RAGState) -> RAGState:
    """Generate an answer from the available documents."""
    question = state["question"]
    docs = state.get("documents", [])
    print(f"[generate] question={question[:60]}... docs={len(docs)}")

    context = _format_context(docs) if docs else "No context available."

    prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = _llm.invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=prompt),
    ])

    return {**state, "answer": response.content.strip()}

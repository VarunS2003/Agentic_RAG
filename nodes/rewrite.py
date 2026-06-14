"""
Query rewrite node — improves the user's raw question before retrieval.

A short rewrite step meaningfully improves recall:
- removes filler words
- expands abbreviations
- makes implicit context explicit

We keep the rewrite prompt minimal so the local 4b model handles it reliably.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from config.models import get_rewrite_llm
_llm = get_rewrite_llm()
from state import RAGState


# shared `rewrite_llm` imported above

_SYSTEM = (
    "You are a search query optimizer. "
    "Rewrite the user's question into a clear, concise search query "
    "that captures the key information need. "
    "Return ONLY the rewritten query — no explanation, no quotes."
)


def rewrite_query(state: RAGState) -> RAGState:
    """Rewrite the user question into a better retrieval query."""
    question = state["question"]
    print(f"[rewrite_query] original={question[:60]}...")

    response = _llm.invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"Question: {question}"),
    ])

    rewritten = response.content.strip()
    return {**state, "rewritten_query": rewritten}

"""
Shared state that flows through every node in the LangGraph pipeline.
All nodes read from and write to this TypedDict.
"""

from typing import TypedDict, List, Optional
from langchain_core.documents import Document


class RAGState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────────
    question: str                       # original user question

    # ── Query rewriting ────────────────────────────────────────────────────────
    rewritten_query: str                # cleaned / expanded query

    # ── Retrieval ─────────────────────────────────────────────────────────────
    documents: List[Document]           # retrieved + graded docs
    web_results: List[Document]         # docs from web search (if triggered)

    # ── Generation ────────────────────────────────────────────────────────────
    answer: str                         # final answer

    # ── Control flow ──────────────────────────────────────────────────────────
    needs_web_search: bool              # set by grade_docs if not enough context
    answer_is_grounded: bool            # set by check_answer
    retry_count: int                    # tracks how many times we've regenerated

"""
Document grading node.

Asks the LLM whether each retrieved chunk is actually relevant to the question.
Irrelevant chunks are dropped before generation.

If fewer than MIN_RELEVANT_DOCS chunks pass, the pipeline routes to web search
instead of trying to generate from weak context.
"""

import json
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from config.settings import MIN_RELEVANT_DOCS
from config.models import get_grading_llm
_llm = get_grading_llm()
from state import RAGState



_SYSTEM = """You are a relevance grader. Given a question and a document chunk,
decide whether the chunk contains information useful for answering the question.

Respond ONLY with valid JSON in this exact format:
{"relevant": true}   or   {"relevant": false}

Do not add any other text."""


def _grade_one(llm, question: str, doc: Document) -> bool:
    """Return True if the doc is relevant to the question."""
    prompt = f"Question: {question}\n\nDocument chunk:\n{doc.page_content}"
    response = llm.invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=prompt),
    ])
    try:
        parsed = json.loads(response.content)
        return bool(parsed.get("relevant", False))
    except (json.JSONDecodeError, AttributeError):
        # if the model doesn't follow JSON format, be conservative and keep the doc
        return True


def grade_docs(state: RAGState) -> RAGState:
    """Filter retrieved docs; route to web search if not enough pass."""
    question = state["question"]
    docs: List[Document] = state.get("documents", [])
    print(f"[grade_docs] grading {len(docs)} docs")

    relevant = [doc for doc in docs if _grade_one(_llm, question, doc)]

    needs_web = len(relevant) < MIN_RELEVANT_DOCS

    return {
        **state,
        "documents": relevant,
        "needs_web_search": needs_web,
    }

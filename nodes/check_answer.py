"""
Answer check node — hallucination + usefulness gate.

Two checks in one pass:
  1. Grounded — is the answer supported by the provided documents?
  2. Useful   — does the answer actually address the question?

If either check fails and we haven't retried yet, the pipeline loops back
to generate with the web search results included for extra context.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from config.models import get_grading_llm
_llm = get_grading_llm()
from state import RAGState



_SYSTEM = """You are a quality checker for AI-generated answers.

Given a question, the context documents used, and the generated answer, evaluate:
1. Is the answer grounded in the context? (not hallucinated)
2. Does the answer actually address the question?

Respond ONLY with valid JSON:
{"grounded": true, "useful": true}

Use false for either if the answer fails that check."""


def check_answer(state: RAGState) -> RAGState:
    """Check whether the generated answer is grounded and useful."""
    question = state["question"]
    answer = state.get("answer", "")
    print(f"[check_answer] answer_len={len(answer)}")
    docs = state.get("documents", [])

    context_preview = "\n".join(d.page_content[:300] for d in docs[:3])

    prompt = (
        f"Question: {question}\n\n"
        f"Context (first 3 docs):\n{context_preview}\n\n"
        f"Answer: {answer}"
    )

    response = _llm.invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=prompt),
    ])

    try:
        parsed = json.loads(response.content)
        grounded = bool(parsed.get("grounded", True))
        useful = bool(parsed.get("useful", True))
        answer_ok = grounded and useful
    except (json.JSONDecodeError, AttributeError):
        answer_ok = True  # be lenient if the model can't follow JSON format

    return {
        **state,
        "answer_is_grounded": answer_ok,
        "retry_count": state.get("retry_count", 0) + 1,
    }

"""
Web search fallback node.

Triggered when the grader decides local docs aren't sufficient.
Uses DuckDuckGo — no API key, no cost.

The search results are turned into LangChain Documents and appended
to state["documents"] so the generate node can treat them uniformly.
"""

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.documents import Document

from config.settings import WEB_SEARCH_MAX_RESULTS
from state import RAGState


_search = DuckDuckGoSearchResults(
    num_results=WEB_SEARCH_MAX_RESULTS,
    output_format="list",
)


def web_search(state: RAGState) -> RAGState:
    """Search the web using the rewritten query and add results to documents."""
    query = state.get("rewritten_query") or state["question"]

    raw_results = _search.invoke(query)

    # raw_results is a list of dicts with keys: snippet, title, link
    web_docs = [
        Document(
            page_content=r.get("snippet", ""),
            metadata={"source": r.get("link", ""), "title": r.get("title", "")},
        )
        for r in raw_results
        if r.get("snippet")
    ]

    # merge web results with whatever local docs survived grading
    existing = state.get("documents", [])
    return {
        **state,
        "documents": existing + web_docs,
        "web_results": web_docs,
    }

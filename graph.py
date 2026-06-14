"""
LangGraph pipeline definition.

Wires all nodes into a StateGraph and defines the conditional routing logic.
The graph is built once and reused across queries.
"""

from langgraph.graph import StateGraph, END

from state import RAGState
from nodes.rewrite import rewrite_query
from nodes.grade import grade_docs
from nodes.web_search import web_search
from nodes.generate import generate
from nodes.check_answer import check_answer
from config.settings import MAX_RETRY_COUNT


# ── Routing functions (pure logic — no LLM calls) ─────────────────────────────

def route_after_grading(state: RAGState) -> str:
    """After grading, decide: generate from local docs OR go to web search."""
    if state.get("needs_web_search", False):
        return "web_search"
    return "generate"


def route_after_check(state: RAGState) -> str:
    """After the answer check, decide: done OR retry generation."""
    if state.get("answer_is_grounded", True):
        return END
    if state.get("retry_count", 0) >= MAX_RETRY_COUNT:
        # give up after max retries — return whatever we have
        return END
    return "generate"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(vectorstore, bm25_store) -> StateGraph:
    """
    Build and compile the LangGraph StateGraph.

    vectorstore and bm25_store are injected here so the retrieve node
    has access to them without using global state.
    """
    from nodes.retrieve import make_retrieve_node
    hybrid_retrieve = make_retrieve_node(vectorstore, bm25_store)

    graph = StateGraph(RAGState)

    # register nodes
    graph.add_node("rewrite_query",   rewrite_query)
    graph.add_node("hybrid_retrieve", hybrid_retrieve)
    graph.add_node("grade_docs",      grade_docs)
    graph.add_node("web_search",      web_search)
    graph.add_node("generate",        generate)
    graph.add_node("check_answer",    check_answer)

    # entry point
    graph.set_entry_point("rewrite_query")

    # fixed edges
    graph.add_edge("rewrite_query",   "hybrid_retrieve")
    graph.add_edge("hybrid_retrieve", "grade_docs")
    graph.add_edge("web_search",      "generate")
    graph.add_edge("generate",        "check_answer")

    # conditional edges
    graph.add_conditional_edges(
        "grade_docs",
        route_after_grading,
        {"web_search": "web_search", "generate": "generate"},
    )

    graph.add_conditional_edges(
        "check_answer",
        route_after_check,
        {"generate": "generate", END: END},
    )

    return graph.compile()

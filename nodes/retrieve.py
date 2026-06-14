"""
Hybrid retrieval node.

Runs two independent searches on the same query:
  1. Semantic search  — cosine similarity via Chroma
  2. Keyword search   — BM25Okapi

Then merges them using Reciprocal Rank Fusion (RRF).
This handles both "find me passages about neural networks" (semantic wins)
and "find the exact phrase 'activation function'" (BM25 wins).
"""

from __future__ import annotations
from langchain_community.vectorstores import Chroma

from config.settings import TOP_K_SEMANTIC, TOP_K_BM25
from state import RAGState
from utils.bm25_store import BM25Store
from utils.hybrid import reciprocal_rank_fusion


def make_retrieve_node(vectorstore: Chroma, bm25_store: BM25Store):
    """
    Factory — returns a node function pre-bound to the vectorstore + BM25 index.
    We use a factory so the graph can be built once with injected dependencies.
    """

    def hybrid_retrieve(state: RAGState) -> RAGState:
        query = state.get("rewritten_query") or state["question"]

        # 1. semantic search
        semantic_docs = vectorstore.similarity_search(query, k=TOP_K_SEMANTIC)

        # 2. BM25 keyword search
        bm25_docs = bm25_store.search(query, top_k=TOP_K_BM25)

        # 3. fuse both lists with RRF
        fused_docs = reciprocal_rank_fusion(semantic_docs, bm25_docs)

        return {**state, "documents": fused_docs}

    return hybrid_retrieve

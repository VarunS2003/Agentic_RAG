"""
Reciprocal Rank Fusion (RRF) for combining semantic + BM25 result lists.

RRF score for a document d = Σ  1 / (k + rank(d, list_i))
where k=60 is the standard constant that dampens the effect of very high ranks.

No score normalisation needed — RRF works purely on rank positions,
so semantic similarity scores and BM25 scores never need to be on the same scale.
"""

from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from config.settings import RRF_K, TOP_K_FINAL


def reciprocal_rank_fusion(
    semantic_docs: List[Document],
    bm25_docs: List[Document],
    k: int = RRF_K,
    top_k: int = TOP_K_FINAL,
) -> List[Document]:
    """
    Merge two ranked lists into one using RRF.
    Uses page_content as the deduplication key.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(semantic_docs, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_docs, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        doc_map[key] = doc

    # sort by fused score descending, return top_k documents
    ranked_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in ranked_keys[:top_k]]

"""
Thin wrapper around rank_bm25.BM25Okapi.

We keep the BM25 index in memory (rebuilt from Chroma docs on startup).
For small PDFs this is perfectly fine — BM25 on 200 chunks is instant.
"""

from __future__ import annotations
from typing import List
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class BM25Store:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        # tokenise by whitespace — simple and effective for most English text
        tokenised = [doc.page_content.lower().split() for doc in documents]
        self.index = BM25Okapi(tokenised)

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """Return top_k documents ranked by BM25 score."""
        tokens = query.lower().split()
        scores = self.index.get_scores(tokens)

        # pair each doc with its score, sort descending
        ranked = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [doc for doc, _ in ranked[:top_k]]

# Agentic RAG Pipeline — Local LLM Edition

A lightweight, agentic Retrieval-Augmented Generation pipeline built with **LangGraph** + **LangChain**, running entirely on local Ollama models with optional web search fallback.

---

## Stack

| Layer | Choice |
|---|---|
| Orchestration | LangGraph (StateGraph) |
| LLM | Ollama — `qwen3.5:4b` (generation), `gemma3:4b` (grading) |
| Embeddings | Ollama — `qwen3-embedding:4b` |
| Vector Store | ChromaDB (persistent) |
| Keyword Search | BM25 (rank_bm25) |
| Retrieval Strategy | Hybrid — semantic + BM25 with RRF fusion |
| Web Search | DuckDuckGo (ddg-search, free, no key needed) |
| Document Loaders | LangChain PDF / text loaders |

---

## Architecture — LangGraph Nodes

```
[ingest_docs]
      │
      ▼
 ┌────────────────────────────────────────┐
 │              Query Entry               │
 └────────────┬───────────────────────────┘
              │
              ▼
       [rewrite_query]          ← cleans + expands the user question
              │
              ▼
      [hybrid_retrieve]         ← BM25 + semantic search → RRF fusion
              │
              ▼
       [grade_docs]             ← LLM checks each doc for relevance
              │
         ┌───┴───┐
     relevant?   not enough?
         │            │
         ▼            ▼
    [generate]   [web_search]  ← DuckDuckGo fallback
         │            │
         └─────┬───────┘
               │
               ▼
        [check_answer]          ← hallucination + usefulness check
               │
         ┌─────┴──────┐
      grounded?    not grounded?
         │               │
         ▼               ▼
    [__end__]       [generate]  ← one retry with web context
```

---

## Project Layout

```
agentic_rag/
├── main.py                  # entry point — ingest + run pipeline
├── graph.py                 # LangGraph StateGraph definition
├── state.py                 # shared TypedDict state
├── config/
│   └── settings.py          # all tuneable knobs in one place
├── nodes/
│   ├── ingest.py            # load PDFs → chunk → embed → store
│   ├── rewrite.py           # query rewriting node
│   ├── retrieve.py          # hybrid BM25 + semantic retrieval
│   ├── grade.py             # relevance grader node
│   ├── generate.py          # answer generation node
│   ├── web_search.py        # DuckDuckGo fallback node
│   └── check_answer.py      # hallucination / quality check node
└── utils/
    ├── hybrid.py            # RRF fusion logic
    └── bm25_store.py        # BM25 index wrapper
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Make sure Ollama is running with required models
```bash
ollama pull qwen3-embedding:4b
ollama pull qwen3.5:4b
ollama pull gemma3:4b
```

### 3. Ingest a PDF and ask a question
```bash
python main.py --pdf path/to/your.pdf --query "What is the main finding?"
```

### 4. Ask follow-up questions (reuses existing vector store)
```bash
python main.py --query "Summarise the methodology"
```

---

## Key Design Decisions

- **Hybrid retrieval** uses BM25 for exact keyword hits and cosine similarity for semantic matches. Reciprocal Rank Fusion (RRF) merges both lists without needing score normalisation.
- **Grading** happens before generation so we only pass relevant chunks. If fewer than 2 docs pass grading, the pipeline routes to web search.
- **Web search** uses DuckDuckGo — no API key, no cost, works offline-ish for cached results.
- **One retry** is built in: if the answer check detects hallucination or the answer is unhelpful, the pipeline re-generates once with the web context included.
- All LLM calls use structured output (JSON mode via `format="json"`) to keep grading decisions machine-readable.

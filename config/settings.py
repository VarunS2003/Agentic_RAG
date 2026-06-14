"""
All tuneable settings live here.
Change these to adjust behaviour without touching pipeline code.
"""

# ── Ollama models ──────────────────────────────────────────────────────────────
GENERATION_MODEL   = "qwen3.5:4b"      # main answer generation
GRADING_MODEL      = "qwen3.5:4b"       # relevance + hallucination checks
EMBEDDING_MODEL    = "qwen3-embedding:4b"

OLLAMA_BASE_URL    = "http://localhost:11434"

# Which backend to use for models: "ollama" or "gemini"
# Set via env or edit here. If set to "gemini", ensure GEMINI_API_KEY is available.
MODEL_BACKEND = "gemini"

# Optional: Gemini model name to use when MODEL_BACKEND == "gemini"
GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCUjVI4J7IFqM6T6q7fzYqUOIIj8DT97p4")

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE         = 512               # characters per chunk
CHUNK_OVERLAP      = 64                # overlap between adjacent chunks

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K_SEMANTIC     = 5                 # how many docs semantic search returns
TOP_K_BM25         = 5                 # how many docs BM25 returns
TOP_K_FINAL        = 5                 # how many docs after RRF fusion
RRF_K              = 60                # RRF constant (60 is the standard default)

# ── Grading thresholds ─────────────────────────────────────────────────────────
MIN_RELEVANT_DOCS  = 2                 # if fewer docs pass grading → web search

# ── Web search ─────────────────────────────────────────────────────────────────
WEB_SEARCH_MAX_RESULTS = 3

# ── Chroma ─────────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION  = "rag_docs"

# ── Generation ────────────────────────────────────────────────────────────────
GENERATION_TEMPERATURE = 0.2           # low temp = more factual answers
MAX_RETRY_COUNT        = 1             # max re-generation attempts

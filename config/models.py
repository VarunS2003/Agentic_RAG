"""
Shared model instances for the pipeline.

Centralises ChatOllama/OllamaEmbeddings declarations so other modules
import ready-to-use objects instead of instantiating models locally.
"""
from langchain_ollama import ChatOllama, OllamaEmbeddings

from config.settings import (
    GENERATION_MODEL,
    GRADING_MODEL,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    GENERATION_TEMPERATURE,
    MODEL_BACKEND,
    GEMINI_MODEL,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_API_KEY,
)

import logging
logger = logging.getLogger("rag")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# LLM used for generation and light-weight tasks like rewriting
generation_llm = ChatOllama(
    model=GENERATION_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=GENERATION_TEMPERATURE,
)

# Reuse the same generation model for rewrites (same config)
rewrite_llm = generation_llm

# LLM used for grading / JSON outputs
grading_llm = ChatOllama(
    model=GRADING_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    format="json",
)

# Embeddings function
ollama_embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL,
)


# Optional: Google Gemini (GenAI) client configuration.
# If the `google.genai` (or `genai`) package is installed and an
# API key is available via the `GEMINI_API_KEY` env var, a client
# object will be exposed as `gemini_client` for ad-hoc calls.
import os

gemini_client = None
GEMINI_MODEL = GEMINI_MODEL

try:
    # Try google.generativeai (correct package name)
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_client = genai
    logger.info("[Gemini] Client initialized (google.generativeai) with API key")
except Exception as e:
    logger.warning("[Gemini] Failed to initialize google.generativeai: %s", e)
    gemini_client = None


# Example usage (commented):
# from config.models import gemini_client, GEMINI_MODEL
# if gemini_client is not None:
#     my_file = gemini_client.files.upload(file="image.png")
#     response = gemini_client.models.generate_content(
#         model=GEMINI_MODEL,
#         contents=[my_file, "Describe all elements present in this P&ID diagram"],
#     )


# --- Backend selection helpers -----------------------------------------------
class _SimpleResponse:
    def __init__(self, content: str):
        self.content = content


class _GeminiLLMWrapper:
    """Minimal wrapper exposing an `invoke(messages)` method like ChatOllama.

    It concatenates system + human messages into a single prompt and calls
    `client.models.generate_content`. The wrapper returns an object with a
    `content` attribute containing the model text output.
    """

    def __init__(self, client, model_name: str):
        self.client = client
        self.model = model_name

    def invoke(self, messages):
        text = "\n\n".join(getattr(m, "content", "") for m in messages)
        logger.info("[Gemini LLM] invoking %s (prompt_len=%d)", self.model, len(text))
        try:
            response = gemini_client.GenerativeModel(self.model).generate_content(text)
            out = response.text
            logger.info("[Gemini LLM] response received (len=%d)", len(out))
            return _SimpleResponse(out)
        except Exception as e:
            logger.error("[Gemini LLM] error: %s", e)
            return _SimpleResponse("")


def get_generation_llm():
    """Return an LLM-like object with `invoke(messages)` for generation tasks.

    Chooses between Ollama and Gemini depending on `MODEL_BACKEND`.
    """
    if MODEL_BACKEND == "gemini" and gemini_client is not None:
        logger.info("[get_generation_llm] Using Gemini backend")
        return _GeminiLLMWrapper(gemini_client, GEMINI_MODEL)
    logger.info("[get_generation_llm] Using Ollama backend")
    return generation_llm


def get_rewrite_llm():
    if MODEL_BACKEND == "gemini" and gemini_client is not None:
        logger.info("[get_rewrite_llm] Using Gemini backend")
        return _GeminiLLMWrapper(gemini_client, GEMINI_MODEL)
    logger.info("[get_rewrite_llm] Using Ollama backend")
    return rewrite_llm


def get_grading_llm():
    if MODEL_BACKEND == "gemini" and gemini_client is not None:
        logger.info("[get_grading_llm] Using Gemini backend")
        return _GeminiLLMWrapper(gemini_client, GEMINI_MODEL)
    logger.info("[get_grading_llm] Using Ollama backend")
    return grading_llm


def get_embedding_function():
    """Return an embedding function suitable for Chroma.

    Always returns Ollama embeddings to maintain compatibility with existing Chroma collection
    (which was created with 2560-dim Ollama embeddings, not 3072-dim Gemini embeddings).
    """
    logger.info("[get_embedding_function] Using Ollama backend")
    return ollama_embeddings


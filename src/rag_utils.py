"""
rag_utils.py — Retrieval-Augmented Generation (RAG) pipeline.

Loads the persisted ChromaDB vector store (built from the Bitext customer
service dataset), embeds incoming user queries using a multilingual
sentence-transformer, and retrieves the most contextually relevant
Q&A documents via Maximal Marginal Relevance (MMR) search.
"""

import asyncio
import logging
import os
import torch
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_DIR       = Path(os.getenv("CHROMA_DIR", str(Path(__file__).parent.parent / "data" / "chroma_db")))
EMBEDDING_MODEL  = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

TOP_K       = int(os.getenv("RAG_TOP_K",       "4"))
FETCH_K     = int(os.getenv("RAG_FETCH_K",     "20"))
MMR_LAMBDA  = float(os.getenv("RAG_MMR_LAMBDA",  "0.6"))

# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Retrieval pipeline that fetches relevant Bitext SOP context for a query.

    Uses MMR (Maximal Marginal Relevance) retrieval to balance result
    relevance and diversity, reducing redundant context fed to the LLM.
    Cross-lingual embeddings allow Indonesian queries to match English
    or mixed-language documents in the store.
    """

    def __init__(self) -> None:
        logger.info("Initialising RAG pipeline …")

        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"ChromaDB not found at '{CHROMA_DIR}'. "
                "Run `python src/build_rag.py` first to build the vector store."
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", device.upper())

        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        self._vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=self._embeddings,
        )

        self._retriever = self._vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k":           TOP_K,
                "fetch_k":     FETCH_K,
                "lambda_mult": MMR_LAMBDA,
            },
        )

        doc_count = self._vectorstore._collection.count()
        logger.info(
            "RAG pipeline ready on %s — %d documents in store.",
            device.upper(),
            doc_count,
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def retrieve(self, query: str) -> dict:
        """
        Retrieve the most relevant SOP context for a user query.

        Args:
            query: Raw user message (may be in any language).

        Returns:
            A dict with:
              - ``context``  (str)  – Concatenated document snippets.
              - ``sources``  (list) – Unique source labels for UI display.
              - ``intents``  (list) – Detected intent categories.

        Raises:
            RuntimeError: if the vector store is empty or retrieval fails.
        """
        loop = asyncio.get_event_loop()
        try:
            docs = await loop.run_in_executor(
                None, lambda: self._retriever.invoke(query)
            )
        except Exception as exc:
            logger.exception("RAG retrieval failed for query: %s", query[:80])
            raise RuntimeError("RAG retrieval failed.") from exc

        if not docs:
            logger.warning("RAG returned no documents for query: %s", query[:80])
            return {
                "context": (
                    "No specific customer service SOP found. "
                    "Use general empathetic customer service best practices."
                ),
                "sources": [],
                "intents": [],
            }

        context     = "\n\n---\n\n".join(doc.page_content for doc in docs)
        sources     = list({doc.metadata.get("source", "Bitext CS SOP") for doc in docs})
        intents     = list({doc.metadata.get("intent", "general")        for doc in docs})

        logger.debug(
            "Retrieved %d docs (intents: %s) for query: %s",
            len(docs), intents, query[:60],
        )
        return {"context": context, "sources": sources, "intents": intents}

    async def similarity_search(self, query: str, k: int = 5) -> list[dict]:
        """
        Return the top-k most similar documents with their similarity scores.

        Useful for debugging retrieval quality during development or
        evaluating RAG performance via RAGAS.
        """
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None,
                lambda: self._vectorstore.similarity_search_with_score(query, k=k),
            )
        except Exception as exc:
            logger.exception("Similarity search failed for query: %s", query[:80])
            raise RuntimeError("Similarity search failed.") from exc

        return [
            {
                "content": doc.page_content,
                "source":  doc.metadata.get("source", "Unknown"),
                "intent":  doc.metadata.get("intent", "unknown"),
                "score":   round(float(score), 4),
            }
            for doc, score in results
        ]

    def get_stats(self) -> dict:
        """Return basic statistics about the vector store."""
        count = self._vectorstore._collection.count()
        return {
            "total_documents":  count,
            "embedding_model":  EMBEDDING_MODEL,
            "top_k":            TOP_K,
            "fetch_k":          FETCH_K,
            "mmr_lambda":       MMR_LAMBDA,
        }

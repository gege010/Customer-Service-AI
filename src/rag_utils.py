"""
rag_utils.py — Retrieval-Augmented Generation (RAG) pipeline.

Loads the persisted ChromaDB vector store (built from the Bitext customer
service dataset), embeds incoming user queries using a multilingual
sentence-transformer, and retrieves the most contextually relevant
Q&A documents via Maximal Marginal Relevance (MMR) search.
"""

import asyncio
import logging
import torch
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_DIR      = Path(__file__).parent.parent / "data" / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K      = 4     # Final documents returned to the LLM
FETCH_K    = 20    # Candidate pool size for MMR diversity filtering
MMR_LAMBDA = 0.6   # Relevance (1.0) ↔ Diversity (0.0) balance


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

    def __init__(self):
        logger.info("Initialising RAG pipeline …")

        # Prefer GPU for faster embedding generation
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"ChromaDB not found at '{CHROMA_DIR}'.  "
                "Run `python src/build_rag.py` first to build the vector store."
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
        """
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(
            None, lambda: self._retriever.invoke(query)
        )

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

        context_parts = [doc.page_content for doc in docs]
        sources  = list({doc.metadata.get("source", "Bitext CS SOP") for doc in docs})
        intents  = list({doc.metadata.get("intent", "general")        for doc in docs})
        context  = "\n\n---\n\n".join(context_parts)

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
        results = await loop.run_in_executor(
            None,
            lambda: self._vectorstore.similarity_search_with_score(query, k=k),
        )
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
            "total_documents": count,
            "embedding_model": EMBEDDING_MODEL,
            "top_k": TOP_K,
            "fetch_k": FETCH_K,
        }
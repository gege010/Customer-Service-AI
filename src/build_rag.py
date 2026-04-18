"""
build_rag.py — One-time script to build the ChromaDB vector store.

Reads the Bitext customer service dataset, formats each Q&A pair as a
rich context document, embeds them with a multilingual sentence-transformer,
and persists the results into a local ChromaDB collection.

Usage (run once from the project root):
    python src/build_rag.py
"""

import os
import sys
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Paths & Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT    = Path(__file__).parent.parent
DATA_RAW        = PROJECT_ROOT / "data" / "raw" / os.getenv(
    "BITEXT_CSV", "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)
CHROMA_DIR      = PROJECT_ROOT / "data" / "chroma_db"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
BATCH_SIZE      = int(os.getenv("RAG_INGEST_BATCH_SIZE", "500"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_bitext(csv_path: Path) -> pd.DataFrame:
    """Load and clean the Bitext CSV dataset."""
    logger.info("Loading Bitext dataset from: %s", csv_path)

    if not csv_path.exists():
        logger.error("Bitext CSV not found at: %s", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    required_cols = {"instruction", "intent", "response"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error("CSV is missing required columns: %s", missing)
        sys.exit(1)

    df = df[list(required_cols)].dropna()
    df = df.drop_duplicates(subset=["instruction"])

    logger.info("Loaded %d unique Q&A pairs after cleaning.", len(df))
    return df


def build_documents(df: pd.DataFrame) -> list[Document]:
    """
    Convert each Bitext row into a LangChain Document.

    The page_content combines the question and answer so the embedding
    captures the full semantic context.  Metadata carries intent and
    source labels used later in the UI sources panel.
    """
    docs = []
    for _, row in df.iterrows():
        intent   = str(row["intent"]).strip()
        question = str(row["instruction"]).strip()
        answer   = str(row["response"]).strip()

        for placeholder, replacement in {
            "{{Order Number}}": "[order number]",
            "{{Name}}":         "[customer name]",
            "{{URL}}":          "[website URL]",
            "{{Phone}}":        "[phone number]",
            "{{EMAIL}}":        "[email address]",
            "{{Account}}":      "[account number]",
        }.items():
            question = question.replace(placeholder, replacement)
            answer   = answer.replace(placeholder, replacement)

        content = (
            f"[Intent: {intent.replace('_', ' ').title()}]\n"
            f"Customer: {question}\n"
            f"Agent: {answer}"
        )

        docs.append(Document(
            page_content=content,
            metadata={
                "intent": intent,
                "source": f"Bitext CS SOP — {intent.replace('_', ' ').title()}",
            },
        ))

    logger.info("Built %d documents ready for embedding.", len(docs))
    return docs


def build_vectorstore(docs: list[Document]) -> Chroma:
    """Embed the documents and persist them to ChromaDB."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s for embeddings.", device.upper())

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    if CHROMA_DIR.exists():
        import shutil
        logger.warning("Existing ChromaDB found — removing to rebuild from scratch.")
        shutil.rmtree(CHROMA_DIR)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Ingesting %d documents into ChromaDB in batches of %d …",
        len(docs), BATCH_SIZE,
    )

    vectorstore: Chroma | None = None
    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Ingesting batches"):
        batch = docs[i : i + BATCH_SIZE]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=str(CHROMA_DIR),
            )
        else:
            vectorstore.add_documents(batch)

    doc_count = vectorstore._collection.count() if vectorstore else 0
    logger.info("ChromaDB successfully built at: %s", CHROMA_DIR)
    logger.info("Total documents in store: %d", doc_count)
    return vectorstore


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df   = load_bitext(DATA_RAW)
    docs = build_documents(df)
    build_vectorstore(docs)
    logger.info("Done! You can now start the API server.")


if __name__ == "__main__":
    main()

# Customer Service AI — Groq + Bitext RAG

End-to-end AI customer service system that combines **Groq Cloud LLM** (Llama 3.3 70B) with a **Cross-Lingual Retrieval-Augmented Generation** (RAG) pipeline. Users interact in Bahasa Indonesia; the system retrieves semantically relevant policies from the Bitext dataset and streams a polished response in real time.

Built as a portfolio project demonstrating full-stack AI/ML engineering from data preprocessing and LLM fine-tuning through to async API serving and production-quality UI.

---

## Architecture

```text
User (Bahasa Indonesia)
        │
        ▼
┌─────────────────────┐
│  Streamlit UI       │  Dark-theme chat with true token streaming
│  (src/app.py)       │
└────────┬────────────┘
         │  HTTP POST /chat/stream
         ▼
┌─────────────────────┐
│  FastAPI Backend    │  Async inference server    (src/api.py)
│  (src/api.py)       │
└──┬──────────────────┘
   │
   ├──▶ RAG Pipeline (ChromaDB)           (src/rag_utils.py)
   │       ├── Query embedding via MiniLM-L12 (multilingual)
   │       ├── MMR retrieval (TOP_K=4, FETCH_K=20)
   │       └── Documents sourced from Bitext 27K CS dataset
   │
   └──▶ Groq Cloud LLM
           ├── Model:  Llama 3.3 70B Versatile
           ├── Input:  RAG context + conversation history (last 6 turns)
           └── Output: Streamed Bahasa Indonesia response
```

**Key flows:**
| Endpoint | Description |
|---|---|
| `GET  /health` | Liveness probe — checks RAG & LLM readiness |
| `GET  /rag/stats` | Vector store statistics (document count, model info) |
| `POST /rag/search` | Debug: raw similarity search with scores |
| `POST /chat` | Blocking full-response endpoint |
| `POST /chat/stream` | **Primary**: streaming endpoint for the UI |

---

## ML Engineering Highlights

### Cross-Lingual RAG
Users query in Indonesian; retrieval runs on the English/bilingual Bitext corpus using `paraphrase-multilingual-MiniLM-L12-v2`.  Because the embedding space is shared across 50+ languages, semantic similarity is preserved without translation, eliminating latency and error propagation from external translation APIs.

### Bitext Knowledge Base
The [Bitext Customer Service Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (~27K Q&A pairs across 27 intents) is pre-processed and ingested into ChromaDB as rich context documents:

```
[Intent: Cancel Order]
Customer: I need to cancel order #12345
Agent: Of course! I can help you cancel that order…
```

Intent metadata is stored alongside each embedding, enabling intent-aware retrieval.

### Maximal Marginal Relevance (MMR)
Standard top-k retrieval tends to return near-duplicate documents.  MMR balances relevance and diversity (λ = 0.6), ensuring the LLM receives varied, complementary context snippets rather than repetitive policy fragments.

### Groq Cloud Inference
Groq's LPU hardware delivers sub-second first-token latency for Llama 3.3 70B, enabling genuine real-time token streaming.  This replaces the previous TinyLlama 1.1B + LoRA setup, which required local GPU and produced lower-quality multilingual responses.

### LoRA Fine-Tuning — ML Research Track
`notebooks/02_llm_finetuning.ipynb` is an **independent ML research demonstration**, separate from the live production system.  It shows end-to-end PEFT/LoRA fine-tuning (QLoRA, 4-bit) of TinyLlama 1.1B on Bitext, including loss curve visualisation and adapter export.  The trained weights are stored locally in `models/lora_cs_model/` (gitignored — not shipped).  This track exists to showcase ML engineering depth; the live API uses Groq for superior response quality without requiring a local GPU.

### Prompt Engineering
The system prompt explicitly handles:
- **Greetings / thanks** — warm, natural responses (no forced SOP lookup).
- **Placeholder replacement** — `{{URL}}`, `{{Phone}}`, etc. mapped to natural language.
- **Honesty constraints** — model is instructed not to hallucinate policies absent from the retrieved context.
- **Empathy-first structure** — acknowledge feelings before delivering instructions.

---

## Project Structure

```text
Customer-Service-AI/
├── .env                            # GROQ_API_KEY (not committed)
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── chroma_db/                  # ChromaDB vector store (auto-generated)
│   ├── processed/                  # Cleaned dataset for fine-tuning
│   └── raw/
│       └── Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv
│
├── models/                         # ⚠️ All contents are gitignored
│   ├── checkpoints/                # QLoRA training checkpoints
│   └── lora_cs_model/              # Fine-tuned LoRA adapter (local only)
│
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb   # ✅ Used — EDA & ChromaDB data prep
│   └── 02_llm_finetuning.ipynb          # 📚 ML Research — QLoRA/LoRA fine-tuning demo
│
└── src/                            # ✅ Live production system
    ├── build_rag.py    # One-time: ingest Bitext CSV → ChromaDB vector store
    ├── api.py          # FastAPI backend: RAG retrieval + Groq LLM streaming
    ├── app.py          # Streamlit dark-theme frontend UI
    └── rag_utils.py    # ChromaDB + MMR retrieval pipeline
```

---

## Setup

**Prerequisites:** Python 3.11+, NVIDIA GPU with CUDA 12.1+ *(optional — CPU works, slower embeddings)*

### 1 — Create environment

```bash
conda create -n cs_ai python=3.11 -y
conda activate cs_ai
pip install -r requirements.txt
```

### 2 — Configure API key

```bash
# Create .env in the project root
echo "GROQ_API_KEY=your_groq_key_here" > .env
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3 — Build the RAG vector store

Run this **once** to ingest the Bitext dataset into ChromaDB:

```bash
# From project root
python src/build_rag.py
```

Expected output:
```
Loaded 24635 unique Q&A pairs after cleaning.
Built 24635 documents ready for embedding.
Using device: CUDA for embeddings.
Ingesting 24635 documents into ChromaDB in batches of 500 …
ChromaDB successfully built at: data/chroma_db
Total documents in store: 24635
```

---

## Running the Application

The system uses a client–server architecture; both servers must be running simultaneously.

**Terminal 1 — Backend (FastAPI):**
```bash
conda activate cs_ai
cd src
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Wait for `All backend components ready. Waiting for requests …` before starting the UI.

**Terminal 2 — Frontend (Streamlit):**
```bash
conda activate cs_ai
cd src
streamlit run app.py
```

The UI will open automatically at `http://localhost:8501`.

---

## API Reference

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe |
| GET | `/rag/stats` | Vector store statistics |
| POST | `/rag/search` | `{"query": "...", "k": 5}` — similarity search |
| POST | `/chat` | `ChatRequest` → `ChatResponse` (blocking) |
| POST | `/chat/stream` | `ChatRequest` → `text/plain` stream |

**ChatRequest schema:**
```json
{
  "message": "Bagaimana cara membatalkan pesanan?",
  "session_id": "uuid-here",
  "history": [
    {"role": "user", "content": "Halo"},
    {"role": "assistant", "content": "Halo! Ada yang bisa saya bantu?"}
  ]
}
```

---

## Engineering Notes

### Why Groq over local TinyLlama?
TinyLlama 1.1B, even with LoRA fine-tuning, struggles to generate coherent multilingual responses.  Groq's Llama 3.3 70B delivers dramatically better instruction-following and Indonesian language quality at near-zero latency thanks to dedicated LPU hardware.  The LoRA fine-tuning experiment (Notebook 02) remains as proof of ML engineering capability and can be re-integrated for fully offline deployment.

### Async design
ChromaDB embedding calls and LLM inference are both blocking operations.  Wrapping them in `asyncio.get_event_loop().run_in_executor()` prevents blocking the FastAPI event loop, keeping the server responsive under concurrent load.

### RAG cold-start & caching
On first startup the sentence-transformer model is downloaded from HuggingFace Hub (~120 MB).  Subsequent starts load from the local cache, bringing startup time to under 10 seconds on GPU and ~30 seconds on CPU.

### Over-grounding prevention
Without explicit handling, the LLM retrieves SOPs even for simple greetings, producing awkward policy-heavy responses.  The system prompt conditionally bypasses the SOP for conversational turns (detected by semantic intent), ensuring natural dialogue flow alongside accurate policy answers.

---

## Evaluation & Future Work

| Area | Status | Notes |
|---|---|---|
| RAG quality (RAGAS) | 🔜 Planned | Faithfulness + answer relevancy |
| Intent classification layer | 🔜 Planned | Pre-route queries before LLM |
| LLM fine-tuning (LoRA) | ✅ Completed | `notebooks/02_llm_finetuning.ipynb` |
| Streaming UI | ✅ Completed | Token-by-token via Groq |
| Docker / Compose | 🔜 Planned | One-command deployment |
| Feedback loop | 🔜 Planned | Collect low-confidence turns for future fine-tuning |
| Persistent sessions | 🔜 Planned | Redis or SQLite conversation store |

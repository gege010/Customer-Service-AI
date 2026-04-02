"""
api.py — FastAPI backend for the Customer Service AI.

Architecture:
  1. User query arrives at the /chat/stream endpoint.
  2. The RAG pipeline queries ChromaDB (built from Bitext dataset) to
     retrieve the most relevant customer-service Q&A context.
  3. The retrieved context + conversation history are concatenated into a
     carefully-engineered prompt and sent to Groq's Llama 3.3 70B model.
  4. The LLM response is streamed token-by-token back to the Streamlit UI.

Additional endpoints:
  GET  /health          — Liveness check for monitoring tools.
  GET  /rag/stats       — Vector store statistics (doc count, model info).
  POST /rag/search      — Debug endpoint: raw similarity search results.
  POST /chat            — Blocking (non-streaming) fallback endpoint.
  POST /chat/stream     — Primary streaming endpoint consumed by the UI.
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from rag_utils import RAGPipeline

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Shared state: populated on startup, cleared on shutdown
ml_state: dict = {}

# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialise the local RAG pipeline and connect to Groq Cloud.
    Shutdown: release shared state.
    """
    logger.info("Starting Customer Service AI backend …")

    # 1. Local RAG (ChromaDB + multilingual embeddings)
    ml_state["rag"] = RAGPipeline()

    # 2. Cloud LLM via Groq
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        logger.error("GROQ_API_KEY not found in environment!  Set it in .env.")

    ml_state["llm"] = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.4,
        streaming=True,
    )

    logger.info("All backend components ready.  Waiting for requests …")
    yield

    ml_state.clear()
    logger.info("Backend shut down cleanly.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Service AI — NovaMart Enterprise API",
    description=(
        "End-to-end AI customer service system: "
        "Cross-lingual RAG (ChromaDB) + Groq Cloud LLM (Llama 3.3 70B)."
    ),
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default")
    history: list[Message] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: list[str]
    intents: list[str]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=20)


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------

def build_prompt() -> ChatPromptTemplate:
    """
    Build the system prompt that governs Aria's personality and behaviour.

    Key design decisions:
    - Always respond in fluent Bahasa Indonesia.
    - Distinguish conversational turns (greetings/thanks) from actionable
      issues so the model doesn't force SOPs onto simple messages.
    - Instruct explicit placeholder replacement to prevent template artefacts
      (e.g. {{URL}}, {{Phone}}) from leaking into the final response.
    - Limit the response to what is actually in the retrieved SOP context;
      do not hallucinate policies that aren't supported by the documents.
    """
    system_instruction = """\
Anda adalah Aria, Customer Service Agent yang empatik dan profesional untuk NovaMart.

INSTRUKSI KETAT:
1. BAHASA: Selalu balas dalam Bahasa Indonesia yang fasih dan sopan, terlepas dari bahasa yang digunakan pengguna atau konteks SOP.
2. SAPAAN & TERIMA KASIH: Jika pengguna hanya menyapa (mis. "Halo", "Selamat pagi") atau berterima kasih, balas dengan hangat dan bersahabat — jangan gunakan SOP.
3. MASALAH SPESIFIK: Jika ada pertanyaan atau masalah, gunakan 'Konteks SOP' di bawah untuk memberikan panduan yang jelas dan bertahap.
4. PLACEHOLDER: Ganti semua placeholder seperti {{URL}}, {{Phone}}, {{EMAIL}}, [website URL], [phone number], atau [email address] dengan "website kami", "nomor telepon kami", atau "email support kami".
5. KEJUJURAN: Jika SOP tidak mencakup pertanyaan pengguna, sampaikan dengan sopan bahwa Anda akan meneruskan ke tim yang relevan — jangan mengarang kebijakan.
6. EMPATI: Mulai respons dengan mengakui perasaan pelanggan sebelum memberikan solusi.

Konteks SOP (dari basis pengetahuan Bitext):
{context}"""

    return ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}"),
    ])


# ---------------------------------------------------------------------------
# Core streaming logic
# ---------------------------------------------------------------------------

async def stream_response(
    query: str,
    history: list[Message],
) -> AsyncGenerator[str, None]:
    """
    Full pipeline: RAG retrieval → prompt assembly → Groq streaming.

    Keeps the last 6 messages of history to maintain conversational
    coherence without wasting tokens.
    """
    llm = ml_state["llm"]
    rag = ml_state["rag"]

    # 1. Retrieve relevant SOP context from ChromaDB
    rag_result = await rag.retrieve(query)
    context    = rag_result["context"]

    # 2. Convert UI message history to LangChain message objects
    formatted_history: list = []
    for msg in history[-6:]:
        if msg.role == "user":
            formatted_history.append(HumanMessage(content=msg.content))
        else:
            formatted_history.append(AIMessage(content=msg.content))

    # 3. Assemble and invoke the chain
    chain = build_prompt() | llm

    async for chunk in chain.astream({
        "context":      context,
        "chat_history": formatted_history,
        "user_input":   query,
    }):
        if chunk.content:
            yield chunk.content


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Liveness probe — returns OK when the backend is fully initialised."""
    if "llm" not in ml_state or "rag" not in ml_state:
        raise HTTPException(status_code=503, detail="Backend not yet initialised.")
    return {
        "status": "ok",
        "rag":    "ready",
        "llm":    "ready (Groq Llama-3.3-70B)",
    }


@app.get("/rag/stats", tags=["RAG"])
async def rag_stats():
    """Return statistics about the loaded ChromaDB vector store."""
    if "rag" not in ml_state:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready.")
    return ml_state["rag"].get_stats()


@app.post("/rag/search", tags=["RAG"])
async def rag_search(request: SearchRequest):
    """
    Debug endpoint: run a raw similarity search against the vector store.
    Returns the top-k most similar documents with their scores.
    """
    if "rag" not in ml_state:
        raise HTTPException(status_code=503, detail="RAG pipeline not ready.")
    results = await ml_state["rag"].similarity_search(request.query, k=request.k)
    return {"query": request.query, "results": results}


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Blocking chat endpoint — returns a complete JSON response.
    Useful for testing and non-streaming clients.
    """
    if "llm" not in ml_state:
        raise HTTPException(status_code=503, detail="Backend not ready.")

    rag        = ml_state["rag"]
    rag_result = await rag.retrieve(request.message)
    context    = rag_result["context"]

    formatted_history: list = []
    for msg in request.history[-6:]:
        if msg.role == "user":
            formatted_history.append(HumanMessage(content=msg.content))
        else:
            formatted_history.append(AIMessage(content=msg.content))

    chain          = build_prompt() | ml_state["llm"]
    final_response = await chain.ainvoke({
        "context":      context,
        "chat_history": formatted_history,
        "user_input":   request.message,
    })

    return ChatResponse(
        response=final_response.content,
        session_id=request.session_id,
        sources=rag_result["sources"],
        intents=rag_result.get("intents", []),
    )


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """
    Primary streaming endpoint consumed by the Streamlit UI.
    Streams LLM tokens as plain text for real-time rendering.
    """
    if "llm" not in ml_state:
        raise HTTPException(status_code=503, detail="Backend not ready.")

    return StreamingResponse(
        stream_response(request.message, request.history),
        media_type="text/plain",
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
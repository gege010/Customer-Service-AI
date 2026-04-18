"""
app.py — Streamlit frontend for the Customer Service AI.

Connects to the FastAPI backend (api.py) via HTTP and renders a premium
dark-theme chat interface with real-time token streaming.
"""

import os
import uuid
import requests
from datetime import datetime

import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL   = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_STREAM_URL = f"{API_BASE_URL}/chat/stream"
API_HEALTH_URL = f"{API_BASE_URL}/health"

COMPANY_NAME = "NovaMart"
AGENT_NAME   = "Aria"
AGENT_AVATAR = "🎧"
VERSION      = "4.0.0"

QUICK_REPLIES = [
    "Halo! Bisa bantu saya?",
    "Bagaimana cara melacak pesanan saya?",
    "Saya ingin membatalkan pesanan.",
    "Pesanan saya rusak saat diterima.",
    "Bagaimana cara mengajukan pengembalian dana?",
]


# ---------------------------------------------------------------------------
# Page setup & CSS injection
# ---------------------------------------------------------------------------

def page_config():
    st.set_page_config(
        page_title=f"Aria CS — {COMPANY_NAME}",
        page_icon=AGENT_AVATAR,
        layout="centered",
        initial_sidebar_state="expanded",
    )


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [data-testid="stApp"] {
            font-family: 'Inter', sans-serif;
            background: #0f1117;
            color: #e2e8f0;
        }

        #MainMenu, footer, header { visibility: hidden; }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1f2e 0%, #0f1117 100%);
            border-right: 1px solid #2d3748;
        }
        [data-testid="stSidebar"] .stMarkdown p { color: #a0aec0; font-size: 0.85rem; }

        [data-testid="stChatMessage"] {
            border-radius: 16px;
            padding: 12px 16px;
            margin-bottom: 8px;
            border: 1px solid rgba(255,255,255,0.05);
        }

        [data-testid="stChatMessage"][data-testid*="user"] {
            background: linear-gradient(135deg, #1e3a5f 0%, #1a2744 100%);
        }

        [data-testid="stChatMessage"]:not([data-testid*="user"]) {
            background: linear-gradient(135deg, #1a2035 0%, #0f1117 100%);
        }

        [data-testid="stChatInput"] textarea {
            background: #1a1f2e !important;
            border: 1px solid #4a5568 !important;
            border-radius: 12px !important;
            color: #e2e8f0 !important;
            font-family: 'Inter', sans-serif !important;
        }
        [data-testid="stChatInput"] textarea:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.15) !important;
        }

        .stButton button {
            background: linear-gradient(135deg, #1e3a5f, #1a2744) !important;
            color: #90cdf4 !important;
            border: 1px solid #2b6cb0 !important;
            border-radius: 20px !important;
            font-size: 0.8rem !important;
            padding: 6px 14px !important;
            transition: all 0.2s ease !important;
            white-space: normal !important;
            height: auto !important;
        }
        .stButton button:hover {
            background: linear-gradient(135deg, #2b4c7e, #2a3f6f) !important;
            border-color: #63b3ed !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(66,153,225,0.3) !important;
        }

        .msg-caption {
            font-size: 0.70rem;
            color: #718096;
            margin-top: 4px;
            text-align: right;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(72,187,120,0.12);
            border: 1px solid rgba(72,187,120,0.3);
            border-radius: 20px;
            padding: 3px 10px;
            font-size: 0.75rem;
            color: #68d391;
        }
        .status-dot {
            width: 7px;
            height: 7px;
            background: #68d391;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50%       { opacity: 0.4; }
        }
        .status-badge.offline {
            background: rgba(237,137,54,0.12);
            border-color: rgba(237,137,54,0.3);
            color: #ed8936;
        }
        .status-badge.offline .status-dot {
            background: #ed8936;
            animation: none;
        }

        .source-chip {
            display: inline-block;
            background: rgba(102,126,234,0.15);
            border: 1px solid rgba(102,126,234,0.3);
            border-radius: 12px;
            padding: 2px 10px;
            font-size: 0.72rem;
            color: #a3bffa;
            margin: 2px 3px;
        }

        hr { border-color: #2d3748; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "quick_replies_visible" not in st.session_state:
        st.session_state.quick_replies_visible = True


# ---------------------------------------------------------------------------
# Backend health
# ---------------------------------------------------------------------------

def backend_online() -> bool:
    try:
        return requests.get(API_HEALTH_URL, timeout=3).status_code == 200
    except requests.exceptions.RequestException:
        return False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Tentang Sistem")
        st.markdown(
            f"""
            **Model LLM** — 🤖 {os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")} (Groq Cloud)

            **Knowledge Base (RAG)**
            📚 Bitext CS Dataset — 26K+ Q&A pairs
            🔍 MMR Semantic Search via ChromaDB
            🌐 Multilingual Embeddings (MiniLM-L12)

            **Stack** — FastAPI · LangChain · ChromaDB
                         Sentence-Transformers · Streamlit

            **Version:** `{VERSION}`
            """,
            unsafe_allow_html=False,
        )
        st.divider()

        st.markdown("#### 🔌 Status Koneksi")
        if backend_online():
            st.markdown(
                '<div class="status-badge">'
                '<div class="status-dot"></div>Backend Online'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-badge offline">'
                '<div class="status-dot"></div>Backend Offline'
                '</div>',
                unsafe_allow_html=True,
            )
            st.warning("Jalankan `uvicorn api:app` di terminal untuk memulai backend.")

        st.divider()

        if st.button("🗑️ Hapus Percakapan", use_container_width=True):
            st.session_state.messages              = []
            st.session_state.quick_replies_visible = True
            st.session_state.session_id            = str(uuid.uuid4())
            st.rerun()

        st.markdown(
            "<p style='font-size:0.72rem;color:#4a5568;margin-top:16px;'>"
            "Sesi ini bersifat lokal dan tidak disimpan ke server.</p>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def render_header():
    st.markdown(
        f"<h1 style='margin:0;font-size:1.8rem;font-weight:700;"
        f"background:linear-gradient(90deg,#667eea,#764ba2);"
        f"-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>"
        f"{AGENT_AVATAR} {AGENT_NAME} — Customer Service AI</h1>",
        unsafe_allow_html=True,
    )
    st.caption(f"Powered by **Llama 3.3 70B** (Groq) + **Bitext RAG** (ChromaDB) · v{VERSION}")
    st.divider()


# ---------------------------------------------------------------------------
# Message history renderer
# ---------------------------------------------------------------------------

def render_messages():
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else AGENT_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            st.markdown(
                f'<div class="msg-caption">{msg["time"]}</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Quick-reply buttons
# ---------------------------------------------------------------------------

def render_quick_replies() -> str | None:
    if not st.session_state.quick_replies_visible:
        return None

    st.markdown("💡 **Pertanyaan Cepat:**")
    cols = st.columns(len(QUICK_REPLIES))
    for i, reply in enumerate(QUICK_REPLIES):
        if cols[i].button(reply, key=f"qr_{i}", use_container_width=True):
            return reply
    return None


# ---------------------------------------------------------------------------
# Backend streaming call
# ---------------------------------------------------------------------------

def stream_to_placeholder(prompt: str, placeholder) -> str:
    """
    POST the prompt to the streaming backend and render tokens
    token-by-token into the given Streamlit placeholder.
    """
    formatted_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    payload = {
        "message":    prompt,
        "session_id": st.session_state.session_id,
        "history":    formatted_history,
    }

    accumulated = ""

    # Guard: if backend is offline, fail immediately rather than hanging
    if not backend_online():
        msg = (
            "❌ **Backend Offline** — Pastikan server sudah berjalan:\n"
            "`cd src && uvicorn api:app --host 127.0.0.1 --port 8000`"
        )
        placeholder.markdown(msg)
        return msg

    try:
        with requests.post(
            API_STREAM_URL, json=payload, stream=True, timeout=120
        ) as response:
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    accumulated += chunk
                    placeholder.markdown(accumulated + "▌")

            placeholder.markdown(accumulated)
            return accumulated

    except requests.exceptions.Timeout:
        msg = "⏱️ **Waktu habis.** Server terlalu lama merespons. Coba lagi."
        placeholder.markdown(msg)
        return msg
    except requests.exceptions.HTTPError as exc:
        st.error(f"Server error ({exc.response.status_code}): periksa log backend.")
        placeholder.markdown(
            f"⚠️ Terjadi kesalahan server ({exc.response.status_code}). "
            "Silakan coba lagi dalam beberapa saat."
        )
        return ""
    except requests.exceptions.RequestException as exc:
        st.error(f"Koneksi gagal: {exc}")
        placeholder.markdown("❌ **Koneksi Gagal.** Pastikan backend sedang berjalan.")
        return ""


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main():
    page_config()
    inject_css()
    init_session()
    render_sidebar()
    render_header()
    render_messages()

    quick_reply_prompt = render_quick_replies()
    typed_prompt       = st.chat_input(
        f"Tanyakan sesuatu kepada {AGENT_NAME}…", key="chat_input"
    )

    prompt = typed_prompt or quick_reply_prompt
    if not prompt:
        return

    # ── Display user message ───────────────────────────────────────────────
    ts = datetime.now().strftime("%H:%M")
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        st.markdown(
            f'<div class="msg-caption">Anda · {ts}</div>',
            unsafe_allow_html=True,
        )

    st.session_state.messages.append(
        {"role": "user", "content": prompt, "time": ts}
    )
    st.session_state.quick_replies_visible = False

    # ── Display agent streaming response ────────────────────────────────────
    with st.chat_message("assistant", avatar=AGENT_AVATAR):
        with st.spinner("Aria sedang mengetik…"):
            ai_response = stream_to_placeholder(prompt, st.empty())

        ts_agent = datetime.now().strftime("%H:%M")
        st.markdown(
            f'<div class="msg-caption">{AGENT_NAME} · {ts_agent}</div>',
            unsafe_allow_html=True,
        )

    if ai_response:
        st.session_state.messages.append(
            {"role": "assistant", "content": ai_response.strip(), "time": ts_agent}
        )
    st.rerun()


if __name__ == "__main__":
    main()

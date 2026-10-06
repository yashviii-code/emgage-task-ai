
import io
import os
import sys
import time
from typing import List, Dict, Any, Optional

import requests
import streamlit as st
from dotenv import load_dotenv

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rag.retriever import EmgageRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf

# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DEFAULT_MODEL = "llama-3.1-8b-instant"
FALLBACK_MODEL = "openai/gpt-oss-120b"
TEMPERATURE = 0.1  # Set temperature low as requested for factual accuracy and sources
TOP_K = 6

# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Emgage Docs AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS FOR EXACT UI MATCH (DARK THEME)
# =========================================================

st.markdown("""
<style>
    /* Dark background styling */
    .stApp {
        background-color: #0B0F19;
        color: #F1F5F9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: #070A12;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Sidebar headers and text */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 24px;
        padding-bottom: 12px;
    }
    .sidebar-logo {
        width: 44px;
        height: 44px;
        border-radius: 10px;
        background: linear-gradient(135deg, #1E293B, #0F172A);
        border: 1.5px solid #3B82F6;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        box-shadow: 0 0 14px rgba(59, 130, 246, 0.35);
    }
    .sidebar-title {
        font-size: 19px;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: 0.3px;
        line-height: 1.2;
    }
    .sidebar-subtitle {
        font-size: 11.5px;
        color: #60A5FA;
        font-weight: 500;
        letter-spacing: 0.2px;
    }

    .sidebar-section-title {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.1px;
        color: #64748B;
        margin-top: 22px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .backend-item {
        font-size: 12.5px;
        color: #CBD5E1;
        margin-bottom: 8px;
        line-height: 1.4;
    }
    .backend-item strong {
        color: #E2E8F0;
    }

    /* Custom File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.6);
        border: 1px dashed rgba(59, 130, 246, 0.4);
        border-radius: 8px;
        padding: 10px;
    }
    [data-testid="stFileUploader"] label {
        color: #94A3B8 !important;
        font-size: 12px !important;
    }

    /* Main Area Header */
    .main-header-container {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-top: -10px;
        margin-bottom: 6px;
    }
    .main-header-title {
        font-size: 32px;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.5px;
    }
    .main-header-subtitle {
        font-size: 14.5px;
        color: #94A3B8;
        margin-bottom: 24px;
        line-height: 1.5;
    }

    /* Welcome Card */
    .welcome-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 28px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    }
    .welcome-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 14px;
    }
    .welcome-badge {
        background: #D97706;
        color: #FFFFFF;
        font-size: 11px;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 4px;
        letter-spacing: 0.5px;
    }
    .welcome-title {
        font-size: 18px;
        font-weight: 700;
        color: #F8FAFC;
    }
    .welcome-intro {
        color: #94A3B8;
        font-size: 14px;
        line-height: 1.6;
        margin-bottom: 18px;
    }
    .welcome-section-title {
        font-size: 14.5px;
        font-weight: 700;
        color: #F1F5F9;
        margin-bottom: 10px;
    }
    .welcome-list {
        color: #CBD5E1;
        font-size: 13.5px;
        line-height: 1.9;
        margin: 0;
        padding-left: 20px;
    }

    /* Chat message bubble styling */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding: 16px 0;
    }

    /* Chat Input Bar */
    [data-testid="stChatInput"] {
        background: #0D1322 !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 10px !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #F8FAFC !important;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# RETRIEVER CACHE
# =========================================================

@st.cache_resource
def load_retriever():
    """Load ChromaDB retriever with all-MiniLM-L6-v2."""
    retriever = EmgageRetriever()
    retriever.load()
    return retriever


try:
    retriever = load_retriever()
    rag_available = True
except Exception as e:
    retriever = None
    rag_available = False


# =========================================================
# DIRECT HTTP REQUESTS TO GROQ API
# =========================================================

def call_groq_api_direct(
    messages: List[Dict[str, str]],
    temperature: float = TEMPERATURE,
    max_tokens: int = 1500
) -> str:
    """
    Direct HTTP request to Groq API using python requests.
    Tries Groq llama-3.1-8b-instant first, with graceful fallback if model is unavailable.
    """
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY not configured in .env file."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    models_to_try = [DEFAULT_MODEL, FALLBACK_MODEL]

    for model_name in models_to_try:
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)

            if resp.status_code == 200:
                data = resp.json()
                choice = data["choices"][0]["message"]
                content = choice.get("content") or ""
                # If a reasoning model was used and content is empty, use reasoning
                if not content.strip() and "reasoning" in choice:
                    content = choice.get("reasoning", "")
                return content.strip()

            elif resp.status_code == 404:
                # Try next model if model not found
                continue

            elif resp.status_code == 429:
                time.sleep(2)
                continue

            else:
                err_msg = resp.json().get("error", {}).get("message", resp.text)
                return f"Groq API Error ({resp.status_code}): {err_msg}"

        except requests.RequestException as e:
            return f"Network error during Groq API call: {str(e)}"

    return f"Failed to complete request with Groq API (models tried: {', '.join(models_to_try)})."


# =========================================================
# PDF PROCESSING HELPER
# =========================================================

def process_uploaded_pdf(file_bytes: bytes, filename: str) -> int:
    """Extract text from uploaded PDF, chunk it, and add to ChromaDB."""
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    full_text = ""
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            full_text += f"\n--- Page {idx + 1} ---\n{text}"

    if not full_text.strip():
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(full_text)

    if chunks and retriever:
        retriever.add_pdf_content(filename, chunks)

    return len(chunks)


# =========================================================
# SYSTEM PROMPT (Strictly Grounded with Sources)
# =========================================================

RAG_SYSTEM_PROMPT = """You are an expert AI documentation assistant for Emgage HRMS.
Answer the user's question accurately and concisely based strictly on the retrieved documentation and uploaded PDF context provided below.

--- DOCUMENTATION CONTEXT ---
{context}
--- END CONTEXT ---

Guidelines:
1. Answer factually and accurately using the context above.
2. Structure your answer with clear bullet points and bold headers for readability.
3. If relevant, include a 'Key Notes' section highlighting essential information.
4. If the retrieved context contains relevant pages, you MUST list them at the very end under '### Sources' in markdown link format:
   - [Document Title](URL)
   For uploaded PDFs, cite them as:
   - [Uploaded Document: filename]
5. If the documentation does not contain enough information, state that clearly without fabricating facts.
"""

NO_RAG_SYSTEM_PROMPT = """You are an expert AI documentation assistant for Emgage HRMS.
The knowledge base is currently initializing or unavailable. Answer using general HRMS domain knowledge, clearly noting that it is general guidance.
"""


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    # Branding
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">🤖</div>
        <div>
            <div class="sidebar-title">Emgage AI</div>
            <div class="sidebar-subtitle">Multi-Agent RAG System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Section 1: UPLOAD PDF DOCUMENTS
    st.markdown('<div class="sidebar-section-title">📁 UPLOAD PDF DOCUMENTS</div>', unsafe_allow_html=True)
    
    if "indexed_pdfs" not in st.session_state:
        st.session_state.indexed_pdfs = set()

    uploaded_pdf = st.file_uploader(
        "upload/upload",
        type=["pdf"],
        help="Upload PDF documents (up to 200MB) to index and query seamlessly alongside official docs."
    )

    if uploaded_pdf is not None:
        if uploaded_pdf.name not in st.session_state.indexed_pdfs:
            with st.spinner(f"Indexing '{uploaded_pdf.name}' into ChromaDB..."):
                chunk_count = process_uploaded_pdf(uploaded_pdf.getvalue(), uploaded_pdf.name)
                if chunk_count > 0:
                    st.session_state.indexed_pdfs.add(uploaded_pdf.name)
                    st.success(f"✓ Indexed {chunk_count} chunks from {uploaded_pdf.name}")
                else:
                    st.error("No readable text found in uploaded PDF.")
        else:
            st.info(f"✓ {uploaded_pdf.name} is indexed and ready")

    st.caption("200MB per file • PDF")

    # Section 2: SYSTEM BACKEND
    st.markdown('<div class="sidebar-section-title">⚙️ SYSTEM BACKEND</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="backend-item">⚡ <strong>LLM:</strong> Groq (llama-3.1-8b-instant)</div>
    <div class="backend-item">🌐 <strong>API:</strong> Direct HTTP (requests)</div>
    <div class="backend-item">🧠 <strong>Embedding:</strong> all-MiniLM-L6-v2</div>
    <div class="backend-item">🗄️ <strong>Vector DB:</strong> ChromaDB</div>
    """, unsafe_allow_html=True)

    # Section 3: ACTIVE AGENTS
    st.markdown('<div class="sidebar-section-title">✳️ ACTIVE AGENTS</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="backend-item">⚙️ <strong>RAG:</strong> ChromaDB retrieval</div>
    <div class="backend-item">⚡ <strong>LLM:</strong> Groq direct API call</div>
    """, unsafe_allow_html=True)


# =========================================================
# MAIN AREA HEADER
# =========================================================

st.markdown("""
<div class="main-header-container">
    <div style="width: 42px; height: 42px; border-radius: 10px; background: #2563EB; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 15px rgba(37, 99, 235, 0.45);">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="white">
            <path d="M12 2L2 7v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-10-5zm-1 6h2v2h-2V8zm0 4h2v6h-2v-6z"/>
        </svg>
    </div>
    <div class="main-header-title">Emgage Documentation Assistant</div>
</div>
<div class="main-header-subtitle">
    Ask technical questions, generate API code, or query uploaded PDF documentation seamlessly.
</div>
""", unsafe_allow_html=True)


# =========================================================
# CHAT SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# WELCOME HERO CARD (Displayed when no messages yet)
# =========================================================

if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-header">
            <span class="welcome-badge">smart_</span>
            <span class="welcome-title">Welcome to Emgage Docs AI! 👋</span>
        </div>
        <p class="welcome-intro">
            I am a multi-agent system designed to answer technical questions and generate code grounded strictly in official documentation and uploaded PDF files.
        </p>
        <div class="welcome-section-title">What you can do:</div>
        <ul class="welcome-list">
            <li><strong>Documentation Q&A:</strong> "How does authentication work in Emgage?"</li>
            <li><strong>API Code Generation:</strong> "Generate Python example for Login API"</li>
            <li><strong>Feature Overview:</strong> "What API endpoints are available?"</li>
            <li><strong>Query PDFs:</strong> Upload any PDF document in the sidebar to search and retrieve insights</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =========================================================
# CHAT INPUT & EXECUTION
# =========================================================

user_input = st.chat_input("Ask a question about Emgage documentation or your uploaded PDFs...")

if user_input:
    # 1. Display user query
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # 2. Retrieve relevant context via ChromaDB
    context = ""
    sources = []

    if rag_available and retriever:
        with st.spinner("Searching ChromaDB documentation..."):
            try:
                results = retriever.search(user_input, top_k=TOP_K)
                context = retriever.get_context(user_input, top_k=TOP_K)

                seen = set()
                for r in results:
                    src = r.get("source", "")
                    raw_title = r.get("title", "")
                    # Clean title: remove " | Emgage HRMS Documentation" or " - Emgage..."
                    clean_title = raw_title.split("|")[0].split(" - ")[0].strip() or "Documentation"
                    if src and src not in seen:
                        seen.add(src)
                        sources.append({
                            "title": clean_title,
                            "url": src,
                            "is_pdf": r.get("is_pdf", False)
                        })
                sources = sources[:4]
            except Exception as e:
                context = ""
                st.warning(f"ChromaDB retrieval notice: {e}")

    # 3. Format system prompt
    if context:
        sys_prompt = RAG_SYSTEM_PROMPT.format(context=context)
    else:
        sys_prompt = NO_RAG_SYSTEM_PROMPT

    messages_for_llm = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_input}
    ]

    # 4. Generate response via Direct HTTP API call to Groq
    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            raw_answer = call_groq_api_direct(messages_for_llm, temperature=TEMPERATURE)

            # Check if answer contains Sources header; if not and sources exist, append cleanly
            final_answer = raw_answer.strip()
            if sources and "### Sources" not in final_answer and "## Sources" not in final_answer and "Sources\n" not in final_answer:
                sources_md = "\n\n### Sources\n"
                for s in sources:
                    if s.get("is_pdf"):
                        sources_md += f"• [{s['title']}]\n"
                    elif s['url'].startswith("http"):
                        sources_md += f"• [{s['title']}]({s['url']})\n"
                    else:
                        sources_md += f"• [{s['title']}]\n"
                final_answer += sources_md

            st.markdown(final_answer)

    # 5. Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer
    })
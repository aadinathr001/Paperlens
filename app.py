"""
app.py — PaperLens Streamlit UI
Session-based RAG assistant for PDF/TXT documents.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

# Load .env (works locally; on Render set env vars in dashboard)
load_dotenv()

from document_processor import (
    MAX_FILE_SIZE_MB,
    MAX_FILES,
    process_file,
    validate_file,
)
from rag import TOP_K, build_faiss_index, run_rag

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PaperLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* ── Global typography ── */
        html, body, [class*="css"] {
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }

        /* ── Hero header ── */
        .hero-header {
            background: linear-gradient(135deg, #1e3a5f 0%, #0f6291 50%, #1a9ecf 100%);
            border-radius: 16px;
            padding: 2rem 2.5rem;
            margin-bottom: 1.5rem;
            text-align: center;
            box-shadow: 0 4px 24px rgba(15, 98, 145, 0.25);
        }
        .hero-header h1 {
            color: #ffffff;
            font-size: 2.6rem;
            font-weight: 800;
            margin: 0 0 0.3rem 0;
            letter-spacing: -0.5px;
        }
        .hero-header p {
            color: #b8dff5;
            font-size: 1.05rem;
            margin: 0;
        }

        /* ── Answer card ── */
        .answer-card {
            background: #f0f8ff;
            border-left: 5px solid #1a9ecf;
            border-radius: 10px;
            padding: 1.2rem 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 12px rgba(26, 158, 207, 0.1);
        }
        .answer-card p {
            margin: 0;
            font-size: 1rem;
            line-height: 1.7;
            color: #1a2b3c;
        }

        /* ── Source badge ── */
        .source-badge {
            display: inline-block;
            background: #e1f5fe;
            color: #0277bd;
            border: 1px solid #81d4fa;
            border-radius: 20px;
            padding: 0.2rem 0.75rem;
            font-size: 0.8rem;
            font-weight: 600;
            margin: 0.2rem 0.2rem 0.2rem 0;
        }

        /* ── Chunk card ── */
        .chunk-card {
            background: #fafafa;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.6rem;
        }
        .chunk-card .chunk-meta {
            font-size: 0.78rem;
            color: #888;
            font-weight: 600;
            margin-bottom: 0.4rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .chunk-card .chunk-text {
            font-size: 0.9rem;
            color: #333;
            line-height: 1.6;
        }
        .chunk-card .chunk-score {
            font-size: 0.75rem;
            color: #aaa;
            margin-top: 0.4rem;
        }

        /* ── Sidebar tweaks ── */
        .sidebar-section-title {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #888;
            margin-top: 1rem;
            margin-bottom: 0.3rem;
        }

        /* ── Status pills ── */
        .status-ready {
            background: #e8f5e9;
            color: #2e7d32;
            border: 1px solid #a5d6a7;
            border-radius: 20px;
            padding: 0.2rem 0.8rem;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .status-waiting {
            background: #fff3e0;
            color: #e65100;
            border: 1px solid #ffcc80;
            border-radius: 20px;
            padding: 0.2rem 0.8rem;
            font-size: 0.82rem;
            font-weight: 600;
        }

        /* Streamlit widget overrides */
        div[data-testid="stFileUploader"] label {
            font-weight: 600;
        }
        .stTextInput > label, .stTextArea > label {
            font-weight: 600;
        }
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    defaults: Dict[str, Any] = {
        "faiss_index": None,
        "metadata": [],           # List of chunk dicts
        "processed_files": [],    # Names of successfully processed files
        "last_answer": None,
        "last_chunks": [],
        "last_question": "",
        "processing_error": None,
        "groq_api_key": os.environ.get("GROQ_API_KEY", ""),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session() -> None:
    """Clear all document/index state but keep the API key."""
    api_key = st.session_state.get("groq_api_key", "")
    keys_to_clear = [
        "faiss_index", "metadata", "processed_files",
        "last_answer", "last_chunks", "last_question", "processing_error",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    init_session_state()
    st.session_state["groq_api_key"] = api_key


# ---------------------------------------------------------------------------
# Helper: index is ready
# ---------------------------------------------------------------------------
def index_is_ready() -> bool:
    return (
        st.session_state.get("faiss_index") is not None
        and len(st.session_state.get("metadata", [])) > 0
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🔍 PaperLens")
        st.markdown("*Session-based document Q&A*")
        st.divider()

        # ── Session status ──
        st.markdown('<div class="sidebar-section-title">Session Status</div>', unsafe_allow_html=True)

        if index_is_ready():
            n_chunks = len(st.session_state["metadata"])
            n_files = len(st.session_state["processed_files"])
            st.markdown(
                f'<span class="status-ready">✓ Ready</span>', unsafe_allow_html=True
            )
            st.markdown(f"**{n_files}** file(s) · **{n_chunks}** chunks indexed")
            st.markdown("**Loaded files:**")
            for fname in st.session_state["processed_files"]:
                st.markdown(f"• `{fname}`")
        else:
            st.markdown(
                '<span class="status-waiting">○ Waiting for documents</span>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Clear session ──
        if st.button("🗑️ Clear Session & Reset", use_container_width=True):
            reset_session()
            st.success("Session cleared!")
            st.rerun()

        st.divider()

        # ── Limits ──
        st.markdown('<div class="sidebar-section-title">Limits</div>', unsafe_allow_html=True)
        st.markdown(
            f"• Max **{MAX_FILES}** files per session  \n"
            f"• Max **{MAX_FILE_SIZE_MB} MB** per file  \n"
            f"• Top-**{TOP_K}** chunks retrieved  \n"
            f"• PDF & TXT only"
        )


# ---------------------------------------------------------------------------
# Document upload + processing panel
# ---------------------------------------------------------------------------
def render_upload_panel() -> None:
    st.markdown("### 📄 Upload Documents")
    st.markdown(
        f"Upload up to **{MAX_FILES}** PDF or TXT files (max {MAX_FILE_SIZE_MB} MB each). "
        "Documents are processed in-session only — nothing is stored."
    )

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help=f"Max {MAX_FILES} files, {MAX_FILE_SIZE_MB} MB each",
        label_visibility="collapsed",
    )

    if not uploaded_files:
        return

    # ── Validate count ──
    if len(uploaded_files) > MAX_FILES:
        st.error(
            f"❌ Too many files. You uploaded {len(uploaded_files)}, but the limit is {MAX_FILES}."
        )
        return

    # ── Validate each file ──
    errors: List[str] = []
    valid_files = []
    for uf in uploaded_files:
        err = validate_file(uf.name, uf.size)
        if err:
            errors.append(err)
        else:
            valid_files.append(uf)

    if errors:
        for err in errors:
            st.error(err)
        if not valid_files:
            return
        st.warning("Proceeding with valid files only.")

    # ── Check if these are new files (avoid reprocessing on every Streamlit rerun) ──
    incoming_names = sorted([f.name for f in valid_files])
    already_processed = sorted(st.session_state.get("processed_files", []))
    if incoming_names == already_processed and index_is_ready():
        return  # Nothing new to do

    # ── Process ──
    if st.button("⚙️ Process Documents", use_container_width=True, type="primary"):
        all_chunks: List[Dict[str, Any]] = []
        success_names: List[str] = []
        proc_errors: List[str] = []

        progress = st.progress(0, text="Processing documents…")

        for i, uf in enumerate(valid_files):
            progress.progress((i) / len(valid_files), text=f"Processing '{uf.name}'…")
            try:
                file_bytes = uf.read()
                chunks = process_file(file_bytes, uf.name)
                all_chunks.extend(chunks)
                success_names.append(uf.name)
                st.success(f"✅ '{uf.name}' — {len(chunks)} chunks extracted")
            except Exception as exc:
                proc_errors.append(f"'{uf.name}': {exc}")
                st.error(f"❌ {proc_errors[-1]}")

        progress.progress(1.0, text="Building FAISS index…")

        if proc_errors and not all_chunks:
            st.error("All files failed to process. Please check your files and try again.")
            progress.empty()
            return

        try:
            index, metadata = build_faiss_index(all_chunks)
            st.session_state["faiss_index"] = index
            st.session_state["metadata"] = metadata
            st.session_state["processed_files"] = success_names
            st.session_state["last_answer"] = None
            st.session_state["last_chunks"] = []
            st.session_state["last_question"] = ""
            progress.empty()
            st.success(
                f"🚀 Index ready! {len(all_chunks)} chunks from {len(success_names)} file(s) indexed."
            )
            st.rerun()
        except Exception as exc:
            progress.empty()
            st.error(f"❌ Failed to build FAISS index: {exc}")


# ---------------------------------------------------------------------------
# Q&A panel
# ---------------------------------------------------------------------------
def render_qa_panel() -> None:
    st.markdown("### ❓ Ask a Question")

    if not index_is_ready():
        st.info("⬆️ Upload and process documents first, then ask questions here.")
        return


    with st.form("qa_form", clear_on_submit=False):
        question = st.text_area(
            "Your question",
            placeholder="e.g. What is the main contribution of this paper?",
            height=100,
            key="question_input",
        )
        submitted = st.form_submit_button("🔎 Search & Answer", use_container_width=True, type="primary")

    if submitted:
        question = question.strip()
        if not question:
            st.warning("⚠️ Please enter a question before submitting.")
            return

        with st.spinner("Retrieving relevant chunks and generating answer…"):
            try:
                answer, retrieved_chunks = run_rag(
                    question=question,
                    index=st.session_state["faiss_index"],
                    metadata=st.session_state["metadata"],
                    groq_api_key=st.session_state.get("groq_api_key"),
                )
                st.session_state["last_answer"] = answer
                st.session_state["last_chunks"] = retrieved_chunks
                st.session_state["last_question"] = question
            except ValueError as exc:
                st.error(f"❌ {exc}")
                return
            except Exception as exc:
                st.error(f"❌ Unexpected error: {exc}")
                return

    # ── Render previous/current answer ──
    if st.session_state.get("last_answer"):
        _render_answer()


def _render_answer() -> None:
    answer: str = st.session_state["last_answer"]
    chunks: List[Dict[str, Any]] = st.session_state["last_chunks"]
    question: str = st.session_state["last_question"]

    st.divider()
    st.markdown("#### 💬 Answer")
    st.markdown(
        f'<div class="answer-card"><p>{_escape_html(answer)}</p></div>',
        unsafe_allow_html=True,
    )

    # ── Source references ──
    if chunks:
        st.markdown("**📌 Source References**")
        seen_sources = set()
        badges_html = ""
        for chunk in chunks:
            src = chunk["source"]
            pg = chunk.get("page")
            key = (src, pg)
            if key not in seen_sources:
                seen_sources.add(key)
                label = f"{src} — Page {pg}" if pg else src
                badges_html += f'<span class="source-badge">📄 {_escape_html(label)}</span>'
        st.markdown(badges_html, unsafe_allow_html=True)

    # ── Retrieved chunks (expandable) ──
    if chunks:
        with st.expander(f"🔬 Inspect Retrieved Chunks (Top {len(chunks)})", expanded=False):
            for i, chunk in enumerate(chunks, start=1):
                src = chunk["source"]
                pg = chunk.get("page")
                score = chunk.get("score", 0.0)
                location = f"Page {pg}" if pg else "TXT file"
                meta_line = f"Chunk {i} · {src} · {location}"
                score_line = f"Similarity score: {score:.4f}"

                st.markdown(
                    f"""
                    <div class="chunk-card">
                        <div class="chunk-meta">{_escape_html(meta_line)}</div>
                        <div class="chunk-text">{_escape_html(chunk['text'][:800])}{'…' if len(chunk['text']) > 800 else ''}</div>
                        <div class="chunk-score">{score_line}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# HTML escape helper
# ---------------------------------------------------------------------------
def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
        .replace("\n", "<br>")
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    init_session_state()
    render_sidebar()

    # ── Hero header ──
    st.markdown(
        """
        <div class="hero-header">
            <h1>🔍 PaperLens</h1>
            <p>Upload PDF or TXT documents and ask questions — powered by Sentence Transformers, FAISS, and Groq LLM</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Two-column layout ──
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        render_upload_panel()

    with col_right:
        render_qa_panel()

    # ── Footer ──
    st.divider()
    st.markdown(
        "<div style='text-align:center; color:#aaa; font-size:0.8rem;'>"
        "PaperLens · Session-based RAG · No data is stored · "
        "Powered by <b>Sentence Transformers</b> + <b>FAISS</b> + <b>Groq</b>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

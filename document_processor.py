"""
document_processor.py
Handles PDF/TXT extraction, text cleaning, and chunking with page metadata.
"""

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_FILE_SIZE_MB: int = 10
MAX_FILES: int = 5
ALLOWED_EXTENSIONS: set[str] = {".pdf", ".txt"}
CHUNK_WORD_SIZE: int = 600
CHUNK_OVERLAP_WORDS: int = 100


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_file(file_name: str, file_size_bytes: int) -> Optional[str]:
    """
    Returns an error string if the file is invalid, otherwise None.
    """
    ext = _get_extension(file_name)
    if ext not in ALLOWED_EXTENSIONS:
        return f"❌ '{file_name}': Unsupported file type '{ext}'. Only PDF and TXT files are allowed."
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size_bytes > max_bytes:
        size_mb = file_size_bytes / (1024 * 1024)
        return f"❌ '{file_name}': File size {size_mb:.1f} MB exceeds the {MAX_FILE_SIZE_MB} MB limit."
    return None


def _get_extension(file_name: str) -> str:
    """Return lowercase file extension including the dot."""
    dot_idx = file_name.rfind(".")
    if dot_idx == -1:
        return ""
    return file_name[dot_idx:].lower()


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes, file_name: str) -> List[Dict[str, Any]]:
    """
    Extract text from a PDF, one dict per page.
    Returns list of {"text": str, "source": str, "page": int}
    """
    import pymupdf  # PyMuPDF

    pages: List[Dict[str, Any]] = []
    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc[page_num]
            raw_text = page.get_text("text")
            cleaned = clean_text(raw_text)
            if cleaned.strip():
                pages.append({
                    "text": cleaned,
                    "source": file_name,
                    "page": page_num + 1,  # 1-indexed
                })
        doc.close()
    except Exception as exc:
        raise ValueError(f"Failed to extract text from PDF '{file_name}': {exc}") from exc

    if not pages:
        raise ValueError(f"No readable text found in '{file_name}'. The PDF may be scanned or image-based.")

    return pages


def extract_text_from_txt(file_bytes: bytes, file_name: str) -> List[Dict[str, Any]]:
    """
    Extract text from a plain-text file.
    Returns a single-item list with {"text": str, "source": str, "page": None}
    """
    try:
        raw_text = file_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        raise ValueError(f"Failed to decode '{file_name}': {exc}") from exc

    cleaned = clean_text(raw_text)
    if not cleaned.strip():
        raise ValueError(f"No readable text found in '{file_name}'.")

    return [{"text": cleaned, "source": file_name, "page": None}]


def extract_text(file_bytes: bytes, file_name: str) -> List[Dict[str, Any]]:
    """Dispatch to the correct extractor based on file extension."""
    ext = _get_extension(file_name)
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes, file_name)
    elif ext == ".txt":
        return extract_text_from_txt(file_bytes, file_name)
    else:
        raise ValueError(f"Unsupported file type: '{ext}'")


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalise whitespace, collapse multiple blank lines, strip control chars.
    """
    # Remove null bytes and other non-printable control chars (keep \n \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    # Collapse runs of spaces/tabs into a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse more than 2 consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = CHUNK_WORD_SIZE,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> List[Dict[str, Any]]:
    """
    Split page-level dicts into word-based chunks with overlap.
    Each chunk carries {"text", "source", "page"} metadata.
    'page' is the page where the chunk *starts*.
    """
    chunks: List[Dict[str, Any]] = []

    for page_info in pages:
        source = page_info["source"]
        page = page_info["page"]
        words = page_info["text"].split()

        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "text": chunk_text,
                "source": source,
                "page": page,
            })
            if end == len(words):
                break
            start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------------------------
# Public pipeline entry point
# ---------------------------------------------------------------------------

def process_file(file_bytes: bytes, file_name: str) -> List[Dict[str, Any]]:
    """
    Full pipeline: extract → clean (already done inside extract) → chunk.
    Returns list of chunk dicts with metadata.
    """
    pages = extract_text(file_bytes, file_name)
    chunks = chunk_pages(pages)
    return chunks

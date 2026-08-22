
import os
import re
import glob
import json
from dataclasses import dataclass, asdict
from typing import List

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    text: str
    word_count: int


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    if not HAS_PDFPLUMBER:
        raise RuntimeError(
            "pdfplumber is not installed. Run: pip install pdfplumber"
        )
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def _extract_metadata(raw_text: str):

    source, title = "unknown", "unknown"
    lines = raw_text.splitlines()
    body_start = 0
    for i, line in enumerate(lines[:5]):
        if line.startswith("Source:"):
            source = line.replace("Source:", "").strip()
            body_start = i + 1
        elif line.startswith("Title:"):
            title = line.replace("Title:", "").strip()
            body_start = i + 1
    body = "\n".join(lines[body_start:]).strip()
    return source, title, body


def clean_text(text: str) -> str:

    text = text.replace("\r", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def load_corpus(raw_dir: str) -> List[dict]:

    docs = []
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.txt"))) + sorted(
        glob.glob(os.path.join(raw_dir, "*.pdf"))
    )
    for path in paths:
        doc_id = os.path.splitext(os.path.basename(path))[0]
        if path.endswith(".pdf"):
            raw_text = _read_pdf(path)
            source, title = path, doc_id
            body = raw_text
        else:
            raw_text = _read_txt(path)
            source, title, body = _extract_metadata(raw_text)
        body = clean_text(body)
        docs.append({"doc_id": doc_id, "source": source, "title": title, "text": body})
    return docs


def chunk_document(doc: dict, chunk_size: int = 150, overlap: int = 30) -> List[Chunk]:
  
    words = doc["text"].split()
    chunks = []
    if not words:
        return chunks

    step = max(chunk_size - overlap, 1)
    idx = 0
    chunk_no = 0
    while idx < len(words):
        window = words[idx: idx + chunk_size]
        if not window:
            break
        chunk_text = " ".join(window)
        chunk_id = f"{doc['doc_id']}::chunk{chunk_no}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc["doc_id"],
                source=doc["source"],
                title=doc["title"],
                text=chunk_text,
                word_count=len(window),
            )
        )
        chunk_no += 1
        idx += step
    return chunks


def build_chunk_corpus(
    raw_dir: str, chunk_size: int = 150, overlap: int = 30
) -> List[Chunk]:
    docs = load_corpus(raw_dir)
    all_chunks: List[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    return all_chunks


def corpus_stats(docs: List[dict], chunks: List[Chunk]) -> dict:
    doc_word_counts = [len(d["text"].split()) for d in docs]
    chunk_word_counts = [c.word_count for c in chunks]
    return {
        "num_documents": len(docs),
        "num_chunks": len(chunks),
        "avg_chunks_per_doc": round(len(chunks) / max(len(docs), 1), 2),
        "total_words": sum(doc_word_counts),
        "avg_doc_words": round(sum(doc_word_counts) / max(len(docs), 1), 1),
        "min_doc_words": min(doc_word_counts) if doc_word_counts else 0,
        "max_doc_words": max(doc_word_counts) if doc_word_counts else 0,
        "avg_chunk_words": round(sum(chunk_word_counts) / max(len(chunks), 1), 1),
    }


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(here, "..", "data", "raw_corpus")
    docs = load_corpus(raw_dir)
    chunks = build_chunk_corpus(raw_dir, chunk_size=150, overlap=30)
    stats = corpus_stats(docs, chunks)
    print(json.dumps(stats, indent=2))

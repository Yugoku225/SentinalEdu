"""Reference-document retrieval (TASKS 4.3 / 4.4, DECISIONS D-07, D-09).

Ingest: PDF/TXT -> per-page text -> ~500-character chunks on sentence boundaries with
100 characters of overlap -> SBERT vectors -> one FAISS IndexFlatIP per assignment.
Persisted as storage/indexes/<assignment_id>/{index.faiss, chunks.json}.

Retrieve: `retrieve(assignment_id, query, k)` returns the top-k chunks with cosine scores.
The caller (Agent 2) injects them into the prompt with their chunk ids, so citations can be
checked against exactly what the model saw.
"""
import json
import logging
import re
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
from pydantic import BaseModel, Field

from sentineledu.config import Settings, get_settings
from sentineledu.services import embeddings

log = logging.getLogger("sentineledu.rag")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TEXT_SUFFIXES = {".txt", ".md", ".text"}
INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"


class Chunk(BaseModel):
    chunk_id: str  # "d1.p2.c3": document 1, page 2, chunk 3
    source: str  # file name of the reference document
    page: int = Field(ge=1)
    text: str


class RetrievedChunk(Chunk):
    score: float  # cosine similarity to the query


class RagError(RuntimeError):
    pass


# --- text extraction and chunking -------------------------------------------------------------

def extract_pages(path: Path | str) -> list[tuple[int, str]]:
    """(page_number, text) for each page with text. TXT/MD count as one page."""
    path = Path(path)
    if not path.is_file():
        raise RagError(f"Reference document not found: {path}")
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [(1, text)] if text.strip() else []
    if suffix == ".pdf" or path.read_bytes()[:5] == b"%PDF-":
        from PyPDF2 import PdfReader  # same reader as the legacy src/pdf_utils.py

        pages = []
        for number, page in enumerate(PdfReader(str(path)).pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((number, text))
        return pages
    raise RagError(f"Unsupported reference document type: {path.name} (use PDF or TXT)")


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def split_sentences(text: str) -> list[str]:
    parts = [re.sub(r"\s+", " ", s).strip() for s in _SENTENCE_END.split(text)]
    return [p for p in parts if p]


def _hard_split(sentence: str, size: int) -> list[str]:
    """A single sentence longer than `size` is split on word boundaries."""
    words, out, cur = sentence.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > size:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Greedy sentence packing: ~`size` chars per chunk, the next chunk repeats up to `overlap`
    trailing chars of sentences from the previous one."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("need 0 <= overlap < size")
    sentences: list[str] = []
    for s in split_sentences(text):
        sentences.extend(_hard_split(s, size) if len(s) > size else [s])

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for s in sentences:
        if current and length + 1 + len(s) > size:
            chunks.append(" ".join(current))
            # carry over trailing sentences up to `overlap` characters
            carried, carried_len = [], 0
            for prev in reversed(current):
                if carried_len + len(prev) + 1 > overlap:
                    break
                carried.insert(0, prev)
                carried_len += len(prev) + 1
            current, length = carried, max(carried_len - 1, 0)
        current.append(s)
        length += len(s) + (1 if length else 0)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(path: Path | str, doc_tag: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    path = Path(path)
    chunks = []
    for page, text in extract_pages(path):
        for n, piece in enumerate(chunk_text(text, size, overlap), start=1):
            chunks.append(Chunk(chunk_id=f"{doc_tag}.p{page}.c{n}", source=path.name, page=page, text=piece))
    return chunks


# --- the per-assignment store -----------------------------------------------------------------

class RagStore:
    """One FAISS index per assignment under <storage_dir>/indexes/<assignment_id>/."""

    def __init__(self, settings: Settings | None = None, *, root: Path | None = None):
        self.settings = settings or get_settings()
        self.root = Path(root or self.settings.storage_dir / "indexes")
        self._loaded: dict[str, tuple[float, faiss.Index, list[Chunk], dict]] = {}

    def dir(self, assignment_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", assignment_id):
            raise RagError(f"assignment_id {assignment_id!r} may only contain letters, digits, '_', '.' and '-'")
        return self.root / assignment_id

    def has_index(self, assignment_id: str) -> bool:
        d = self.dir(assignment_id)
        return (d / INDEX_FILE).is_file() and (d / CHUNKS_FILE).is_file()

    # ingestion ---------------------------------------------------------------------------------

    def ingest(self, assignment_id: str, paths: Sequence[Path | str], *, replace: bool = False) -> int:
        """Add reference documents to the assignment's index (or replace it). Returns the chunk count."""
        new_paths = [Path(p).resolve() for p in paths]
        for p in new_paths:
            if not p.is_file():
                raise RagError(f"Reference document not found: {p}")
        sources = [] if replace or not self.has_index(assignment_id) else self.sources(assignment_id)
        for p in new_paths:
            if str(p) not in sources:
                sources.append(str(p))
        return self._build(assignment_id, sources)

    def rebuild(self, assignment_id: str) -> int:
        """Re-chunk and re-embed the stored source documents (e.g. after one was removed on disk)."""
        if not self.has_index(assignment_id):
            raise RagError(f"No index for assignment {assignment_id!r}")
        return self._build(assignment_id, [s for s in self.sources(assignment_id) if Path(s).is_file()])

    def remove_source(self, assignment_id: str, path: Path | str) -> int:
        target = str(Path(path).resolve())
        remaining = [s for s in self.sources(assignment_id) if s != target]
        return self._build(assignment_id, remaining)

    def delete(self, assignment_id: str) -> None:
        d = self.dir(assignment_id)
        for name in (INDEX_FILE, CHUNKS_FILE):
            (d / name).unlink(missing_ok=True)
        self._loaded.pop(assignment_id, None)

    def _build(self, assignment_id: str, sources: list[str]) -> int:
        chunks: list[Chunk] = []
        for i, src in enumerate(sources, start=1):
            chunks.extend(chunk_document(src, f"d{i}"))
        dim = embeddings.dimension(self.settings)
        index = faiss.IndexFlatIP(dim)
        if chunks:
            vectors = embeddings.encode([c.text for c in chunks], self.settings)
            index.add(vectors)
        d = self.dir(assignment_id)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(d / INDEX_FILE))
        meta = {
            "assignment_id": assignment_id,
            "embedding_model": self.settings.embedding_model,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "sources": sources,
            "chunks": [c.model_dump() for c in chunks],
        }
        (d / CHUNKS_FILE).write_text(json.dumps(meta, indent=1), encoding="utf-8")
        self._loaded.pop(assignment_id, None)
        log.info("Indexed %d chunks from %d document(s) for %s", len(chunks), len(sources), assignment_id)
        return len(chunks)

    # reading -----------------------------------------------------------------------------------

    def _load(self, assignment_id: str) -> tuple[faiss.Index, list[Chunk], dict]:
        d = self.dir(assignment_id)
        if not self.has_index(assignment_id):
            raise RagError(f"No reference documents have been indexed for assignment {assignment_id!r}")
        stamp = (d / CHUNKS_FILE).stat().st_mtime
        cached = self._loaded.get(assignment_id)
        if cached and cached[0] == stamp:
            return cached[1:]
        meta = json.loads((d / CHUNKS_FILE).read_text(encoding="utf-8"))
        index = faiss.read_index(str(d / INDEX_FILE))
        chunks = [Chunk.model_validate(c) for c in meta["chunks"]]
        if index.ntotal != len(chunks):
            raise RagError(f"Index for {assignment_id!r} is corrupt ({index.ntotal} vectors, {len(chunks)} chunks); rebuild it")
        if meta.get("embedding_model") != self.settings.embedding_model:
            raise RagError(f"Index for {assignment_id!r} was built with {meta.get('embedding_model')}, "
                           f"not {self.settings.embedding_model}; rebuild it")
        self._loaded[assignment_id] = (stamp, index, chunks, meta)
        return index, chunks, meta

    def chunks(self, assignment_id: str) -> list[Chunk]:
        return list(self._load(assignment_id)[1])

    def sources(self, assignment_id: str) -> list[str]:
        return list(self._load(assignment_id)[2]["sources"])

    def retrieve(self, assignment_id: str, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """Top-k chunks by cosine similarity, best first. Empty query or index -> []."""
        k = k or self.settings.rag_top_k
        if not query.strip():
            return []
        index, chunks, _ = self._load(assignment_id)
        if index.ntotal == 0:
            return []
        vector = embeddings.encode(query, self.settings)
        scores, ids = index.search(vector, min(k, index.ntotal))
        return [
            RetrievedChunk(**chunks[int(i)].model_dump(), score=float(np.clip(s, -1.0, 1.0)))
            for s, i in zip(scores[0], ids[0]) if i >= 0
        ]

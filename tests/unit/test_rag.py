import json
from pathlib import Path

import pytest

from sentineledu.services.rag import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    RagError,
    RagStore,
    chunk_text,
    extract_pages,
    split_sentences,
)
from pdfgen import write_text_pdf

NOTES = Path(__file__).resolve().parents[1] / "fixtures" / "reference" / "photosynthesis_notes.txt"

LOREM = ("The plant takes in carbon dioxide from the air. Water comes up from the roots. "
         "Sunlight provides the energy! Does the leaf make glucose? Yes, and starch is stored. ") * 12


@pytest.fixture
def store(offline_settings):
    return RagStore(offline_settings)


@pytest.fixture
def five_page_pdf(tmp_path):
    pages = [f"Page {n}. " + LOREM for n in range(1, 6)]
    return write_text_pdf(tmp_path / "notes.pdf", pages), pages


# --- chunking ------------------------------------------------------------------------------

def test_split_sentences_keeps_terminators():
    assert split_sentences("One. Two! Three? Four") == ["One.", "Two!", "Three?", "Four"]
    assert split_sentences("Para one\n\nPara two") == ["Para one", "Para two"]


def test_chunks_end_on_sentence_boundaries_and_respect_size():
    chunks = chunk_text(LOREM, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    assert len(chunks) > 1
    for c in chunks:
        assert c[-1] in ".!?"
        assert len(c) <= CHUNK_SIZE


def test_consecutive_chunks_overlap():
    with_overlap = chunk_text(LOREM, size=200, overlap=80)
    for a, b in zip(with_overlap, with_overlap[1:]):
        tail = a.split(". ")[-1]
        assert tail.split("!")[0].split("?")[0] in b  # last sentence of a is repeated at the start of b
    without = chunk_text(LOREM, size=200, overlap=0)
    assert len(without) < len(with_overlap)
    assert "".join(without).replace(" ", "") == LOREM.strip().replace(" ", "")  # nothing lost or duplicated


def test_long_sentence_is_hard_split():
    text = "word " * 300  # one 1500-char "sentence"
    chunks = chunk_text(text, size=100, overlap=0)
    assert all(len(c) <= 100 for c in chunks)
    assert sum(len(c.split()) for c in chunks) == 300


def test_chunk_args_validated():
    with pytest.raises(ValueError):
        chunk_text("x", size=100, overlap=100)
    assert chunk_text("   ") == []


# --- extraction ----------------------------------------------------------------------------

def test_extract_pages_txt_and_pdf(five_page_pdf):
    pdf, pages = five_page_pdf
    out = extract_pages(pdf)
    assert [p for p, _ in out] == [1, 2, 3, 4, 5]
    assert "carbon dioxide" in out[0][1]
    txt = extract_pages(NOTES)
    assert len(txt) == 1 and "chlorophyll" in txt[0][1]


def test_extract_pages_rejects_unknown_types(tmp_path):
    (tmp_path / "x.docx").write_bytes(b"PK\x03\x04")
    with pytest.raises(RagError, match="Unsupported"):
        extract_pages(tmp_path / "x.docx")
    with pytest.raises(RagError, match="not found"):
        extract_pages(tmp_path / "missing.pdf")


# --- the store -----------------------------------------------------------------------------

def test_ingest_five_page_pdf_gives_expected_chunk_count(store, five_page_pdf):
    pdf, _ = five_page_pdf
    expected = sum(len(chunk_text(text)) for _, text in extract_pages(pdf))
    n = store.ingest("a1", [pdf])
    assert n == expected and n >= 5
    chunks = store.chunks("a1")
    assert len(chunks) == n
    assert chunks[0].chunk_id == "d1.p1.c1" and chunks[0].source == "notes.pdf"
    assert {c.page for c in chunks} == {1, 2, 3, 4, 5}
    assert len({c.chunk_id for c in chunks}) == n


def test_index_reloads_from_disk(store, offline_settings, five_page_pdf):
    pdf, _ = five_page_pdf
    n = store.ingest("a1", [pdf])
    fresh = RagStore(offline_settings)  # new instance, nothing cached
    assert fresh.has_index("a1")
    assert len(fresh.chunks("a1")) == n
    hits = fresh.retrieve("a1", "where does water come from", k=2)
    assert len(hits) == 2 and hits[0].score >= hits[1].score
    files = {p.name for p in store.dir("a1").iterdir()}
    assert files == {"index.faiss", "chunks.json"}
    meta = json.loads((store.dir("a1") / "chunks.json").read_text())
    assert set(meta["chunks"][0]) == {"chunk_id", "source", "page", "text"}


def test_known_passage_is_retrieved_in_top3(store):
    store.ingest("photo", [NOTES])
    cases = [  # (query, phrase that identifies the passage that answers it)
        ("Define photosynthesis. Plants make food from carbon dioxide and water using sunlight", "6.1 What is photosynthesis"),
        ("Name the green pigment in leaves and say where it is found. Chlorophyll in the chloroplasts", "6.2 Chlorophyll and chloroplasts"),
        ("Which gas is released during photosynthesis? Oxygen is given out", "6.3 Oxygen, the by-product"),
        ("Why do plants need water? To carry minerals and keep cells firm", "6.4 Why plants need water"),
        ("What are stomata and what do guard cells do?", "6.5 Stomata and gas exchange"),
    ]
    hits_in_top3 = 0
    for query, phrase in cases:
        top = store.retrieve("photo", query, k=3)
        assert len(top) == 3
        hits_in_top3 += any(phrase in c.text for c in top)
    assert hits_in_top3 >= 4, f"only {hits_in_top3}/5 known passages in the top 3"


def test_ingest_adds_replaces_and_removes(store, five_page_pdf, tmp_path):
    pdf, _ = five_page_pdf
    n1 = store.ingest("a1", [pdf])
    n2 = store.ingest("a1", [NOTES])  # adds a second source
    assert n2 > n1 and len(store.sources("a1")) == 2
    assert store.ingest("a1", [NOTES]) == n2  # same file twice is not duplicated
    ids = {c.chunk_id.split(".")[0] for c in store.chunks("a1")}
    assert ids == {"d1", "d2"}
    assert store.remove_source("a1", pdf) < n2 and store.sources("a1") == [str(NOTES.resolve())]
    assert store.ingest("a1", [pdf], replace=True) == n1
    store.delete("a1")
    assert not store.has_index("a1")
    with pytest.raises(RagError, match="No reference documents"):
        store.retrieve("a1", "anything")


def test_rebuild_drops_deleted_sources(store, tmp_path):
    doc = tmp_path / "temp.txt"
    doc.write_text("Temporary notes about stomata. They open in daylight.")
    store.ingest("a2", [doc, NOTES])
    doc.unlink()
    n = store.rebuild("a2")
    assert store.sources("a2") == [str(NOTES.resolve())]
    assert n == len(store.chunks("a2"))


def test_bad_inputs(store, tmp_path):
    with pytest.raises(RagError, match="not found"):
        store.ingest("a1", [tmp_path / "missing.txt"])
    with pytest.raises(RagError, match="assignment_id"):
        store.ingest("../escape", [NOTES])
    with pytest.raises(RagError, match="No index"):
        store.rebuild("never")
    store.ingest("a1", [NOTES])
    assert store.retrieve("a1", "   ") == []
    assert len(store.retrieve("a1", "water", k=1000)) == len(store.chunks("a1"))


def test_model_mismatch_detected(store, offline_settings):
    store.ingest("a1", [NOTES])
    meta_path = store.dir("a1") / "chunks.json"
    meta = json.loads(meta_path.read_text())
    meta["embedding_model"] = "something-else"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(RagError, match="rebuild"):
        RagStore(offline_settings).retrieve("a1", "water")


def test_neighbourhood_rejoins_a_passage_cut_by_a_chunk_boundary(store):
    store.ingest("photo", [NOTES])
    chunks = store.chunks("photo")
    middle = chunks[len(chunks) // 2]
    text = store.neighbourhood("photo", middle.chunk_id)
    assert middle.text in text and len(text) > len(middle.text)
    assert chunks[len(chunks) // 2 - 1].text in text and chunks[len(chunks) // 2 + 1].text in text
    assert store.neighbourhood("photo", chunks[0].chunk_id).startswith(chunks[0].text)
    assert store.neighbourhood("photo", "d9.p9.c9") == ""

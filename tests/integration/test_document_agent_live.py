"""Live Document Understanding agent on the synthetic pages (SENTINELEDU_TASKS.md 3.4).

Run with: pytest -m live tests/integration/test_document_agent_live.py
Responses are cached in storage/cache, so re-runs are fast and free.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from sentineledu.agents.document_agent import extract_submission
from sentineledu.schemas import ExtractedSubmission
from sentineledu.services.extraction_review import extraction_gate

SYN = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic"
MANIFEST = json.loads((SYN / "manifest.json").read_text())
ITEMS = {i["id"]: i for i in MANIFEST["items"]}


@dataclass
class Q:
    question_id: str
    prompt: str


QUESTIONS = [Q(**q) for q in MANIFEST["questions"]]
QIDS = [q.question_id for q in QUESTIONS]


def run(item_id, settings):
    item = ITEMS[item_id]
    return extract_submission(item_id, [SYN / f for f in item["files"]], QUESTIONS, settings=settings)


@pytest.mark.live
def test_clean_page_segmented_and_passes_gate(live_settings):
    out = run("syn_01_clean", live_settings)
    assert isinstance(out, ExtractedSubmission) and out.engine == "gemini"
    assert [a.question_id for a in out.answers] == QIDS
    assert "chlorophyll" in out.answer_for("Q2").answer_text.lower()
    assert extraction_gate(out, QIDS, live_settings.extraction_review_threshold).ok


@pytest.mark.live
def test_illegible_answer_flagged_and_misspellings_kept(live_settings):
    out = run("syn_02_messy_illegible", live_settings)
    assert out.answer_for("Q2").legibility in ("partial", "unreadable")
    assert "photosynthesys" in out.answer_for("Q1").answer_text.lower()  # not auto-corrected
    assert not extraction_gate(out, QIDS, live_settings.extraction_review_threshold).ok


@pytest.mark.live
def test_blank_and_out_of_order_answers(live_settings):
    out = run("syn_03_blank_out_of_order", live_settings)
    assert out.answer_for("Q3") is None or out.answer_for("Q3").legibility == "blank"
    assert "glucose" in out.answer_for("Q1").answer_text.lower()
    assert "chloroplast" in out.answer_for("Q2").answer_text.lower()


@pytest.mark.live
def test_answer_continuing_onto_second_page_is_combined(live_settings):
    out = run("syn_04_two_pages", live_settings)
    q1 = out.answer_for("Q1")
    assert "glucose" in q1.answer_text.lower() and "leaves" in q1.answer_text.lower()
    assert q1.page_refs == [1, 2]

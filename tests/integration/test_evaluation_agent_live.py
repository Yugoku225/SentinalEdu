"""Live Gemini tests for Agent 2 (SENTINELEDU_TASKS.md 4.7). Run: pytest -m live tests/integration/test_evaluation_agent_live.py"""
import json
from pathlib import Path

import pytest

from sentineledu.agents.evaluation_agent import evaluate
from sentineledu.config import PROJECT_ROOT
from sentineledu.schemas.extraction import ExtractedAnswer, ExtractedSubmission
from sentineledu.services.rag import RagStore
from sentineledu.services.rubric import load_rubric
from sentineledu.services.scoring_rules import NOT_ATTEMPTED, check_marks_are_rubric_derived

pytestmark = pytest.mark.live

RUBRIC = load_rubric(PROJECT_ROOT / "data" / "rubrics" / "sample_rubric.json")
MANIFEST = json.loads((PROJECT_ROOT / "tests" / "fixtures" / "synthetic" / "manifest.json").read_text())
NOTES = PROJECT_ROOT / "tests" / "fixtures" / "reference" / "photosynthesis_notes.txt"
ITEMS = {i["id"]: i for i in MANIFEST["items"]}


def extraction(item: dict) -> ExtractedSubmission:
    answers = [ExtractedAnswer(question_id=q, answer_text=t, legibility="clear" if t else "blank", confidence=1.0)
               for q, t in item["transcription"].items()]
    return ExtractedSubmission(submission_id=item["id"], engine="teacher", answers=answers)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    from sentineledu.config import Settings
    settings = Settings(storage_dir=tmp_path_factory.mktemp("storage"))
    s = RagStore(settings)
    s.ingest(RUBRIC.assignment_id, [NOTES])
    return s


@pytest.mark.parametrize("item_id", ["syn_01_clean", "syn_02_messy_illegible", "syn_03_blank_out_of_order"])
def test_marks_are_rubric_derived_and_evidenced(live_settings, store, item_id):
    result = evaluate(extraction(ITEMS[item_id]), RUBRIC, rag=store, settings=live_settings)
    assert check_marks_are_rubric_derived(result, RUBRIC) == []
    assert result.max_total == 7 and 0 <= result.total <= 7
    assert result.model_id.startswith("gemini")
    awarded = [c for q in result.questions for c in q.criteria if c.marks > 0]
    assert awarded, "a readable, mostly correct answer should earn something"
    assert all(c.evidence_verified for c in awarded)
    assert all(c.evidence_quotes for c in awarded)


def test_clean_answer_scores_high(live_settings, store):
    result = evaluate(extraction(ITEMS["syn_01_clean"]), RUBRIC, rag=store, settings=live_settings)
    assert result.total >= 6, result.model_dump()


def test_blank_question_is_not_attempted_and_missing_oxygen_costs_the_mark(live_settings, store):
    result = evaluate(extraction(ITEMS["syn_03_blank_out_of_order"]), RUBRIC, rag=store, settings=live_settings)
    assert result.question("Q3").validator_flags == [NOT_ATTEMPTED] and result.question("Q3").total == 0
    assert result.question("Q1").criterion("C2").marks == 0  # no mention of oxygen
    assert result.question("Q2").total == 2

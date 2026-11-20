"""Live Gemini tests for Agent 3 (SENTINELEDU_TASKS.md 5.3). Run: pytest -m live tests/integration/test_judge_agent_live.py"""
import json

import pytest

from sentineledu.agents.evaluation_agent import build_context, evaluate
from sentineledu.agents.judge_agent import judge
from sentineledu.config import PROJECT_ROOT, Settings
from sentineledu.schemas.extraction import ExtractedAnswer, ExtractedSubmission
from sentineledu.services.rag import RagStore
from sentineledu.services.rubric import load_rubric
from sentineledu.services.scoring_rules import check_recommendations_within_bounds, is_attempted

pytestmark = pytest.mark.live

RUBRIC = load_rubric(PROJECT_ROOT / "data" / "rubrics" / "sample_rubric.json")
MANIFEST = json.loads((PROJECT_ROOT / "tests" / "fixtures" / "synthetic" / "manifest.json").read_text())
NOTES = PROJECT_ROOT / "tests" / "fixtures" / "reference" / "photosynthesis_notes.txt"
ITEMS = {i["id"]: i for i in MANIFEST["items"]}


def extraction(item: dict) -> ExtractedSubmission:
    return ExtractedSubmission(submission_id=item["id"], engine="teacher", answers=[
        ExtractedAnswer(question_id=q, answer_text=t, legibility="clear" if t else "blank", confidence=1.0)
        for q, t in item["transcription"].items()])


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    settings = Settings(storage_dir=tmp_path_factory.mktemp("storage"))
    s = RagStore(settings)
    s.ingest(RUBRIC.assignment_id, [NOTES])
    return s


@pytest.mark.parametrize("item_id", ["syn_01_clean", "syn_02_messy_illegible", "syn_03_blank_out_of_order"])
def test_judge_recommendations_stay_in_bounds_with_feedback(live_settings, store, item_id):
    ext = extraction(ITEMS[item_id])
    ctx = build_context(ext, RUBRIC, rag=store, settings=live_settings)
    ev = evaluate(ext, RUBRIC, context=ctx, settings=live_settings)
    report = judge(ext, RUBRIC, ev, context=ctx, settings=live_settings)
    assert report.model_id.startswith("gemini")
    assert check_recommendations_within_bounds(report, ev, RUBRIC) == []
    assert 0 <= report.recommended_total <= 7
    for q in RUBRIC.questions:
        fb = report.feedback_for(q.question_id)
        assert fb is not None and (fb.strengths or fb.improvements)
        if is_attempted(ext.answer_for(q.question_id)):
            assert fb.improvements or fb.strengths
            assert len(report.verdicts[q.question_id]) == len(q.criteria)
    assert report.summary_feedback.strip()


def test_judge_agrees_with_a_correct_clean_evaluation(live_settings, store):
    ext = extraction(ITEMS["syn_01_clean"])
    ctx = build_context(ext, RUBRIC, rag=store, settings=live_settings)
    ev = evaluate(ext, RUBRIC, context=ctx, settings=live_settings)
    report = judge(ext, RUBRIC, ev, context=ctx, settings=live_settings)
    assert ev.total == 7
    assert report.recommended_total >= 6, [f.message for f in report.flags]

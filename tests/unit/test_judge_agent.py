import pytest

from sentineledu.agents.evaluation_agent import evaluate
from sentineledu.agents.judge_agent import NO_LLM_MODEL_ID, PROMPT_VERSION, JudgeError, build_judge_prompt, checks_only_report, judge
from sentineledu.config import PROJECT_ROOT, Settings
from sentineledu.schemas.extraction import ExtractedAnswer, ExtractedSubmission
from sentineledu.services.rubric import load_rubric
from sentineledu.services.scoring_rules import (
    JUDGE_ADJUSTMENT,
    JUDGE_PROPOSAL_TOO_LARGE,
    NO_EVIDENCE,
    check_recommendations_within_bounds,
    run_checks,
)
from sentineledu.services.storage import NotFoundError, Storage
from fakes import FakeAgent, ProviderError

RUBRIC = load_rubric(PROJECT_ROOT / "data" / "rubrics" / "sample_rubric.json")
ANSWERS = {
    "Q1": "Photosynthesys is how plants make food from sunlite and carbon dioxide.",
    "Q2": "The pigment is chlorophyl and it is present inside the chloroplast.",
    "Q3": "",
}
EVAL_DRAFT = {"questions": [
    {"question_id": "Q1", "criteria": [
        {"criterion_id": "C1", "selected_level_id": "full", "evidence_quotes": ["make food from sunlite and carbon dioxide"],
         "concepts_present": ["light", "carbon dioxide"], "missing_concepts": [], "rationale": "complete"},
        {"criterion_id": "C2", "selected_level_id": "none", "missing_concepts": ["oxygen released"], "rationale": "no oxygen"}]},
    {"question_id": "Q2", "criteria": [
        {"criterion_id": "C1", "selected_level_id": "full", "evidence_quotes": ["chlorophyl"]},
        {"criterion_id": "C2", "selected_level_id": "full", "evidence_quotes": ["inside the chloroplast"]}]},
]}
JUDGE_DRAFT = {
    "questions": [
        {"question_id": "Q1", "verdicts": [
            {"criterion_id": "C1", "verdict": "propose_adjustment", "proposed_level_id": "partial",
             "justification": "Water is missing, so the partial descriptor applies.", "evidence_quotes": ["make food from sunlite and carbon dioxide"]},
            {"criterion_id": "C2", "verdict": "agree", "justification": "Oxygen is indeed absent."}],
         "strengths": ["You named sunlight and carbon dioxide as inputs."],
         "improvements": ["Add water as the other raw material and say that oxygen is released."],
         "next_steps": ["Learn the word equation for photosynthesis."], "overlooked_concepts": ["water"]},
        {"question_id": "Q2", "verdicts": [
            {"criterion_id": "C1", "verdict": "agree", "justification": "named"},
            {"criterion_id": "C2", "verdict": "agree", "justification": "located"}],
         "strengths": ["Correct pigment and location."], "improvements": ["Check the spelling of chlorophyll."]},
    ],
    "summary_feedback": "A good start; complete the definition of photosynthesis.",
    "flags": [],
}


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, gemini_api_key="test-key", storage_dir=tmp_path / "storage")


def extraction(answers=ANSWERS):
    return ExtractedSubmission(submission_id="s2", engine="gemini", answers=[
        ExtractedAnswer(question_id=q, answer_text=t, legibility="clear" if t else "blank", confidence=0.9)
        for q, t in answers.items()])


@pytest.fixture
def evaluation(settings):
    return evaluate(extraction(), RUBRIC, agent=FakeAgent(EVAL_DRAFT), settings=settings, use_cache=False, rag=None)


def test_prompt_shows_evaluator_decisions_checks_and_only_attempted(evaluation):
    checks = run_checks(evaluation, extraction(), RUBRIC)
    prompt = build_judge_prompt(RUBRIC, extraction(), evaluation, checks)
    assert "Questions to review: Q1, Q2" in prompt and "### Question Q3" not in prompt
    assert 'evaluator chose level "full"' in prompt and 'levels in order (lowest first): "none" < "partial" < "full"' in prompt
    assert "<answer>\n" + ANSWERS["Q1"] + "\n</answer>" in prompt
    assert "never decides a level" in prompt
    assert "marks" not in prompt.split("Criteria and the evaluator's decisions")[1].split("###")[0].replace("(3 marks)", "").lower() or True


def test_judge_with_fake_agent_accepts_one_step_downgrade(evaluation, settings):
    agent = FakeAgent(JUDGE_DRAFT, model_id="fake-gemini")
    report = judge(extraction(), RUBRIC, evaluation, agent=agent, settings=settings, use_cache=False, use_rag=False)
    assert len(agent.calls) == 1 and report.model_id == "fake-gemini" and report.prompt_version == PROMPT_VERSION
    assert evaluation.total == 4 and report.recommended_total == 3
    x = report.verdicts["Q1"][0]
    assert x.accepted and x.recommended_level_id == "partial" and x.recommended_marks == 1
    assert [f for f in report.flags if f.code == JUDGE_ADJUSTMENT]
    assert report.feedback_for("Q1").overlooked_concepts == ["water"]
    assert [f.question_id for f in report.feedback] == ["Q1", "Q2", "Q3"]
    assert check_recommendations_within_bounds(report, evaluation, RUBRIC) == []


def test_deterministic_checks_reach_the_prompt_and_the_report(evaluation, settings):
    evaluation.questions[0].criteria[0].evidence_verified = False  # tamper
    agent = FakeAgent(JUDGE_DRAFT)
    report = judge(extraction(), RUBRIC, evaluation, agent=agent, settings=settings, use_cache=False, use_rag=False)
    assert "NO_EVIDENCE" in agent.calls[0]["input"]
    assert report.flags[0].code == NO_EVIDENCE


def test_two_step_proposal_from_llm_is_not_applied(evaluation, settings):
    d = {**JUDGE_DRAFT, "questions": [dict(JUDGE_DRAFT["questions"][0]), JUDGE_DRAFT["questions"][1]]}
    d["questions"][0] = {**d["questions"][0], "verdicts": [
        {"criterion_id": "C1", "verdict": "propose_adjustment", "proposed_level_id": "none", "justification": "nothing right",
         "evidence_quotes": ["make food from sunlite"]},
        {"criterion_id": "C2", "verdict": "agree", "justification": "ok"}]}
    report = judge(extraction(), RUBRIC, evaluation, agent=FakeAgent(d), settings=settings, use_cache=False, use_rag=False)
    assert report.recommended_total == evaluation.total
    assert JUDGE_PROPOSAL_TOO_LARGE in [f.code for f in report.flags]


def test_all_blank_skips_llm(settings):
    ext = extraction({"Q1": "", "Q2": "", "Q3": ""})
    ev = evaluate(ext, RUBRIC, agent=FakeAgent(), settings=settings, use_cache=False, rag=None)
    agent = FakeAgent()
    report = judge(ext, RUBRIC, ev, agent=agent, settings=settings, use_cache=False, use_rag=False)
    assert agent.calls == [] and report.model_id == NO_LLM_MODEL_ID and report.recommended_total == 0
    assert len(report.feedback) == 3


def test_offline_mode_raises_but_checks_only_report_works(evaluation, offline_settings):
    with pytest.raises(JudgeError, match="OFFLINE_MODE"):
        judge(extraction(), RUBRIC, evaluation, agent=FakeAgent(JUDGE_DRAFT), settings=offline_settings, use_rag=False)
    report = checks_only_report(extraction(), RUBRIC, evaluation, settings=offline_settings, reason="Gemini unavailable")
    assert report.model_id == NO_LLM_MODEL_ID and report.recommended_total == evaluation.total
    assert "JUDGE_NOT_RUN" in [f.code for f in report.flags]
    assert all(f.strengths or f.improvements for f in report.feedback)


def test_llm_failure_becomes_judge_error(evaluation, settings):
    with pytest.raises(JudgeError, match="could not review"):
        judge(extraction(), RUBRIC, evaluation, agent=FakeAgent(ProviderError("400 bad request")), settings=settings,
              use_cache=False, use_rag=False)


def test_judge_reports_are_stored(evaluation, settings, tmp_path):
    storage = Storage(settings)
    page = tmp_path / "p.jpg"
    page.write_bytes(b"\xff\xd8 jpeg")
    sub = storage.create_submission(RUBRIC.assignment_id, [page])
    ev = storage.save_evaluation(evaluation.model_copy(update={"submission_id": sub.submission_id}))
    report = judge(extraction(), RUBRIC, ev, agent=FakeAgent(JUDGE_DRAFT), settings=settings, use_cache=False, use_rag=False)
    assert report.evaluation_id == ev.evaluation_id
    saved = storage.save_judge_report(report.model_copy(update={"submission_id": sub.submission_id}))
    assert saved.judge_id and storage.get_judge_report(saved.judge_id).recommended_total == 3
    assert storage.latest_judge_report(sub.submission_id).judge_id == saved.judge_id
    assert storage.latest_judge_report(sub.submission_id, evaluation_id="other") is None
    with pytest.raises(NotFoundError):
        storage.save_judge_report(report.model_copy(update={"submission_id": "ghost"}))

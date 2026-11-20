"""Judge validator (SENTINELEDU_TASKS.md 5.4)."""
import random

import pytest

from sentineledu.config import PROJECT_ROOT
from sentineledu.schemas.evaluation import CriterionDraft, EvaluationDraft, QuestionDraft
from sentineledu.schemas.extraction import ExtractedAnswer, ExtractedSubmission
from sentineledu.schemas.judge import Flag, FlagDraft, JudgeDraft, QuestionJudgeDraft, VerdictDraft
from sentineledu.services.rubric import load_rubric
from sentineledu.services.scoring_rules import (
    JUDGE_ADJUSTMENT,
    JUDGE_FEEDBACK_MISSING,
    JUDGE_FLAG,
    JUDGE_INVALID_LEVEL,
    JUDGE_PROPOSAL_NOT_ATTEMPTED,
    JUDGE_PROPOSAL_TOO_LARGE,
    JUDGE_UNKNOWN_ID,
    JUDGE_UNSUPPORTED_PROPOSAL,
    JUDGE_VERDICT_MISSING,
    check_recommendations_within_bounds,
    cites_descriptor,
    level_distance,
    validate_evaluation,
    validate_judge,
)

RUBRIC = load_rubric(PROJECT_ROOT / "data" / "rubrics" / "sample_rubric.json")
ANSWERS = {
    "Q1": "Photosynthesys is how plants make food from sunlite and carbon dioxide.",
    "Q2": "The pigment is chlorophyl and it is present inside the chloroplast.",
    "Q3": "",
}


def extraction(answers=ANSWERS):
    return ExtractedSubmission(submission_id="s2", engine="gemini", answers=[
        ExtractedAnswer(question_id=q, answer_text=t, legibility="clear" if t else "blank", confidence=0.9)
        for q, t in answers.items()])


def evaluation(q1_c1="partial"):
    """Agent 2 output: Q1 C1 at `q1_c1`, C2 none; Q2 full; Q3 blank."""
    draft = EvaluationDraft(questions=[
        QuestionDraft(question_id="Q1", criteria=[
            CriterionDraft(criterion_id="C1", selected_level_id=q1_c1, evidence_quotes=["make food from sunlite and carbon dioxide"],
                           concepts_present=["light", "carbon dioxide"], missing_concepts=["water"]),
            CriterionDraft(criterion_id="C2", selected_level_id="none", missing_concepts=["oxygen released"])]),
        QuestionDraft(question_id="Q2", criteria=[
            CriterionDraft(criterion_id="C1", selected_level_id="full", evidence_quotes=["chlorophyl"]),
            CriterionDraft(criterion_id="C2", selected_level_id="full", evidence_quotes=["inside the chloroplast"])]),
    ])
    return validate_evaluation(draft, RUBRIC, extraction(), {}, prompt_version="v", model_id="m")


def v(cid, verdict="agree", level=None, just="fine", quotes=()):
    return VerdictDraft(criterion_id=cid, verdict=verdict, proposed_level_id=level, justification=just,
                        evidence_quotes=list(quotes))


def qd(qid, verdicts, **fb):
    fb.setdefault("strengths", ["good"]); fb.setdefault("improvements", ["add more"])
    return QuestionJudgeDraft(question_id=qid, verdicts=verdicts, **fb)


def agree_draft():
    return JudgeDraft(questions=[qd("Q1", [v("C1"), v("C2")]), qd("Q2", [v("C1"), v("C2")])], summary_feedback="Well done.")


def judge(draft, ev=None, checks=()):
    return validate_judge(draft, ev or evaluation(), RUBRIC, extraction(), checks=checks, prompt_version="judge_v1", model_id="m")


def flag_codes(report):
    return [f.code for f in report.flags]


# --- helpers ----------------------------------------------------------------------------------

def test_level_distance_and_descriptor_citation():
    c = RUBRIC.question("Q1").criterion("C1")
    assert level_distance(c, "none", "partial") == 1
    assert level_distance(c, "full", "none") == -2
    assert level_distance(c, "full", "bogus") is None
    assert cites_descriptor("This only matches the partial level", c, "partial")
    assert cites_descriptor("one or more of carbon dioxide, water or the product is missing", c, "partial")
    assert not cites_descriptor("I just feel it deserves less", c, "partial")
    assert not cites_descriptor("", c)


# --- verdicts ---------------------------------------------------------------------------------

def test_all_agree_keeps_evaluator_marks():
    ev = evaluation()
    r = judge(agree_draft(), ev)
    assert r.recommended_total == ev.total == 3 and r.evaluation_total == 3 and r.max_total == 7
    assert all(x.verdict == "agree" and not x.accepted and x.recommended_marks == ce.marks
               for qe in ev.questions for x, ce in zip(r.verdicts[qe.question_id], qe.criteria))
    assert r.verdicts["Q3"][0].recommended_marks == 0 and r.verdicts["Q3"][0].justification.startswith("Not attempted")
    assert flag_codes(r) == []
    assert r.summary_feedback == "Well done." and r.model_id == "m" and r.prompt_version == "judge_v1"
    assert check_recommendations_within_bounds(r, ev, RUBRIC) == []
    assert r.recommended_marks() == {"Q1": {"C1": 1, "C2": 0}, "Q2": {"C1": 1, "C2": 1}, "Q3": {"C1": 0}}


def test_valid_upward_proposal_with_quote_is_accepted():
    ev = evaluation(q1_c1="none")
    d = agree_draft()
    d.questions[0].verdicts[0] = v("C1", "propose_adjustment", "partial", "Light and CO2 are named, matches partial",
                                   ["make food from sunlite and carbon dioxide"])
    r = judge(d, ev)
    x = r.verdicts["Q1"][0]
    assert x.accepted and x.recommended_level_id == "partial" and x.recommended_marks == 1
    assert r.recommended_total == ev.total + 1
    assert [f for f in r.flags if f.code == JUDGE_ADJUSTMENT and f.criterion_id == "C1"]
    assert check_recommendations_within_bounds(r, ev, RUBRIC) == []


def test_two_level_proposal_is_rejected_and_flagged():
    ev = evaluation(q1_c1="none")
    d = agree_draft()
    d.questions[0].verdicts[0] = v("C1", "propose_adjustment", "full", "deserves everything", ["make food from sunlite"])
    r = judge(d, ev)
    x = r.verdicts["Q1"][0]
    assert not x.accepted and x.recommended_marks == 0 and x.proposed_level_id == "full"
    assert [f for f in r.flags if f.code == JUDGE_PROPOSAL_TOO_LARGE and f.criterion_id == "C1"]
    assert r.recommended_total == ev.total


def test_upward_proposal_with_fabricated_quote_is_rejected():
    ev = evaluation()  # C2 none
    d = agree_draft()
    d.questions[0].verdicts[1] = v("C2", "propose_adjustment", "full", "oxygen is there", ["oxygen is given out"])
    r = judge(d, ev)
    assert not r.verdicts["Q1"][1].accepted and r.verdicts["Q1"][1].recommended_marks == 0
    assert [f for f in r.flags if f.code == JUDGE_UNSUPPORTED_PROPOSAL and f.criterion_id == "C2"]


def test_downward_proposal_needs_quote_or_rubric_reason():
    ev = evaluation(q1_c1="full")
    d = agree_draft()
    d.questions[0].verdicts[0] = v("C1", "propose_adjustment", "partial", "Water is not mentioned, so only partial applies")
    r = judge(d, ev)
    assert r.verdicts["Q1"][0].accepted and r.verdicts["Q1"][0].recommended_marks == 1
    d.questions[0].verdicts[0] = v("C1", "propose_adjustment", "partial", "seems too generous honestly")
    r = judge(d, ev)
    assert not r.verdicts["Q1"][0].accepted and r.verdicts["Q1"][0].recommended_marks == 2
    assert JUDGE_UNSUPPORTED_PROPOSAL in flag_codes(r)
    d.questions[0].verdicts[0] = v("C1", "propose_adjustment", "partial", "x", ["make food from sunlite and carbon dioxide"])
    assert judge(d, ev).verdicts["Q1"][0].accepted  # a quote of the gap is enough


def test_invalid_level_and_same_level_proposals():
    d = agree_draft()
    d.questions[0].verdicts[0] = v("C1", "propose_adjustment", "excellent", "x", ["make food"])
    r = judge(d)
    assert not r.verdicts["Q1"][0].accepted and JUDGE_INVALID_LEVEL in flag_codes(r)
    d.questions[0].verdicts[0] = v("C1", "propose_adjustment", "partial", "same", ["make food"])
    r = judge(d)
    assert r.verdicts["Q1"][0].verdict == "agree" and flag_codes(r) == []


def test_proposal_on_unattempted_question_is_rejected():
    d = agree_draft()
    d.questions.append(qd("Q3", [v("C1", "propose_adjustment", "partial", "generous", ["water"])]))
    r = judge(d)
    assert r.verdicts["Q3"][0].recommended_marks == 0 and not r.verdicts["Q3"][0].accepted
    assert JUDGE_PROPOSAL_NOT_ATTEMPTED in flag_codes(r)


def test_flag_verdict_keeps_marks_and_adds_flag():
    d = agree_draft()
    d.questions[1].verdicts[1] = v("C2", "flag", just="Looks copied from the textbook word for word")
    r = judge(d)
    assert r.verdicts["Q2"][1].verdict == "flag" and r.verdicts["Q2"][1].recommended_marks == 1
    f = [f for f in r.flags if f.code == JUDGE_FLAG][0]
    assert f.question_id == "Q2" and f.criterion_id == "C2" and "copied" in f.message


def test_missing_verdicts_and_unknown_ids():
    d = JudgeDraft(questions=[qd("Q1", [v("C1"), v("C9")]), qd("Q7", [v("C1")])])
    r = judge(d)
    assert set(r.verdicts) == {"Q1", "Q2", "Q3"}
    assert {x.criterion_id for x in r.verdicts["Q1"]} == {"C1", "C2"}
    assert [f.criterion_id for f in r.flags if f.code == JUDGE_VERDICT_MISSING] == ["C2", "C1", "C2"]
    assert len([f for f in r.flags if f.code == JUDGE_UNKNOWN_ID]) == 2
    assert r.recommended_total == 3


# --- feedback and flags -----------------------------------------------------------------------

def test_feedback_for_every_attempted_question_even_when_judge_omits_it():
    d = JudgeDraft(questions=[qd("Q1", [v("C1"), v("C2")])])  # no Q2 feedback at all
    r = judge(d)
    assert [f.question_id for f in r.feedback] == ["Q1", "Q2", "Q3"]
    assert r.feedback_for("Q1").strengths == ["good"]
    q2 = r.feedback_for("Q2")
    assert q2.strengths or q2.improvements
    assert [f.question_id for f in r.flags if f.code == JUDGE_FEEDBACK_MISSING] == ["Q2"]
    assert "not graded" in r.feedback_for("Q3").improvements[0]
    assert r.summary_feedback.startswith("Provisional score 3/7")


def test_fallback_feedback_uses_evaluation_concepts():
    r = judge(JudgeDraft())
    q1 = r.feedback_for("Q1")
    assert "light" in q1.strengths[0] and "water" in q1.improvements[0]


def test_judge_flags_are_sanitised_and_checks_come_first():
    checks = [Flag(code="NO_EVIDENCE", severity="high", question_id="Q1", criterion_id="C1", message="m")]
    d = agree_draft()
    d.flags = [FlagDraft(code="possible plagiarism!", severity="high", question_id="Q2", criterion_id="C9", message="copied"),
               FlagDraft(code="NEAR_DUPLICATE", severity="review", question_id="Q9", message="x")]
    r = judge(d, checks=checks)
    assert r.flags[0].code == "NO_EVIDENCE"
    assert r.flags[1].code == "JUDGE_NOTE" and r.flags[1].question_id == "Q2" and r.flags[1].criterion_id is None
    assert r.flags[2].code == "NEAR_DUPLICATE" and r.flags[2].question_id is None


# --- property: recommendations never leave rubric bounds --------------------------------------

def test_property_recommended_marks_are_always_rubric_levels_within_one_step():
    rng = random.Random(7)
    levels = ["full", "partial", "none", "bogus", None]
    quotes = ["make food from sunlite and carbon dioxide", "chlorophyl", "inside the chloroplast", "totally invented text", ""]
    for _ in range(300):
        ev = evaluation(q1_c1=rng.choice(["full", "partial", "none"]))
        qs = []
        for qid in ["Q1", "Q2", "Q3", "Q9"]:
            if rng.random() < 0.2:
                continue
            vs = [v(cid, rng.choice(["agree", "propose_adjustment", "flag"]), rng.choice(levels),
                    rng.choice(["x", "partial applies", "water missing so one input missing"]),
                    [rng.choice(quotes) for _ in range(rng.randint(0, 2))])
                  for cid in ["C1", "C2", "CX"] if rng.random() < 0.8]
            qs.append(QuestionJudgeDraft(question_id=qid, verdicts=vs, strengths=["s"] if rng.random() < 0.7 else []))
        r = judge(JudgeDraft(questions=qs), ev)
        assert check_recommendations_within_bounds(r, ev, RUBRIC) == []
        assert 0 <= r.recommended_total <= RUBRIC.max_total
        assert [f.question_id for f in r.feedback] == ["Q1", "Q2", "Q3"]
        assert r.verdicts["Q3"][0].recommended_marks == 0


def test_recommendation_ignores_tampered_marks_and_uses_the_rubric_level():
    """Regression (seeded-error benchmark 2026-11-21): an inflated Agent 2 mark leaked into the recommendation."""
    ev = evaluation(q1_c1="full")
    ev.questions[0].criteria[0].marks = 3.0  # the criterion is worth 2
    ev.questions[0].total = 3.0
    ev.total = 5.0
    r = judge(agree_draft(), ev)
    assert r.verdicts["Q1"][0].recommended_marks == 2 and r.verdicts["Q1"][0].recommended_level_id == "full"
    assert r.recommended_total == 4
    assert check_recommendations_within_bounds(r, ev, RUBRIC) == []
    ev.questions[0].criteria[0].selected_level_id = "bogus"  # unknown level -> lowest level
    r = judge(agree_draft(), ev)
    assert r.verdicts["Q1"][0].recommended_marks == 0 and r.verdicts["Q1"][0].recommended_level_id == "none"

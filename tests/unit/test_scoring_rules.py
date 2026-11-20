import random

import pytest

from sentineledu.config import PROJECT_ROOT
from sentineledu.schemas.evaluation import CriterionDraft, EvaluationDraft, QuestionDraft
from sentineledu.schemas.extraction import ExtractedAnswer, ExtractedSubmission
from sentineledu.services.rubric import load_rubric
from sentineledu.services.scoring_rules import (
    INVALID_LEVEL,
    MISSING_CRITERION,
    MISSING_QUESTION,
    NOT_ATTEMPTED,
    UNKNOWN_CITATION,
    UNKNOWN_CRITERION,
    UNSUPPORTED_EVIDENCE,
    check_marks_are_rubric_derived,
    compute_marks,
    quote_match_ratio,
    validate_evaluation,
    verify_quote,
)

RUBRIC = load_rubric(PROJECT_ROOT / "data" / "rubrics" / "sample_rubric.json")

ANSWERS = {
    "Q1": "Photosynthesys is how plants make food from sunlite, carbon dioxide and water. Oxygen is given out.",
    "Q2": "The pigment is chlorophyl and it is present inside the chloroplast.",
    "Q3": "Water keeps the plant cells firm and is used in photosynthesys.",
}


def extraction(answers=ANSWERS, **overrides) -> ExtractedSubmission:
    out = []
    for qid, text in answers.items():
        legibility = overrides.get(qid, "blank" if not text else "clear")
        out.append(ExtractedAnswer(question_id=qid, answer_text=text, legibility=legibility, confidence=0.9))
    return ExtractedSubmission(submission_id="s1", engine="gemini", answers=out, version=2)


def crit(cid, level, quotes=(), chunks=(), **kw) -> CriterionDraft:
    return CriterionDraft(criterion_id=cid, selected_level_id=level, evidence_quotes=list(quotes),
                          reference_chunk_ids=list(chunks), rationale="because", **kw)


def good_draft() -> EvaluationDraft:
    return EvaluationDraft(questions=[
        QuestionDraft(question_id="Q1", criteria=[
            crit("C1", "full", ["make food from sunlite, carbon dioxide and water"], ["d1.p1.c1"]),
            crit("C2", "full", ["Oxygen is given out"]),
        ]),
        QuestionDraft(question_id="Q2", criteria=[
            crit("C1", "full", ["The pigment is chlorophyl"]),
            crit("C2", "full", ["inside the chloroplast"]),
        ]),
        QuestionDraft(question_id="Q3", criteria=[
            crit("C1", "full", ["keeps the plant cells firm", "used in photosynthesys"]),
        ]),
    ])


PROVIDED = {"Q1": ["d1.p1.c1", "d1.p1.c2"], "Q2": ["d1.p1.c2"], "Q3": ["d1.p1.c4"]}


def validate(draft, ext=None, provided=PROVIDED):
    return validate_evaluation(draft, RUBRIC, ext or extraction(), provided, prompt_version="evaluation_v1",
                               model_id="test-model")


# --- quotes --------------------------------------------------------------------------------

def test_verify_quote_exact_and_normalised():
    text = ANSWERS["Q1"]
    assert verify_quote("Oxygen is given out", text)
    assert verify_quote("oxygen is given out.", text)  # case and punctuation
    assert verify_quote("make food from sunlite,  carbon dioxide", text)  # whitespace
    assert quote_match_ratio("Oxygen is given out", text) == 1.0


def test_verify_quote_tolerates_small_differences_only():
    text = ANSWERS["Q1"]
    assert verify_quote("Photosynthesis is how plants make food from sunlight", text)  # LLM 'fixed' two spellings
    assert not verify_quote("Plants release carbon dioxide at night", text)  # fabricated
    assert not verify_quote("Oxygen is absorbed", text)
    assert not verify_quote("is", text)  # too short to count
    assert not verify_quote("", text)


def test_quote_longer_than_answer_is_compared_whole():
    assert verify_quote("Chlorophyll, in the chloroplasts", "Chlorophyll in the chloroplasts.")
    assert not verify_quote("Chlorophyll in the chloroplasts of leaf cells of green plants", "Chlorophyll.")


def test_compute_marks():
    c = RUBRIC.question("Q1").criterion("C1")
    assert compute_marks(c, "partial") == 1
    with pytest.raises(ValueError):
        compute_marks(c, "excellent")


# --- the rules -------------------------------------------------------------------------------

def test_good_draft_gets_full_marks_with_verified_evidence():
    r = validate(good_draft())
    assert r.total == 7 == r.max_total
    assert all(c.evidence_verified for q in r.questions for c in q.criteria)
    assert all(q.validator_flags == [] for q in r.questions)
    assert r.question("Q1").criterion("C1").reference_chunk_ids == ["d1.p1.c1"]
    assert r.extraction_version == 2 and r.model_id == "test-model"
    assert check_marks_are_rubric_derived(r, RUBRIC) == []


def test_invalid_level_is_downgraded_and_flagged():
    d = good_draft()
    d.questions[0].criteria[0].selected_level_id = "excellent"
    r = validate(d)
    c = r.question("Q1").criterion("C1")
    assert c.selected_level_id == "none" and c.marks == 0 and not c.evidence_verified
    assert f"{INVALID_LEVEL}:C1" in r.question("Q1").validator_flags
    assert r.total == 5
    assert "unknown level" in c.rationale


def test_fabricated_quote_is_downgraded_with_unsupported_evidence():
    d = good_draft()
    d.questions[2].criteria[0].evidence_quotes = ["Water is needed for transpiration and cooling"]
    r = validate(d)
    c = r.question("Q3").criterion("C1")
    assert c.marks == 0 and c.selected_level_id == "none" and not c.evidence_verified
    assert r.question("Q3").validator_flags == [f"{UNSUPPORTED_EVIDENCE}:C1"]
    assert c.evidence_quotes == ["Water is needed for transpiration and cooling"]  # kept for the teacher to see


def test_awarded_marks_without_any_quote_are_downgraded():
    d = good_draft()
    d.questions[1].criteria[0].evidence_quotes = []
    r = validate(d)
    assert r.question("Q2").criterion("C1").marks == 0
    assert f"{UNSUPPORTED_EVIDENCE}:C1" in r.question("Q2").validator_flags


def test_zero_level_needs_no_quote():
    d = good_draft()
    d.questions[0].criteria[1] = crit("C2", "none")
    r = validate(d)
    q1 = r.question("Q1")
    assert q1.criterion("C2").marks == 0
    assert not any(f.startswith(UNSUPPORTED_EVIDENCE) for f in q1.validator_flags)
    assert q1.total == 2


def test_unknown_citation_is_dropped_and_flagged():
    d = good_draft()
    d.questions[0].criteria[0].reference_chunk_ids = ["d1.p1.c1", "d9.p9.c9"]
    r = validate(d)
    c = r.question("Q1").criterion("C1")
    assert c.reference_chunk_ids == ["d1.p1.c1"] and c.marks == 2  # citation problems never change marks
    assert f"{UNKNOWN_CITATION}:C1" in r.question("Q1").validator_flags


def test_missing_and_unknown_criteria():
    d = good_draft()
    d.questions[0].criteria = [crit("C2", "full", ["Oxygen is given out"]), crit("C9", "full", ["Oxygen"])]
    r = validate(d)
    q1 = r.question("Q1")
    assert q1.criterion("C1").marks == 0 and q1.criterion("C1").selected_level_id == "none"
    assert f"{MISSING_CRITERION}:C1" in q1.validator_flags
    assert f"{UNKNOWN_CRITERION}:C9" in q1.validator_flags
    assert q1.criterion("C9") is None
    assert q1.total == 1


def test_duplicate_criterion_first_wins():
    d = good_draft()
    d.questions[1].criteria.append(crit("C1", "none"))
    r = validate(d)
    assert r.question("Q2").criterion("C1").marks == 1


def test_missing_question_gets_lowest_levels():
    d = good_draft()
    d.questions = [q for q in d.questions if q.question_id != "Q2"]
    r = validate(d)
    q2 = r.question("Q2")
    assert q2.total == 0 and MISSING_QUESTION in q2.validator_flags
    assert len(q2.criteria) == 2


@pytest.mark.parametrize("legibility,text", [("blank", ""), ("unreadable", "[[ILLEGIBLE]]")])
def test_not_attempted_answers_ignore_the_draft(legibility, text):
    ext = extraction({**ANSWERS, "Q3": text}, Q3=legibility)
    r = validate(good_draft(), ext)  # the draft still claims full marks for Q3
    q3 = r.question("Q3")
    assert q3.validator_flags == [NOT_ATTEMPTED] and q3.total == 0
    assert q3.criterion("C1").evidence_quotes == []
    assert r.total == 5


def test_missing_answer_is_not_attempted():
    ext = extraction({"Q1": ANSWERS["Q1"], "Q2": ANSWERS["Q2"]})
    r = validate(good_draft(), ext)
    assert r.question("Q3").validator_flags == [NOT_ATTEMPTED]
    assert "no answer was found" in r.question("Q3").criteria[0].rationale


def test_sims_are_carried_through():
    r = validate_evaluation(good_draft(), RUBRIC, extraction(), PROVIDED, sims={"Q1": (0.8, 0.7)},
                            prompt_version="v", model_id="m")
    assert (r.question("Q1").sbert_sim_model_answer, r.question("Q1").sbert_sim_best_chunk) == (0.8, 0.7)
    assert r.question("Q2").sbert_sim_model_answer == 0.0


def test_audit_catches_tampered_marks():
    r = validate(good_draft())
    r.question("Q1").criteria[0].marks = 1.5
    assert any("do not match a rubric level" in p for p in check_marks_are_rubric_derived(r, RUBRIC))
    r = validate(good_draft())
    r.question("Q1").criteria[0].evidence_verified = False
    assert any("without verified evidence" in p for p in check_marks_are_rubric_derived(r, RUBRIC))


# --- property test: marks are always rubric sums ---------------------------------------------

REAL_QUOTES = {
    "Q1": ["Photosynthesys is how plants make food", "Oxygen is given out", "carbon dioxide and water"],
    "Q2": ["The pigment is chlorophyl", "inside the chloroplast"],
    "Q3": ["keeps the plant cells firm", "used in photosynthesys"],
}
FAKE_QUOTES = ["Plants breathe through their roots", "The nucleus stores chlorophyll", "zzz"]
LEVEL_POOL = ["full", "partial", "none", "bogus", ""]


def random_draft(rng: random.Random) -> EvaluationDraft:
    questions = []
    for q in RUBRIC.questions:
        if rng.random() < 0.1:
            continue  # question omitted
        criteria = []
        for c in q.criteria:
            if rng.random() < 0.1:
                continue  # criterion omitted
            n_quotes = rng.randint(0, 3)
            quotes = [rng.choice(REAL_QUOTES[q.question_id] + FAKE_QUOTES) for _ in range(n_quotes)]
            chunks = [rng.choice(PROVIDED[q.question_id] + ["bad.id"]) for _ in range(rng.randint(0, 2))]
            criteria.append(crit(c.criterion_id, rng.choice(LEVEL_POOL), quotes, chunks))
        if rng.random() < 0.1:
            criteria.append(crit("CX", "full", ["Oxygen is given out"]))
        questions.append(QuestionDraft(question_id=q.question_id, criteria=criteria))
    return EvaluationDraft(questions=questions)


def test_property_totals_always_equal_rubric_level_sums():
    rng = random.Random(4)
    for _ in range(300):
        answers = dict(ANSWERS)
        overrides = {}
        if rng.random() < 0.2:
            answers["Q3"], overrides["Q3"] = "", "blank"
        r = validate(random_draft(rng), extraction(answers, **overrides))
        assert check_marks_are_rubric_derived(r, RUBRIC) == []
        for qe in r.questions:
            q = RUBRIC.question(qe.question_id)
            expected = sum(q.criterion(ce.criterion_id).level(ce.selected_level_id).marks for ce in qe.criteria)
            assert qe.total == pytest.approx(expected)
            assert qe.total <= q.max_marks
            assert {ce.criterion_id for ce in qe.criteria} == {c.criterion_id for c in q.criteria}
            for ce in qe.criteria:
                assert set(ce.reference_chunk_ids) <= set(PROVIDED[qe.question_id])
                if ce.marks > 0:
                    assert ce.evidence_verified
        assert r.total == pytest.approx(sum(qe.total for qe in r.questions))
        assert 0 <= r.total <= r.max_total == RUBRIC.max_total

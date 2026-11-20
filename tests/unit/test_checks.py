"""One test per deterministic check code (SENTINELEDU_TASKS.md 5.1)."""
from sentineledu.config import PROJECT_ROOT
from sentineledu.schemas.evaluation import CriterionDraft, EvaluationDraft, QuestionDraft
from sentineledu.schemas.extraction import ExtractedAnswer, ExtractedSubmission
from sentineledu.services.rubric import load_rubric
from sentineledu.services.scoring_rules import (
    DIAGRAM_PRESENT,
    ILLEGIBLE_IN_SCORED_ANSWER,
    LOW_OCR_CONFIDENCE,
    NO_EVIDENCE,
    OUT_OF_BOUNDS,
    SBERT_LEVEL_DISAGREEMENT,
    SUM_MISMATCH,
    VALIDATOR_DOWNGRADE,
    run_checks,
    validate_evaluation,
)

RUBRIC = load_rubric(PROJECT_ROOT / "data" / "rubrics" / "sample_rubric.json")
ANSWERS = {
    "Q1": "Photosynthesis is the process by which green plants make their own food using sunlight, carbon dioxide and water. Oxygen is released as a by-product.",
    "Q2": "The green pigment is chlorophyll. It is found in the chloroplasts of leaf cells.",
    "Q3": "Plants need water for photosynthesis and to carry minerals from the roots to the leaves.",
}


def extraction(answers=ANSWERS, **extra):
    out = []
    for qid, text in answers.items():
        kw = {"question_id": qid, "answer_text": text, "legibility": "clear" if text else "blank", "confidence": 0.95}
        kw.update(extra.get(qid, {}))
        out.append(ExtractedAnswer(**kw))
    return ExtractedSubmission(submission_id="s1", engine="gemini", answers=out)


def c(cid, level, quotes):
    return CriterionDraft(criterion_id=cid, selected_level_id=level, evidence_quotes=list(quotes), rationale="r")


def correct_draft():
    return EvaluationDraft(questions=[
        QuestionDraft(question_id="Q1", criteria=[c("C1", "full", ["make their own food using sunlight, carbon dioxide and water"]),
                                                  c("C2", "full", ["Oxygen is released"])]),
        QuestionDraft(question_id="Q2", criteria=[c("C1", "full", ["chlorophyll"]), c("C2", "full", ["chloroplasts"])]),
        QuestionDraft(question_id="Q3", criteria=[c("C1", "full", ["for photosynthesis", "carry minerals"])]),
    ])


def evaluation(ext=None, draft=None, sims=None):
    sims = {**{q: (0.6, 0.5) for q in ANSWERS}, **(sims or {})}  # unremarkable similarity unless a test says otherwise
    return validate_evaluation(draft or correct_draft(), RUBRIC, ext or extraction(), {}, sims=sims,
                               prompt_version="v", model_id="m")


def codes(flags):
    return [f.code for f in flags]


def test_correct_evaluation_has_no_flags():
    assert run_checks(evaluation(), extraction(), RUBRIC) == []


def test_sum_mismatch():
    ev = evaluation()
    ev.questions[0].total = 1.0  # criteria say 3
    flags = run_checks(ev, extraction(), RUBRIC)
    assert codes(flags).count(SUM_MISMATCH) == 2  # question level and assignment level
    assert flags[0].question_id == "Q1" and flags[0].severity == "high"


def test_out_of_bounds():
    ev = evaluation()
    ev.questions[1].criteria[0].marks = 5  # a 1-mark criterion
    ev.questions[1].total = 6
    ev.total = 12
    flags = run_checks(ev, extraction(), RUBRIC)
    assert OUT_OF_BOUNDS in codes(flags)
    assert any(f.criterion_id == "C1" and f.question_id == "Q2" for f in flags if f.code == OUT_OF_BOUNDS)
    assert any(f.question_id is None for f in flags if f.code == OUT_OF_BOUNDS)  # assignment total


def test_sbert_disagreement_high_similarity_lowest_level():
    d = correct_draft()
    d.questions[2].criteria[0] = c("C1", "none", [])
    ev = evaluation(draft=d, sims={"Q3": (0.9, 0.8)})
    flags = run_checks(ev, extraction(), RUBRIC)
    assert [f for f in flags if f.code == SBERT_LEVEL_DISAGREEMENT and f.question_id == "Q3"]


def test_sbert_disagreement_low_similarity_top_level():
    ev = evaluation(sims={"Q2": (0.1, 0.1)})
    flags = run_checks(ev, extraction(), RUBRIC)
    hits = [f for f in flags if f.code == SBERT_LEVEL_DISAGREEMENT]
    assert {f.criterion_id for f in hits} == {"C1", "C2"} and all(f.question_id == "Q2" for f in hits)


def test_sbert_disagreement_skipped_for_unattempted():
    ext = extraction({**ANSWERS, "Q3": ""})
    ev = evaluation(ext=ext, sims={"Q3": (0.95, 0.9)})
    assert SBERT_LEVEL_DISAGREEMENT not in codes(run_checks(ev, ext, RUBRIC))


def test_no_evidence():
    ev = evaluation()
    ev.questions[0].criteria[0].evidence_verified = False  # tampered: marks kept, evidence gone
    flags = run_checks(ev, extraction(), RUBRIC)
    hit = [f for f in flags if f.code == NO_EVIDENCE]
    assert len(hit) == 1 and hit[0].criterion_id == "C1" and hit[0].severity == "high"


def test_low_ocr_confidence():
    ext = extraction(Q2={"confidence": 0.4})
    flags = run_checks(evaluation(ext=ext), ext, RUBRIC, low_confidence=0.6)
    hit = [f for f in flags if f.code == LOW_OCR_CONFIDENCE]
    assert [f.question_id for f in hit] == ["Q2"]
    assert LOW_OCR_CONFIDENCE not in codes(run_checks(evaluation(ext=ext), ext, RUBRIC, low_confidence=0.3))


def test_illegible_in_scored_answer():
    text = "Plants need water for photosynthesis [[ILLEGIBLE]] carry minerals"
    ext = extraction({**ANSWERS, "Q3": text}, Q3={"legibility": "partial"})
    flags = run_checks(evaluation(ext=ext), ext, RUBRIC)
    assert [f.question_id for f in flags if f.code == ILLEGIBLE_IN_SCORED_ANSWER] == ["Q3"]


def test_diagram_present():
    ext = extraction(Q1={"contains_diagram": True})
    flags = run_checks(evaluation(ext=ext), ext, RUBRIC)
    hit = [f for f in flags if f.code == DIAGRAM_PRESENT]
    assert len(hit) == 1 and hit[0].severity == "info" and hit[0].question_id == "Q1"


def test_validator_downgrade_is_surfaced():
    d = correct_draft()
    d.questions[2].criteria[0].evidence_quotes = ["made up quote about transpiration"]
    ev = evaluation(draft=d)
    flags = run_checks(ev, extraction(), RUBRIC)
    hit = [f for f in flags if f.code == VALIDATOR_DOWNGRADE]
    assert len(hit) == 1 and hit[0].question_id == "Q3" and hit[0].criterion_id == "C1"

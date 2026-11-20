import pytest
from pydantic import ValidationError

from sentineledu.schemas import (
    CriterionVerdict,
    EvaluationResult,
    ExtractedAnswer,
    ExtractedSubmission,
    JudgeReport,
    MarkOverride,
    Rubric,
    TeacherDecision,
)


def levels(max_marks):
    return [
        {"level_id": "full", "label": "Full", "marks": max_marks, "descriptor": "complete"},
        {"level_id": "partial", "label": "Partial", "marks": max_marks / 2, "descriptor": "some"},
        {"level_id": "none", "label": "None", "marks": 0, "descriptor": "missing"},
    ]


def rubric_dict():
    return {
        "assignment_id": "A1",
        "title": "Biology quiz",
        "questions": [
            {
                "question_id": "Q1",
                "prompt": "Explain photosynthesis.",
                "max_marks": 5,
                "model_answer": "Plants convert light energy into chemical energy.",
                "criteria": [
                    {"criterion_id": "C1", "description": "Energy conversion", "max_marks": 3, "levels": levels(3)},
                    {"criterion_id": "C2", "description": "Where it happens", "max_marks": 2, "levels": levels(2)},
                ],
            },
            {
                "question_id": "Q2",
                "prompt": "Name the pigment.",
                "max_marks": 1,
                "model_answer": "Chlorophyll.",
                "criteria": [{"criterion_id": "C1", "description": "Pigment", "max_marks": 1, "levels": levels(1)}],
            },
        ],
    }


# --- Rubric ---

def test_valid_rubric_round_trips_json():
    r = Rubric.model_validate(rubric_dict())
    assert r.max_total == 6
    assert Rubric.model_validate_json(r.model_dump_json()) == r
    assert r.question("Q1").criterion("C2").level("partial").marks == 1
    assert r.question("Q1").criterion("C1").lowest_level.level_id == "none"


def test_criterion_ids_may_repeat_across_questions():
    Rubric.model_validate(rubric_dict())  # both questions use "C1"


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["questions"][0].update(max_marks=6), "criteria add up to 5"),
    (lambda d: d["questions"][0]["criteria"][0]["levels"].pop(), "needs a 0-mark level"),
    (lambda d: d["questions"][0]["criteria"][0].update(max_marks=4), "highest level is worth 3"),
    (lambda d: d["questions"][0]["criteria"][0]["levels"][1].update(level_id="full"), "duplicate level_id"),
    (lambda d: d["questions"][0]["criteria"][1].update(criterion_id="C1"), "duplicate criterion_id"),
    (lambda d: d["questions"][1].update(question_id="Q1"), "duplicate question_id"),
    (lambda d: d["questions"][0]["criteria"][0]["levels"][2].update(marks=-1), "greater than or equal to 0"),
    (lambda d: d.update(questions=[]), "at least 1"),
])
def test_invalid_rubrics_rejected(mutate, message):
    d = rubric_dict()
    mutate(d)
    with pytest.raises(ValidationError, match=message):
        Rubric.model_validate(d)


# --- Extraction ---

def answer(**kw):
    base = {"question_id": "Q1", "answer_text": "Plants use light.", "legibility": "clear", "confidence": 0.9}
    return {**base, **kw}


@pytest.mark.parametrize("conf", [-0.1, 1.01])
def test_confidence_out_of_range_rejected(conf):
    with pytest.raises(ValidationError):
        ExtractedAnswer.model_validate(answer(confidence=conf))


def test_unknown_legibility_rejected():
    with pytest.raises(ValidationError):
        ExtractedAnswer.model_validate(answer(legibility="smudged"))


def test_submission_round_trip_and_lookup():
    s = ExtractedSubmission(submission_id="S1", engine="gemini",
                            answers=[ExtractedAnswer.model_validate(answer())])
    assert ExtractedSubmission.model_validate_json(s.model_dump_json()) == s
    assert s.answer_for("Q1").confidence == 0.9
    assert s.answer_for("Q9") is None
    assert s.version == 1


# --- Evaluation / Judge ---

def test_evaluation_rejects_negative_marks():
    with pytest.raises(ValidationError):
        EvaluationResult.model_validate({
            "submission_id": "S1", "total": -1, "max_total": 6, "prompt_version": "v1", "model_id": "m",
            "questions": [],
        })


def test_judge_report_round_trip():
    v = CriterionVerdict(criterion_id="C1", verdict="agree", justification="ok", recommended_marks=3)
    r = JudgeReport(submission_id="S1", verdicts={"Q1": [v]}, feedback=[], summary_feedback="Good",
                    recommended_total=3, max_total=7)
    assert JudgeReport.model_validate_json(r.model_dump_json()) == r
    with pytest.raises(ValidationError):
        CriterionVerdict(criterion_id="C1", verdict="maybe", justification="?", recommended_marks=0)


# --- Teacher decision ---

def test_teacher_decision_requires_reviewer():
    with pytest.raises(ValidationError, match="reviewer name is required"):
        TeacherDecision(submission_id="S1", reviewer="  ", final_marks={"Q1": {"C1": 3}})


def test_override_requires_reason():
    with pytest.raises(ValidationError, match="needs a reason"):
        MarkOverride(question_id="Q1", criterion_id="C1", from_marks=1, to_marks=2, reason=" ")


def test_teacher_decision_total_and_negative_marks():
    d = TeacherDecision(submission_id="S1", reviewer="Ms Rao", final_marks={"Q1": {"C1": 3, "C2": 1.5}, "Q2": {"C1": 1}})
    assert d.total == 5.5
    assert d.approved_at.tzinfo is not None
    with pytest.raises(ValidationError, match="negative"):
        TeacherDecision(submission_id="S1", reviewer="Ms Rao", final_marks={"Q1": {"C1": -1}})

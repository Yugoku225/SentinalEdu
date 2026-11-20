import pytest

from sentineledu.schemas import ExtractedAnswer, ExtractedSubmission
from sentineledu.services.extraction_review import AnswerEdit, apply_correction, extraction_gate

QIDS = ["Q1", "Q2"]


def ans(qid, text="Some answer", legibility="clear", confidence=0.9):
    return ExtractedAnswer(question_id=qid, answer_text=text, legibility=legibility, confidence=confidence)


def ext(answers=None, **kw):
    if answers is None:
        answers = [ans("Q1"), ans("Q2")]
    return ExtractedSubmission(submission_id="S1", engine=kw.pop("engine", "gemini"), answers=answers, **kw)


# --- gate: one case per rule (ARCHITECTURE §4.1) ---

@pytest.mark.parametrize("extraction, ok, reason", [
    (ext(), True, None),
    (ext([ans("Q1", legibility="unreadable", text="[[ILLEGIBLE]]"), ans("Q2")]), False, "Q1: answer is unreadable"),
    (ext([ans("Q1", confidence=0.4), ans("Q2")]), False, "Q1: low extraction confidence"),
    (ext([ans("Q1"), ans("Q2", text="", legibility="blank", confidence=0.2)]), True, None),  # blank is not low-confidence
    (ext([ans("Q1")]), False, "not found on the pages: Q2"),
    (ext([ans("Q1"), ans("Q2")], missing_question_ids=["Q2"]), False, "not found"),
    (ext(unmatched_text="3) I liked this topic"), False, "could not be matched"),
    (ext(unmatched_text="   \n "), True, None),
    (ext([], engine="trocr", unmatched_text="text", missing_question_ids=QIDS), False, "TrOCR fallback"),
])
def test_gate_rules(extraction, ok, reason):
    result = extraction_gate(extraction, QIDS, threshold=0.6)
    assert result.ok is ok
    if reason:
        assert any(reason in r for r in result.reasons), result.reasons
    else:
        assert result.reasons == []


def test_gate_threshold_boundary():
    assert extraction_gate(ext([ans("Q1", confidence=0.6), ans("Q2")]), QIDS, 0.6).ok


def test_teacher_corrected_extraction_always_passes():
    corrected = ext([ans("Q1", confidence=0.1)], engine="teacher", unmatched_text="noise")
    assert extraction_gate(corrected, QIDS, 0.6).ok


# --- teacher correction ---

def test_correction_creates_new_version_and_keeps_original():
    original = ext([ans("Q1", "Plants [[ILLEGIBLE]] light", legibility="partial", confidence=0.5)],
                   missing_question_ids=["Q2"], unmatched_text="Q2 chlorophyl")
    corrected = apply_correction(original, [AnswerEdit(question_id="Q1", answer_text="Plants use light"),
                                            AnswerEdit(question_id="Q2", answer_text="chlorophyl")],
                                 "Ms Rao", unmatched_text="")
    assert original.answers[0].answer_text == "Plants [[ILLEGIBLE]] light"  # untouched
    assert original.version == 1 and corrected.version == 2
    assert corrected.engine == "teacher" and corrected.corrected_by == "Ms Rao"
    q1, q2 = corrected.answers
    assert (q1.answer_text, q1.legibility, q1.confidence) == ("Plants use light", "clear", 1.0)
    assert (q2.question_id, q2.answer_text) == ("Q2", "chlorophyl")  # added from unmatched text
    assert corrected.missing_question_ids == []
    assert corrected.unmatched_text == ""
    assert "Ms Rao corrected: Q1, Q2" in corrected.issues[-1]
    assert extraction_gate(corrected, QIDS, 0.6).ok


def test_correction_legibility_follows_text():
    c = apply_correction(ext(), [AnswerEdit(question_id="Q1", answer_text="  "),
                                 AnswerEdit(question_id="Q2", answer_text="cells [[ILLEGIBLE]]")], "T")
    assert [a.legibility for a in c.answers] == ["blank", "partial"]


def test_correction_keeps_unmatched_text_unless_given():
    assert apply_correction(ext(unmatched_text="x"), [], "T").unmatched_text == "x"


def test_correction_requires_teacher_name():
    with pytest.raises(ValueError, match="teacher name"):
        apply_correction(ext(), [], "  ")

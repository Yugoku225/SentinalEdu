"""Status guards and finalisation (SENTINELEDU_TASKS.md 6.4)."""
import itertools

import pytest
from pydantic import ValidationError

from sentineledu.schemas.evaluation import CriterionDraft, EvaluationDraft, QuestionDraft
from sentineledu.schemas.extraction import ExtractedAnswer, ExtractedSubmission
from sentineledu.schemas.judge import JudgeDraft
from sentineledu.schemas.review import MarkOverride, TeacherDecision
from sentineledu.schemas.submission import SubmissionStatus as S
from sentineledu.services.scoring_rules import validate_evaluation, validate_judge
from sentineledu.services.storage import ALLOWED_TRANSITIONS, FinalizationError, IllegalTransitionError, Storage
from sentineledu.workflow.actions import approve, request_reevaluation, submit_correction
from sentineledu.services.extraction_review import AnswerEdit
from wf_helpers import A1, A2, A3, PAGE, RUBRIC


@pytest.fixture
def storage(tmp_path):
    s = Storage(storage_dir=tmp_path / "storage")
    s.save_rubric(RUBRIC)
    return s


def force(storage: Storage, sid: str, status: S) -> None:
    """Put a submission in any state, bypassing the guards (test set-up only)."""
    with storage._conn() as c:
        c.execute("UPDATE submissions SET status = ? WHERE submission_id = ?", (status.value, sid))


@pytest.fixture
def sid(storage):
    return storage.create_submission(RUBRIC.assignment_id, [PAGE]).submission_id


@pytest.fixture
def pending(storage, sid):
    """A submission at PENDING_TEACHER_REVIEW with extraction, evaluation (Q1/C2 none) and judge report."""
    ext = storage.save_extraction(ExtractedSubmission(submission_id=sid, engine="gemini", answers=[
        ExtractedAnswer(question_id=q, answer_text=t, legibility="clear", confidence=0.9)
        for q, t in (("Q1", A1), ("Q2", A2), ("Q3", A3))]))
    draft = EvaluationDraft(questions=[
        QuestionDraft(question_id="Q1", criteria=[
            CriterionDraft(criterion_id="C1", selected_level_id="full", evidence_quotes=["make their own food using sunlight"]),
            CriterionDraft(criterion_id="C2", selected_level_id="none")]),
        QuestionDraft(question_id="Q2", criteria=[CriterionDraft(criterion_id="C1", selected_level_id="full", evidence_quotes=["chlorophyll"]),
                                                  CriterionDraft(criterion_id="C2", selected_level_id="full", evidence_quotes=["chloroplasts"])]),
        QuestionDraft(question_id="Q3", criteria=[CriterionDraft(criterion_id="C1", selected_level_id="partial", evidence_quotes=["for photosynthesis"])]),
    ])
    ev = storage.save_evaluation(validate_evaluation(draft, RUBRIC, ext, {}, prompt_version="v", model_id="m"))
    storage.save_judge_report(validate_judge(JudgeDraft(), ev, RUBRIC, ext))
    force(storage, sid, S.PENDING_TEACHER_REVIEW)
    return sid


def marks(**over):
    m = {"Q1": {"C1": 2.0, "C2": 0.0}, "Q2": {"C1": 1.0, "C2": 1.0}, "Q3": {"C1": 1.0}}
    for key, v in over.items():
        q, c = key.split("_")
        m[q][c] = v
    return m


# --- transitions ---------------------------------------------------------------------------

LEGAL = {(a, b) for a, targets in ALLOWED_TRANSITIONS.items() for b in targets}
ILLEGAL = [(a, b) for a, b in itertools.product(S, S) if a != b and (a, b) not in LEGAL]


@pytest.mark.parametrize("from_status,to_status", ILLEGAL, ids=[f"{a.value}->{b.value}" for a, b in ILLEGAL])
def test_illegal_transitions_are_refused(storage, sid, from_status, to_status):
    force(storage, sid, from_status)
    with pytest.raises(IllegalTransitionError):
        storage.set_status(sid, to_status)
    assert storage.get_submission(sid).status == from_status


@pytest.mark.parametrize("from_status,to_status", sorted(LEGAL), ids=[f"{a.value}->{b.value}" for a, b in sorted(LEGAL)])
def test_legal_transitions_are_allowed(storage, sid, from_status, to_status):
    force(storage, sid, from_status)
    assert storage.set_status(sid, to_status) == from_status
    assert storage.get_submission(sid).status == to_status


def test_examples_from_the_task():
    assert (S.EXTRACTED, S.FINALIZED) in ILLEGAL
    assert all(b != S.FINALIZED for _, b in LEGAL)  # nothing reaches FINALIZED through set_status
    assert ALLOWED_TRANSITIONS[S.FINALIZED] == frozenset()


def test_same_status_is_a_noop(storage, sid):
    assert storage.set_status(sid, S.UPLOADED) == S.UPLOADED


# --- finalize ------------------------------------------------------------------------------

def test_finalize_with_recommended_marks(storage, pending):
    d = storage.finalize(TeacherDecision(submission_id=pending, reviewer="Ms Rao", final_marks=marks()))
    assert storage.get_submission(pending).status == S.FINALIZED
    assert storage.get_decision(pending).reviewer == "Ms Rao" and d.total == 5
    with pytest.raises(FinalizationError, match="FINALIZED"):
        storage.finalize(TeacherDecision(submission_id=pending, reviewer="Ms Rao", final_marks=marks()))
    with pytest.raises(IllegalTransitionError):
        storage.set_status(pending, S.EXTRACTED)


def test_reviewer_and_override_reason_are_required():
    with pytest.raises(ValidationError, match="reviewer"):
        TeacherDecision(submission_id="x", reviewer="  ", final_marks={})
    with pytest.raises(ValidationError, match="reason"):
        MarkOverride(question_id="Q1", criterion_id="C2", from_marks=0, to_marks=1, reason=" ")


@pytest.mark.parametrize("status", [S.UPLOADED, S.EXTRACTED, S.NEEDS_EXTRACTION_REVIEW, S.EVALUATED, S.FAILED])
def test_finalize_needs_pending_review(storage, pending, status):
    force(storage, pending, status)
    with pytest.raises(FinalizationError, match="only PENDING_TEACHER_REVIEW"):
        storage.finalize(TeacherDecision(submission_id=pending, reviewer="T", final_marks=marks()))
    assert storage.get_decision(pending) is None


def test_changed_mark_needs_matching_override(storage, pending):
    with pytest.raises(FinalizationError, match="Q1/C2.*override"):
        storage.finalize(TeacherDecision(submission_id=pending, reviewer="T", final_marks=marks(Q1_C2=1.0)))
    wrong = MarkOverride(question_id="Q1", criterion_id="C2", from_marks=0, to_marks=0.5, reason="partly there")
    with pytest.raises(FinalizationError, match="override says"):
        storage.finalize(TeacherDecision(submission_id=pending, reviewer="T", final_marks=marks(Q1_C2=1.0), overrides=[wrong]))
    ok = MarkOverride(question_id="Q1", criterion_id="C2", from_marks=0, to_marks=1, reason="'by-product' implies oxygen here")
    d = storage.finalize(TeacherDecision(submission_id=pending, reviewer="T", final_marks=marks(Q1_C2=1.0), overrides=[ok]))
    assert d.total == 6 and storage.get_decision(pending).overrides[0].reason.startswith("'by-product'")


def test_marks_must_cover_rubric_and_stay_in_bounds(storage, pending):
    m = marks(); del m["Q3"]
    with pytest.raises(FinalizationError, match="missing.*Q3/C1"):
        storage.finalize(TeacherDecision(submission_id=pending, reviewer="T", final_marks=m))
    m = marks(); m["Q9"] = {"C1": 1.0}
    with pytest.raises(FinalizationError, match="unknown.*Q9/C1"):
        storage.finalize(TeacherDecision(submission_id=pending, reviewer="T", final_marks=m))
    over = MarkOverride(question_id="Q2", criterion_id="C1", from_marks=1, to_marks=5, reason="bonus")
    with pytest.raises(FinalizationError, match="above the criterion maximum"):
        storage.finalize(TeacherDecision(submission_id=pending, reviewer="T", final_marks=marks(Q2_C1=5.0), overrides=[over]))
    assert storage.get_submission(pending).status == S.PENDING_TEACHER_REVIEW


# --- teacher actions -----------------------------------------------------------------------

def test_approve_builds_overrides_from_changes(storage, pending):
    d = approve(storage, pending, "Ms Rao", changes={("Q3", "C1"): (2.0, "Gave two reasons on re-reading")},
                teacher_comment="Good")
    assert d.total == 6 and [(o.from_marks, o.to_marks) for o in d.overrides] == [(1.0, 2.0)]
    assert storage.get_submission(pending).status == S.FINALIZED


def test_approve_refuses_change_without_reason(storage, pending):
    with pytest.raises(ValidationError, match="reason"):
        approve(storage, pending, "Ms Rao", changes={("Q3", "C1"): (2.0, "")})


def test_correction_moves_back_to_extracted(storage, pending):
    v = submit_correction(storage, pending, [AnswerEdit(question_id="Q3", answer_text=A3 + " It keeps cells firm.")], "Ms Rao")
    assert v.version == 2 and v.engine == "teacher" and v.corrected_by == "Ms Rao"
    assert storage.get_submission(pending).status == S.EXTRACTED
    assert storage.list_extraction_versions(pending) == [1, 2]


def test_request_reevaluation(storage, pending, sid):
    request_reevaluation(storage, pending)
    assert storage.get_submission(pending).status == S.EXTRACTED
    force(storage, sid, S.UPLOADED)
    with pytest.raises(IllegalTransitionError):
        request_reevaluation(storage, sid)


def test_correction_refused_after_finalisation(storage, pending):
    approve(storage, pending, "T")
    with pytest.raises(IllegalTransitionError, match="cannot be corrected"):
        submit_correction(storage, pending, [AnswerEdit(question_id="Q1", answer_text="x")], "T")

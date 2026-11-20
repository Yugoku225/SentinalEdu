"""Each workflow stage on its own, with fake agents (SENTINELEDU_TASKS.md 6.1)."""
import pytest
from PIL import Image as PILImage

from sentineledu.schemas.submission import SubmissionStatus as S
from sentineledu.workflow.steps import AssessmentSteps, StepFailed
from fakes import ProviderError
from wf_helpers import RUBRIC, doc_draft, eval_draft, judge_draft, make_deps, new_submission


def force(storage, sid, status):
    with storage._conn() as c:
        c.execute("UPDATE submissions SET status = ? WHERE submission_id = ?", (status.value, sid))


def status(deps, sid):
    return deps.storage.get_submission(sid).status


# --- integrity_precheck --------------------------------------------------------------------

def test_precheck_ok_and_missing_rubric(tmp_path):
    deps = make_deps(tmp_path)
    sid = new_submission(deps)
    r = AssessmentSteps(deps).integrity_precheck(sid)
    assert r.status == "ok" and not r.stop and "Phase 7" in r.detail
    deps2 = make_deps(tmp_path / "b", store_rubric=False)
    sid2 = new_submission(deps2)
    with pytest.raises(StepFailed, match="No rubric"):
        AssessmentSteps(deps2).integrity_precheck(sid2)


@pytest.mark.parametrize("start,expected", [(S.FAILED, S.UPLOADED), (S.EVALUATED, S.EXTRACTED),
                                            (S.PENDING_TEACHER_REVIEW, S.EXTRACTED), (S.EXTRACTED, S.EXTRACTED)])
def test_precheck_puts_reruns_on_the_right_status(tmp_path, start, expected):
    deps = make_deps(tmp_path)
    sid = new_submission(deps)
    force(deps.storage, sid, start)
    AssessmentSteps(deps).integrity_precheck(sid)
    assert status(deps, sid) == expected


def test_precheck_stops_on_finalised(tmp_path):
    deps = make_deps(tmp_path)
    sid = new_submission(deps)
    force(deps.storage, sid, S.FINALIZED)
    r = AssessmentSteps(deps).integrity_precheck(sid)
    assert r.stop and r.status == "stopped" and status(deps, sid) == S.FINALIZED


# --- document_understanding ----------------------------------------------------------------

def test_document_step_extracts_once_then_reuses(tmp_path):
    deps = make_deps(tmp_path, doc=[doc_draft()])
    sid = new_submission(deps)
    steps = AssessmentSteps(deps)
    r = steps.document_understanding(sid)
    assert r.status == "ok" and r.model_id == "fake-model" and r.data["extraction_version"] == 1
    assert status(deps, sid) == S.EXTRACTED and len(deps.document_agent.calls) == 1
    again = steps.document_understanding(sid)
    assert again.status == "skipped" and "Gemini not called" in again.detail
    assert len(deps.document_agent.calls) == 1


def test_document_step_failure_marks_failed(tmp_path):
    scan = tmp_path / "scan.pdf"  # image-only PDF: the TrOCR fallback cannot read PDFs
    PILImage.new("RGB", (300, 400), "white").save(scan, format="PDF")
    deps = make_deps(tmp_path, doc=[ProviderError("400 INVALID_ARGUMENT")])
    sid = new_submission(deps, files=[scan])
    with pytest.raises(StepFailed, match="Could not read"):
        AssessmentSteps(deps).document_understanding(sid)
    assert status(deps, sid) == S.FAILED
    assert deps.storage.latest_extraction(sid) is None


# --- extraction_gate -----------------------------------------------------------------------

def test_gate_passes_clean_extraction(tmp_path):
    deps = make_deps(tmp_path, doc=[doc_draft()])
    sid = new_submission(deps)
    steps = AssessmentSteps(deps)
    steps.document_understanding(sid)
    r = steps.extraction_gate(sid)
    assert r.status == "ok" and not r.stop and status(deps, sid) == S.EXTRACTED


def test_gate_stops_on_unreadable_answer(tmp_path):
    deps = make_deps(tmp_path, doc=[doc_draft(q2_unreadable=True)])
    sid = new_submission(deps)
    steps = AssessmentSteps(deps)
    steps.document_understanding(sid)
    r = steps.extraction_gate(sid)
    assert r.stop and r.status == "stopped" and any("Q2" in x for x in r.data["gate_reasons"])
    assert status(deps, sid) == S.NEEDS_EXTRACTION_REVIEW
    assert steps.extraction_gate(sid).stop  # re-running without a correction stops again


# --- evaluation ----------------------------------------------------------------------------

def _extracted(tmp_path, **agents):
    deps = make_deps(tmp_path, doc=[doc_draft()], **agents)
    sid = new_submission(deps)
    steps = AssessmentSteps(deps)
    steps.document_understanding(sid)
    return deps, sid, steps


def test_evaluation_step_saves_and_moves_to_evaluated(tmp_path):
    deps, sid, steps = _extracted(tmp_path, ev=[eval_draft()])
    r = steps.evaluation(sid)
    assert r.status == "ok" and r.data["total"] == 7 and r.prompt_version == "evaluation_v1"
    assert status(deps, sid) == S.EVALUATED
    assert deps.storage.latest_evaluation(sid).evaluation_id == r.data["evaluation_id"]


def test_evaluation_failure_keeps_extracted(tmp_path):
    deps, sid, steps = _extracted(tmp_path, ev=[ProviderError("400 bad request")])
    with pytest.raises(StepFailed, match="could not grade"):
        steps.evaluation(sid)
    assert status(deps, sid) == S.EXTRACTED and deps.storage.latest_evaluation(sid) is None


# --- judge ---------------------------------------------------------------------------------

def test_judge_step_saves_report(tmp_path):
    deps, sid, steps = _extracted(tmp_path, ev=[eval_draft()], jd=[judge_draft()])
    steps.evaluation(sid)
    r = steps.judge(sid)
    assert r.status == "ok" and r.data["recommended_total"] == 7
    report = deps.storage.latest_judge_report(sid)
    assert report.evaluation_id == deps.storage.latest_evaluation(sid).evaluation_id
    assert status(deps, sid) == S.EVALUATED  # the hand-over happens in mark_pending_review


def test_judge_failure_falls_back_to_checks_only(tmp_path):
    deps, sid, steps = _extracted(tmp_path, ev=[eval_draft()], jd=[ProviderError("400 bad request")])
    steps.evaluation(sid)
    r = steps.judge(sid)
    assert r.status == "fallback" and r.model_id == "none"
    assert "JUDGE_NOT_RUN" in [f.code for f in deps.storage.latest_judge_report(sid).flags]


def test_judge_failure_without_fallback_stops(tmp_path):
    deps, sid, steps = _extracted(tmp_path, ev=[eval_draft()], jd=[ProviderError("400 bad request")], judge_fallback=False)
    steps.evaluation(sid)
    with pytest.raises(StepFailed, match="could not review"):
        steps.judge(sid)
    assert deps.storage.latest_judge_report(sid) is None and status(deps, sid) == S.EVALUATED


# --- integrity_postcheck and mark_pending_review --------------------------------------------

def test_postcheck_is_a_placeholder(tmp_path):
    deps = make_deps(tmp_path)
    r = AssessmentSteps(deps).integrity_postcheck(new_submission(deps))
    assert r.status == "skipped" and "Phase 7" in r.detail


def test_mark_pending_review_needs_a_matching_judge_report(tmp_path):
    deps, sid, steps = _extracted(tmp_path, ev=[eval_draft(), eval_draft()], jd=[judge_draft()])
    steps.evaluation(sid)
    with pytest.raises(StepFailed, match="no judge report"):
        steps.mark_pending_review(sid)
    steps.judge(sid)
    assert steps.mark_pending_review(sid).status == "ok"
    assert status(deps, sid) == S.PENDING_TEACHER_REVIEW
    # a newer evaluation without its own report must not be handed over
    deps.storage.set_status(sid, S.EXTRACTED)
    steps.evaluation(sid)
    with pytest.raises(StepFailed):
        steps.mark_pending_review(sid)
    assert RUBRIC.assignment_id == deps.storage.get_submission(sid).assignment_id

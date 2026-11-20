"""The whole Agno workflow with fake agents (SENTINELEDU_TASKS.md 6.2, 6.3, 6.6). Runs offline."""
import json

import pytest

from sentineledu.cli import main as cli_main
from sentineledu.schemas.submission import SubmissionStatus as S
from sentineledu.services.extraction_review import AnswerEdit
from sentineledu.services.storage import FinalizationError, IllegalTransitionError
from sentineledu.workflow.actions import approve, submit_correction
from sentineledu.workflow.assessment_workflow import assess_files, run_assessment
from sentineledu.workflow.steps import STEP_ORDER
from fakes import ProviderError
from wf_helpers import PAGE, RUBRIC, doc_draft, eval_draft, judge_draft, make_deps


def test_happy_path_reaches_pending_teacher_review(tmp_path):
    deps = make_deps(tmp_path, doc=[doc_draft()], ev=[eval_draft()], jd=[judge_draft()])
    run = assess_files(RUBRIC.assignment_id, [PAGE], deps, student_ref="S01")
    assert run.status == S.PENDING_TEACHER_REVIEW and run.stopped_at is None
    assert [s.step for s in run.steps] == STEP_ORDER
    assert [s.status for s in run.steps] == ["ok"] * 7
    assert run.evaluation_total == 7 and run.recommended_total == 7 and run.max_total == 7
    assert run.extraction_version == 1 and run.evaluation_id
    assert [len(a.calls) for a in (deps.document_agent, deps.evaluation_agent, deps.judge_agent)] == [1, 1, 1]


def test_audit_log_has_one_row_per_step(tmp_path):
    deps = make_deps(tmp_path, doc=[doc_draft()], ev=[eval_draft()], jd=[judge_draft()])
    run = assess_files(RUBRIC.assignment_id, [PAGE], deps)
    rows = deps.storage.audit_log(run.submission_id, run.run_id)
    assert [r["step"] for r in rows] == STEP_ORDER
    by_step = {r["step"]: r for r in rows}
    assert by_step["document_understanding"]["model_id"] == "fake-model"
    assert by_step["document_understanding"]["prompt_version"] == "document_v1"
    assert by_step["evaluation"]["prompt_version"] == "evaluation_v1"
    assert by_step["judge"]["prompt_version"] == "judge_v1"
    assert all(isinstance(r["duration_ms"], int) and r["error"] is None for r in rows)


def test_gate_stops_then_correction_and_resume_call_agent1_once(tmp_path):
    deps = make_deps(tmp_path, doc=[doc_draft(q2_unreadable=True)], ev=[eval_draft()], jd=[judge_draft()])
    run = assess_files(RUBRIC.assignment_id, [PAGE], deps)
    sid = run.submission_id
    assert run.status == S.NEEDS_EXTRACTION_REVIEW and run.stopped_at == "extraction_gate"
    assert "Q2" in run.reason
    assert [s.step for s in run.steps] == STEP_ORDER[:3]  # nothing after the gate ran
    assert deps.evaluation_agent.calls == []

    submit_correction(deps.storage, sid, [AnswerEdit(question_id="Q2", answer_text=
                      "The green pigment is chlorophyll. It is found in the chloroplasts of leaf cells.")], "Ms Rao")
    resumed = run_assessment(sid, deps)
    assert resumed.status == S.PENDING_TEACHER_REVIEW
    assert resumed.steps[1].status == "skipped" and resumed.extraction_version == 2
    assert len(deps.document_agent.calls) == 1  # Gemini read the pages exactly once across correction and resume
    assert resumed.evaluation_total == 7


def test_rerun_of_evaluated_submission_keeps_history(tmp_path):
    deps = make_deps(tmp_path, doc=[doc_draft()], ev=[eval_draft(), eval_draft()], jd=[judge_draft(), judge_draft()])
    first = assess_files(RUBRIC.assignment_id, [PAGE], deps)
    second = run_assessment(first.submission_id, deps)
    assert second.status == S.PENDING_TEACHER_REVIEW
    assert second.evaluation_id != first.evaluation_id
    assert len(deps.storage.list_evaluations(first.submission_id)) == 2
    assert len(deps.document_agent.calls) == 1 and len(deps.evaluation_agent.calls) == 2
    assert len(deps.storage.audit_log(first.submission_id)) == 14


def test_evaluation_error_stops_the_run_and_a_retry_resumes(tmp_path):
    deps = make_deps(tmp_path, doc=[doc_draft()], ev=[ProviderError("400 bad request"), eval_draft()], jd=[judge_draft()])
    run = assess_files(RUBRIC.assignment_id, [PAGE], deps)
    assert run.status == S.EXTRACTED and run.stopped_at == "evaluation"
    assert run.steps[-1].status == "error" and "could not grade" in run.steps[-1].error
    assert [s.step for s in run.steps] == STEP_ORDER[:4]  # judge and later steps did not run
    retry = run_assessment(run.submission_id, deps)
    assert retry.status == S.PENDING_TEACHER_REVIEW and len(deps.document_agent.calls) == 1


def test_unexpected_exception_is_contained(tmp_path, monkeypatch):
    """A bug inside a step must stop the run, not let Agno carry on to the next step."""
    import sentineledu.workflow.steps as steps_module

    def broken(*args, **kwargs):
        raise KeyError("simulated bug")
    monkeypatch.setattr(steps_module, "build_context", broken)
    deps = make_deps(tmp_path, doc=[doc_draft()], ev=[eval_draft()], jd=[judge_draft()])
    run = assess_files(RUBRIC.assignment_id, [PAGE], deps)
    assert run.stopped_at == "evaluation" and run.steps[-1].status == "error"
    assert "KeyError" in run.steps[-1].error
    assert [s.step for s in run.steps] == STEP_ORDER[:4]
    assert run.status == S.EXTRACTED and deps.evaluation_agent.calls == []


def test_judge_outage_still_reaches_review_with_flag(tmp_path):
    deps = make_deps(tmp_path, doc=[doc_draft()], ev=[eval_draft()], jd=[ProviderError("400 bad request")])
    run = assess_files(RUBRIC.assignment_id, [PAGE], deps)
    assert run.status == S.PENDING_TEACHER_REVIEW
    assert run.steps[STEP_ORDER.index("judge")].status == "fallback" and any("JUDGE_NOT_RUN" in f for f in run.flags)


def test_finalisation_only_through_teacher_decision(tmp_path):
    deps = make_deps(tmp_path, doc=[doc_draft()], ev=[eval_draft()], jd=[judge_draft()])
    run = assess_files(RUBRIC.assignment_id, [PAGE], deps)
    sid = run.submission_id
    with pytest.raises(IllegalTransitionError):
        deps.storage.set_status(sid, S.FINALIZED)
    decision = approve(deps.storage, sid, "Ms Rao")
    assert decision.total == 7 and deps.storage.get_submission(sid).status == S.FINALIZED
    again = run_assessment(sid, deps)  # a finalised submission is never re-graded by the workflow
    assert again.status == S.FINALIZED and again.stopped_at == "integrity_precheck"
    with pytest.raises(FinalizationError):
        approve(deps.storage, sid, "Ms Rao")


def test_cli_status_and_rubric_import(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "cli_storage"))
    monkeypatch.setenv("OFFLINE_MODE", "true")
    from sentineledu.config import PROJECT_ROOT
    assert cli_main(["rubric-import", str(PROJECT_ROOT / "data" / "rubrics" / "sample_rubric.json")]) == 0
    assert json.loads(capsys.readouterr().out)["assignment_id"] == RUBRIC.assignment_id
    assert cli_main(["status", "nope"]) == 2

"""The single Agno workflow that takes a submission to PENDING_TEACHER_REVIEW (TASKS 6.2, ARCHITECTURE §7).

Verified Agno 3.2.0 behaviour this module depends on (see DECISIONS D-26):
- A step whose executor raises does NOT stop the workflow: later steps still run and the run
  reports "completed". So every executor here catches its own errors and returns
  `StepOutput(stop=True, success=False)` instead of raising.
- Steps are retried 3 extra times by default. A retried Gemini stage would multiply API calls
  (and the runner already retries transient errors), so every step uses `max_retries=0`.
- `StepOutput(stop=True)` ends the run early; that is how the extraction gate stops it.

Teacher approval is deliberately outside the workflow (D-15): the run ends at
PENDING_TEACHER_REVIEW and `Storage.finalize()` is the only way to FINALIZED.
"""
import logging
import time
import uuid
from pathlib import Path
from typing import Sequence

from agno.workflow import Step, Workflow
from agno.workflow.types import StepInput, StepOutput
from pydantic import BaseModel

from sentineledu.schemas.submission import SubmissionStatus
from sentineledu.workflow.steps import STEP_ORDER, AssessmentSteps, StepFailed, StepResult, WorkflowDeps

log = logging.getLogger("sentineledu.workflow")


class StepRecord(BaseModel):
    step: str
    status: str  # ok | skipped | stopped | fallback | error
    duration_ms: int | None = None
    model_id: str | None = None
    prompt_version: str | None = None
    detail: str | None = None
    error: str | None = None


class AssessmentRun(BaseModel):
    """What one run did. Printed by the CLI; shown by the dashboard."""
    run_id: str
    submission_id: str
    assignment_id: str
    status: SubmissionStatus
    stopped_at: str | None = None  # step that ended the run early, if any
    reason: str | None = None
    steps: list[StepRecord]
    extraction_version: int | None = None
    evaluation_id: str | None = None
    evaluation_total: float | None = None
    recommended_total: float | None = None
    max_total: float | None = None
    flags: list[str] = []
    duration_ms: int = 0


def _executor(steps: AssessmentSteps, name: str, run_id: str):
    method = steps.by_name(name)
    storage = steps.storage

    def run(step_input: StepInput) -> StepOutput:
        submission_id = str(step_input.input)
        start = time.monotonic()
        try:
            result: StepResult = method(submission_id)
        except Exception as e:  # noqa: BLE001 - must not escape: Agno would carry on with the next step
            ms = int((time.monotonic() - start) * 1000)
            expected = isinstance(e, StepFailed)
            if not expected:
                log.exception("[%s] unexpected error in %s", submission_id, name)
            message = str(e) if expected else f"{type(e).__name__}: {e}"
            storage.log_step(submission_id, name, "error", run_id=run_id, duration_ms=ms, error=message)
            return StepOutput(content=message, success=False, error=message, stop=True)
        ms = int((time.monotonic() - start) * 1000)
        storage.log_step(submission_id, name, result.status, run_id=run_id, model_id=result.model_id,
                         prompt_version=result.prompt_version, duration_ms=ms, detail=result.detail)
        log.info("[%s] %-22s %-8s %6d ms  %s", submission_id, name, result.status, ms, result.detail)
        return StepOutput(content=result.detail, stop=result.stop)

    run.__name__ = name
    return run


def build_workflow(deps: WorkflowDeps, run_id: str) -> Workflow:
    steps = AssessmentSteps(deps)
    return Workflow(
        name="sentineledu-assessment",
        description="Raw submission -> extraction -> gate -> evaluation -> judge -> pending teacher review",
        steps=[Step(name=name, executor=_executor(steps, name, run_id), max_retries=0) for name in STEP_ORDER],
        telemetry=False,
    )


def run_assessment(submission_id: str, deps: WorkflowDeps) -> AssessmentRun:
    """Run (or resume) the workflow for a stored submission and summarise what happened."""
    storage = deps.storage
    sub = storage.get_submission(submission_id)  # NotFoundError before anything runs
    run_id = uuid.uuid4().hex[:12]
    start = time.monotonic()
    build_workflow(deps, run_id).run(input=submission_id)
    return summarise_run(deps, submission_id, run_id, sub.assignment_id, int((time.monotonic() - start) * 1000))


def summarise_run(deps: WorkflowDeps, submission_id: str, run_id: str, assignment_id: str, duration_ms: int = 0) -> AssessmentRun:
    storage = deps.storage
    records = [StepRecord(**{k: row[k] for k in StepRecord.model_fields}) for row in storage.audit_log(submission_id, run_id)]
    ended = next((r for r in records if r.status in ("stopped", "error")), None)
    sub = storage.get_submission(submission_id)
    run = AssessmentRun(run_id=run_id, submission_id=submission_id, assignment_id=assignment_id, status=sub.status,
                        stopped_at=ended.step if ended else None,
                        reason=(ended.error or ended.detail) if ended else None, steps=records, duration_ms=duration_ms)
    extraction = storage.latest_extraction(submission_id)
    if extraction is not None:
        run.extraction_version = extraction.version
    if sub.status in (SubmissionStatus.EVALUATED, SubmissionStatus.PENDING_TEACHER_REVIEW, SubmissionStatus.FINALIZED):
        evaluation = storage.latest_evaluation(submission_id)
        if evaluation is not None:
            run.evaluation_id, run.evaluation_total, run.max_total = evaluation.evaluation_id, evaluation.total, evaluation.max_total
            report = storage.latest_judge_report(submission_id, evaluation.evaluation_id)
            if report is not None:
                run.recommended_total = report.recommended_total
                run.flags = [f"{f.severity}:{f.code}" + (f":{f.question_id}" if f.question_id else "")
                             + (f"/{f.criterion_id}" if f.criterion_id else "") for f in report.flags]
    return run


def assess_files(assignment_id: str, files: Sequence[Path | str], deps: WorkflowDeps,
                 student_ref: str | None = None) -> AssessmentRun:
    """Store a new submission and run the workflow on it (the one-command path, TASKS 6.5)."""
    sub = deps.storage.create_submission(assignment_id, files, student_ref=student_ref)
    return run_assessment(sub.submission_id, deps)

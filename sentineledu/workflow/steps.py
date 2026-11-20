"""The seven assessment stages as plain methods (TASKS 6.1, ARCHITECTURE §3 and §7).

Each method takes a `submission_id`, loads what it needs from storage, does its work, persists the
result and moves the status. It returns a `StepResult` saying whether the workflow should continue.
`assessment_workflow.py` wraps each method as an Agno step; the methods are also unit tested directly.

Resume semantics (TASKS 6.3): every stage is idempotent with respect to stored state.
- Document understanding is skipped when an extraction already exists (e.g. a teacher correction),
  so Gemini is never asked to read the same submission twice.
- Re-running an evaluated or pending submission returns it to EXTRACTED first, and evaluation then
  appends a NEW evaluation and judge report; earlier runs are kept.
"""
import logging
from dataclasses import dataclass, field
from typing import Any

from sentineledu.agents.document_agent import DocumentExtractionError, extract_submission
from sentineledu.agents.evaluation_agent import EvaluationError, QuestionContext, build_context, evaluate
from sentineledu.agents.judge_agent import JudgeError, checks_only_report, judge
from sentineledu.agents.runner import ResponseCache
from sentineledu.config import Settings, get_settings
from sentineledu.schemas.rubric import Rubric
from sentineledu.schemas.submission import Submission, SubmissionStatus
from sentineledu.services.extraction_review import extraction_gate
from sentineledu.services.rag import RagStore
from sentineledu.services.storage import NotFoundError, Storage

log = logging.getLogger("sentineledu.workflow")

S = SubmissionStatus

# Step names, in workflow order (also used in the audit log).
INTEGRITY_PRECHECK = "integrity_precheck"
DOCUMENT_UNDERSTANDING = "document_understanding"
EXTRACTION_GATE = "extraction_gate"
EVALUATION = "evaluation"
JUDGE = "judge"
INTEGRITY_POSTCHECK = "integrity_postcheck"
MARK_PENDING_REVIEW = "mark_pending_review"
STEP_ORDER = [INTEGRITY_PRECHECK, DOCUMENT_UNDERSTANDING, EXTRACTION_GATE, EVALUATION, JUDGE,
              INTEGRITY_POSTCHECK, MARK_PENDING_REVIEW]


class StepFailed(RuntimeError):
    """A stage could not complete. The workflow stops; stored state is left consistent."""


@dataclass
class StepResult:
    status: str  # ok | skipped | stopped | fallback
    detail: str = ""
    model_id: str | None = None
    prompt_version: str | None = None
    stop: bool = False
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowDeps:
    """Everything the stages need. Agents left as None are built from settings on first use;
    tests pass fakes here."""
    storage: Storage
    settings: Settings = field(default_factory=get_settings)
    rag: RagStore | None = None
    document_agent: Any = None
    evaluation_agent: Any = None
    judge_agent: Any = None
    cache: ResponseCache | None = None
    use_cache: bool = True
    judge_fallback: bool = True  # Gemini down for the Judge -> checks-only report instead of stopping
    enhance_images: bool = False

    def __post_init__(self):
        if self.rag is None:
            self.rag = RagStore(self.settings)
        if self.cache is None and self.use_cache:
            self.cache = ResponseCache(self.settings.cache_dir)


class AssessmentSteps:
    def __init__(self, deps: WorkflowDeps):
        self.deps = deps
        self.storage = deps.storage
        self.settings = deps.settings
        self._contexts: dict[str, dict[str, QuestionContext]] = {}  # per run: evaluation and judge share it

    # --- helpers --------------------------------------------------------------------------------

    def _submission(self, submission_id: str) -> Submission:
        return self.storage.get_submission(submission_id)

    def _rubric(self, sub: Submission) -> Rubric:
        try:
            return self.storage.get_rubric(sub.assignment_id)
        except NotFoundError:
            raise StepFailed(f"No rubric is stored for assignment {sub.assignment_id!r}. Import one first.") from None

    def _context(self, submission_id: str, extraction, rubric: Rubric) -> dict[str, QuestionContext]:
        key = f"{submission_id}:v{extraction.version}"
        if key not in self._contexts:
            self._contexts[key] = build_context(extraction, rubric, rag=self.deps.rag, settings=self.settings)
        return self._contexts[key]

    # --- 1. integrity pre-check -----------------------------------------------------------------

    def integrity_precheck(self, submission_id: str) -> StepResult:
        """Check the submission can run, and put a re-run back on the right starting status.

        Exact-duplicate detection (EXACT_DUPLICATE / TEXT_DUPLICATE) is added here in Phase 7 (task 7.1).
        """
        sub = self._submission(submission_id)
        if sub.status == S.FINALIZED:
            return StepResult("stopped", "Already finalised by a teacher; nothing to do.", stop=True)
        self._rubric(sub)  # fail early, before paying for OCR, if the rubric is missing
        notes = []
        if sub.status == S.FAILED:
            self.storage.set_status(submission_id, S.UPLOADED)
            notes.append("retrying a FAILED submission")
        elif sub.status in (S.EVALUATED, S.PENDING_TEACHER_REVIEW):
            self.storage.set_status(submission_id, S.EXTRACTED)
            notes.append(f"re-evaluating ({sub.status.value} -> EXTRACTED); earlier results are kept")
        notes.append("duplicate checks are added in Phase 7")
        return StepResult("ok", "; ".join(notes))

    # --- 2. document understanding (Agent 1) ----------------------------------------------------

    def document_understanding(self, submission_id: str) -> StepResult:
        sub = self._submission(submission_id)
        latest = self.storage.latest_extraction(submission_id)
        if latest is not None:
            if sub.status == S.UPLOADED:
                self.storage.set_status(submission_id, S.EXTRACTED)
            who = f", corrected by {latest.corrected_by}" if latest.corrected_by else ""
            return StepResult("skipped", f"Reusing extraction v{latest.version} ({latest.engine}{who}); Gemini not called.",
                              model_id=latest.model_id, prompt_version=latest.prompt_version,
                              data={"extraction_version": latest.version})
        rubric = self._rubric(sub)
        try:
            extraction = extract_submission(
                submission_id, self.storage.submission_paths(sub), rubric.questions,
                agent=self.deps.document_agent, settings=self.settings, cache=self.deps.cache,
                use_cache=self.deps.use_cache, enhance_images=self.deps.enhance_images)
        except DocumentExtractionError as e:
            self.storage.set_status(submission_id, S.FAILED)
            raise StepFailed(f"Could not read the submission: {e}") from e
        saved = self.storage.save_extraction(extraction)
        self.storage.set_status(submission_id, S.EXTRACTED)
        status = "fallback" if saved.engine == "trocr" else "ok"
        return StepResult(status, f"Extraction v{saved.version} by {saved.engine}; {len(saved.answers)} answer(s), "
                                  f"missing {saved.missing_question_ids or 'none'}.",
                          model_id=saved.model_id, prompt_version=saved.prompt_version,
                          data={"extraction_version": saved.version})

    # --- 3. extraction gate ---------------------------------------------------------------------

    def extraction_gate(self, submission_id: str) -> StepResult:
        sub = self._submission(submission_id)
        extraction = self.storage.get_extraction(submission_id)
        rubric = self._rubric(sub)
        gate = extraction_gate(extraction, [q.question_id for q in rubric.questions],
                               self.settings.extraction_review_threshold)
        if not gate.ok:
            self.storage.set_status(submission_id, S.NEEDS_EXTRACTION_REVIEW)
            return StepResult("stopped", "Teacher must check the transcription: " + " ".join(gate.reasons),
                              stop=True, data={"gate_reasons": gate.reasons})
        if sub.status == S.NEEDS_EXTRACTION_REVIEW:  # e.g. threshold lowered since the last run
            self.storage.set_status(submission_id, S.EXTRACTED)
        return StepResult("ok", f"Extraction v{extraction.version} passed the gate.")

    # --- 4. evaluation (pre-compute + Agent 2 + validator) --------------------------------------

    def evaluation(self, submission_id: str) -> StepResult:
        sub = self._submission(submission_id)
        rubric = self._rubric(sub)
        extraction = self.storage.get_extraction(submission_id)
        context = self._context(submission_id, extraction, rubric)
        try:
            result = evaluate(extraction, rubric, context=context, agent=self.deps.evaluation_agent,
                              settings=self.settings, cache=self.deps.cache, use_cache=self.deps.use_cache)
        except EvaluationError as e:
            raise StepFailed(str(e)) from e  # status stays EXTRACTED; re-running retries from here
        saved = self.storage.save_evaluation(result)
        self.storage.set_status(submission_id, S.EVALUATED)
        flags = saved.all_flags
        return StepResult("ok", f"{saved.total:g}/{saved.max_total:g}" + (f"; validator flags {flags}" if flags else ""),
                          model_id=saved.model_id, prompt_version=saved.prompt_version,
                          data={"evaluation_id": saved.evaluation_id, "total": saved.total, "max_total": saved.max_total})

    # --- 5. judge (checks + Agent 3 + validator) ------------------------------------------------

    def judge(self, submission_id: str) -> StepResult:
        sub = self._submission(submission_id)
        rubric = self._rubric(sub)
        extraction = self.storage.get_extraction(submission_id)
        evaluation = self.storage.latest_evaluation(submission_id)
        if evaluation is None:
            raise StepFailed("No evaluation to review.")
        context = self._context(submission_id, extraction, rubric)
        status = "ok"
        try:
            report = judge(extraction, rubric, evaluation, context=context, agent=self.deps.judge_agent,
                           settings=self.settings, cache=self.deps.cache, use_cache=self.deps.use_cache)
        except JudgeError as e:
            if not self.deps.judge_fallback:
                raise StepFailed(str(e)) from e  # status stays EVALUATED
            log.warning("Judge unavailable for %s, using the checks-only report: %s", submission_id, e)
            report = checks_only_report(extraction, rubric, evaluation, settings=self.settings,
                                        reason=f"The Judge could not run ({str(e)[:200]}); no second opinion was taken.")
            status = "fallback"
        saved = self.storage.save_judge_report(report)
        changes = [f for f in saved.flags if f.code == "JUDGE_ADJUSTMENT"]
        return StepResult(status, f"recommends {saved.recommended_total:g}/{saved.max_total:g} "
                                  f"({len(changes)} adjustment(s), {len(saved.flags)} flag(s))",
                          model_id=saved.model_id, prompt_version=saved.prompt_version,
                          data={"judge_id": saved.judge_id, "recommended_total": saved.recommended_total})

    # --- 6. integrity post-check ----------------------------------------------------------------

    def integrity_postcheck(self, submission_id: str) -> StepResult:
        """Near-duplicate scan against the cohort arrives in Phase 7 (task 7.2/7.4). Flags only, never marks."""
        self._submission(submission_id)
        return StepResult("skipped", "Near-duplicate scan is added in Phase 7.")

    # --- 7. hand over to the teacher ------------------------------------------------------------

    def mark_pending_review(self, submission_id: str) -> StepResult:
        evaluation = self.storage.latest_evaluation(submission_id)
        report = self.storage.latest_judge_report(submission_id)
        if evaluation is None or report is None or report.evaluation_id != evaluation.evaluation_id:
            raise StepFailed("The latest evaluation has no judge report; cannot hand over to the teacher.")
        self.storage.set_status(submission_id, S.PENDING_TEACHER_REVIEW)
        return StepResult("ok", "Waiting for the teacher's approval.")

    def by_name(self, name: str):
        return getattr(self, name)

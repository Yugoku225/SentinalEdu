"""SQLite + local-file persistence (ARCHITECTURE §8, DECISIONS D-12).

Phase 3: submissions and versioned extractions. Phase 4: rubric versions and evaluation runs. Phase 5: judge reports. Phase 6: status guards, teacher decisions, audit log.
Uploaded originals and earlier extraction versions are never overwritten.
"""
import hashlib
import json
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

from sentineledu.config import Settings, get_settings
from sentineledu.schemas.evaluation import EvaluationResult
from sentineledu.schemas.extraction import ExtractedSubmission
from sentineledu.schemas.judge import JudgeReport
from sentineledu.schemas.review import TeacherDecision
from sentineledu.schemas.rubric import Rubric
from sentineledu.schemas.submission import Submission, SubmissionStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
    submission_id TEXT PRIMARY KEY,
    assignment_id TEXT NOT NULL,
    student_ref   TEXT,
    files         TEXT NOT NULL,          -- JSON list of paths relative to storage_dir
    file_sha256   TEXT NOT NULL,
    status        TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_submissions_assignment ON submissions(assignment_id);
CREATE INDEX IF NOT EXISTS idx_submissions_sha ON submissions(assignment_id, file_sha256);

CREATE TABLE IF NOT EXISTS extractions (
    submission_id TEXT NOT NULL REFERENCES submissions(submission_id),
    version       INTEGER NOT NULL,
    engine        TEXT NOT NULL,
    corrected_by  TEXT,
    payload       TEXT NOT NULL,          -- ExtractedSubmission JSON
    created_at    TEXT NOT NULL,
    PRIMARY KEY (submission_id, version)
);

CREATE TABLE IF NOT EXISTS rubrics (
    assignment_id TEXT NOT NULL,
    version       INTEGER NOT NULL,
    title         TEXT NOT NULL,
    payload       TEXT NOT NULL,          -- Rubric JSON
    created_at    TEXT NOT NULL,
    PRIMARY KEY (assignment_id, version)
);

CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id      TEXT PRIMARY KEY,
    submission_id      TEXT NOT NULL REFERENCES submissions(submission_id),
    extraction_version INTEGER NOT NULL,
    model_id           TEXT NOT NULL,
    prompt_version     TEXT NOT NULL,
    total              REAL NOT NULL,
    payload            TEXT NOT NULL,     -- EvaluationResult JSON
    created_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evaluations_submission ON evaluations(submission_id, created_at);

CREATE TABLE IF NOT EXISTS judge_reports (
    judge_id       TEXT PRIMARY KEY,
    submission_id  TEXT NOT NULL REFERENCES submissions(submission_id),
    evaluation_id  TEXT,
    model_id       TEXT NOT NULL,
    recommended_total REAL NOT NULL,
    payload        TEXT NOT NULL,         -- JudgeReport JSON
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_judge_submission ON judge_reports(submission_id, created_at);

CREATE TABLE IF NOT EXISTS teacher_decisions (
    submission_id TEXT PRIMARY KEY REFERENCES submissions(submission_id),
    reviewer      TEXT NOT NULL,
    total         REAL NOT NULL,
    judge_id      TEXT,
    evaluation_id TEXT,
    payload       TEXT NOT NULL,          -- TeacherDecision JSON
    approved_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id  TEXT NOT NULL,
    run_id         TEXT,
    step           TEXT NOT NULL,
    status         TEXT NOT NULL,         -- ok | skipped | stopped | error | fallback
    model_id       TEXT,
    prompt_version TEXT,
    duration_ms    INTEGER,
    detail         TEXT,
    error          TEXT,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_submission ON audit_log(submission_id, id);
"""


class NotFoundError(KeyError):
    pass


class IllegalTransitionError(ValueError):
    """A status change the lifecycle does not allow (ARCHITECTURE §3.1)."""


class FinalizationError(ValueError):
    """The teacher decision cannot finalise this submission (the message says why)."""


_S = SubmissionStatus
# FINALIZED is deliberately absent as a target: only Storage.finalize() can reach it.
ALLOWED_TRANSITIONS: dict[SubmissionStatus, frozenset[SubmissionStatus]] = {
    _S.UPLOADED: frozenset({_S.EXTRACTED, _S.FAILED}),
    _S.EXTRACTED: frozenset({_S.NEEDS_EXTRACTION_REVIEW, _S.EVALUATED}),
    _S.NEEDS_EXTRACTION_REVIEW: frozenset({_S.EXTRACTED}),
    _S.EVALUATED: frozenset({_S.PENDING_TEACHER_REVIEW, _S.EXTRACTED}),  # EXTRACTED: re-evaluate
    _S.PENDING_TEACHER_REVIEW: frozenset({_S.EXTRACTED}),  # teacher corrects the text or asks for re-evaluation
    _S.FAILED: frozenset({_S.UPLOADED}),  # retry
    _S.FINALIZED: frozenset(),
}


def sha256_files(paths: Sequence[Path]) -> str:
    """Hash of the files' bytes, in order. Same pages in the same order -> same hash."""
    outer = hashlib.sha256()
    for p in paths:
        outer.update(hashlib.sha256(Path(p).read_bytes()).digest())
    return outer.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Storage:
    def __init__(self, settings: Settings | None = None, *, storage_dir: Path | None = None):
        self.root = Path(storage_dir or (settings or get_settings()).storage_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "sentinel.db"
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def path(self, relative: str) -> Path:
        return self.root / relative

    # --- submissions -------------------------------------------------------

    def create_submission(self, assignment_id: str, files: Sequence[Path | str],
                          student_ref: str | None = None) -> Submission:
        """Copy the uploaded files into storage/uploads/<submission_id>/ and record the submission."""
        sources = [Path(f) for f in files]
        if not sources:
            raise ValueError("a submission needs at least one file")
        missing = [str(p) for p in sources if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"missing upload(s): {missing}")

        submission_id = uuid.uuid4().hex[:12]
        upload_dir = self.root / "uploads" / submission_id
        upload_dir.mkdir(parents=True)
        stored = []
        for page, src in enumerate(sources, start=1):
            dest = upload_dir / f"page{page:02d}_{src.name}"
            shutil.copyfile(src, dest)
            stored.append(str(dest.relative_to(self.root)))

        sub = Submission(submission_id=submission_id, assignment_id=assignment_id, student_ref=student_ref,
                         files=stored, file_sha256=sha256_files([self.path(f) for f in stored]))
        with self._conn() as c:
            c.execute(
                "INSERT INTO submissions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sub.submission_id, sub.assignment_id, sub.student_ref, json.dumps(sub.files),
                 sub.file_sha256, sub.status.value, sub.created_at.isoformat()),
            )
        return sub

    def get_submission(self, submission_id: str) -> Submission:
        with self._conn() as c:
            row = c.execute("SELECT * FROM submissions WHERE submission_id = ?", (submission_id,)).fetchone()
        if row is None:
            raise NotFoundError(submission_id)
        return Submission(**{**dict(row), "files": json.loads(row["files"])})

    def submission_paths(self, submission: Submission) -> list[Path]:
        return [self.path(f) for f in submission.files]

    def set_status(self, submission_id: str, status: SubmissionStatus) -> SubmissionStatus:
        """Move a submission to `status` if the lifecycle allows it; setting the current status is a no-op.

        Returns the previous status. FINALIZED can only be reached through finalize().
        """
        status = SubmissionStatus(status)
        current = self.get_submission(submission_id).status  # raises NotFoundError
        if status == current:
            return current
        if status == SubmissionStatus.FINALIZED:
            raise IllegalTransitionError("FINALIZED can only be set by finalize() with a TeacherDecision")
        if status not in ALLOWED_TRANSITIONS[current]:
            raise IllegalTransitionError(f"{submission_id}: {current.value} -> {status.value} is not allowed")
        with self._conn() as c:
            cur = c.execute("UPDATE submissions SET status = ? WHERE submission_id = ? AND status = ?",
                            (status.value, submission_id, current.value))
        if cur.rowcount != 1:
            raise IllegalTransitionError(f"{submission_id} changed status concurrently; retry")
        return current

    def list_submissions(self, assignment_id: str | None = None) -> list[Submission]:
        with self._conn() as c:
            if assignment_id is None:
                rows = c.execute("SELECT * FROM submissions ORDER BY created_at").fetchall()
            else:
                rows = c.execute("SELECT * FROM submissions WHERE assignment_id = ? ORDER BY created_at",
                                 (assignment_id,)).fetchall()
        return [Submission(**{**dict(r), "files": json.loads(r["files"])}) for r in rows]

    # --- extractions -------------------------------------------------------

    def save_extraction(self, extraction: ExtractedSubmission) -> ExtractedSubmission:
        """Store as the next version for its submission and return it with that version number."""
        self.get_submission(extraction.submission_id)  # raises if unknown
        with self._conn() as c:
            (latest,) = c.execute("SELECT COALESCE(MAX(version), 0) FROM extractions WHERE submission_id = ?",
                                  (extraction.submission_id,)).fetchone()
            saved = extraction.model_copy(update={"version": latest + 1})
            c.execute(
                "INSERT INTO extractions VALUES (?, ?, ?, ?, ?, ?)",
                (saved.submission_id, saved.version, saved.engine, saved.corrected_by,
                 saved.model_dump_json(), _now()),
            )
        return saved

    def get_extraction(self, submission_id: str, version: int | None = None) -> ExtractedSubmission:
        """A specific version, or the latest one if `version` is None."""
        with self._conn() as c:
            if version is None:
                row = c.execute("SELECT payload FROM extractions WHERE submission_id = ? "
                                "ORDER BY version DESC LIMIT 1", (submission_id,)).fetchone()
            else:
                row = c.execute("SELECT payload FROM extractions WHERE submission_id = ? AND version = ?",
                                (submission_id, version)).fetchone()
        if row is None:
            raise NotFoundError(f"{submission_id} v{version or 'latest'}")
        return ExtractedSubmission.model_validate_json(row["payload"])

    def latest_extraction(self, submission_id: str) -> ExtractedSubmission | None:
        try:
            return self.get_extraction(submission_id)
        except NotFoundError:
            return None

    def list_extraction_versions(self, submission_id: str) -> list[int]:
        with self._conn() as c:
            rows = c.execute("SELECT version FROM extractions WHERE submission_id = ? ORDER BY version",
                             (submission_id,)).fetchall()
        return [r["version"] for r in rows]

    # --- rubrics -----------------------------------------------------------

    def save_rubric(self, rubric: Rubric) -> Rubric:
        """Store as the next version for its assignment and return it with that version number."""
        with self._conn() as c:
            (latest,) = c.execute("SELECT COALESCE(MAX(version), 0) FROM rubrics WHERE assignment_id = ?",
                                  (rubric.assignment_id,)).fetchone()
            saved = rubric.model_copy(update={"version": latest + 1})
            c.execute("INSERT INTO rubrics VALUES (?, ?, ?, ?, ?)",
                      (saved.assignment_id, saved.version, saved.title, saved.model_dump_json(), _now()))
        return saved

    def get_rubric(self, assignment_id: str, version: int | None = None) -> Rubric:
        with self._conn() as c:
            if version is None:
                row = c.execute("SELECT payload FROM rubrics WHERE assignment_id = ? ORDER BY version DESC LIMIT 1",
                                (assignment_id,)).fetchone()
            else:
                row = c.execute("SELECT payload FROM rubrics WHERE assignment_id = ? AND version = ?",
                                (assignment_id, version)).fetchone()
        if row is None:
            raise NotFoundError(f"rubric {assignment_id} v{version or 'latest'}")
        return Rubric.model_validate_json(row["payload"])

    def list_assignments(self) -> list[str]:
        with self._conn() as c:
            rows = c.execute("SELECT DISTINCT assignment_id FROM rubrics ORDER BY assignment_id").fetchall()
        return [r["assignment_id"] for r in rows]

    # --- evaluations -------------------------------------------------------

    def save_evaluation(self, evaluation: EvaluationResult) -> EvaluationResult:
        """Append a grading run (never overwrites earlier runs) and return it with its evaluation_id."""
        self.get_submission(evaluation.submission_id)  # raises if unknown
        saved = evaluation.model_copy(update={"evaluation_id": uuid.uuid4().hex[:12]})
        with self._conn() as c:
            c.execute(
                "INSERT INTO evaluations VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (saved.evaluation_id, saved.submission_id, saved.extraction_version, saved.model_id,
                 saved.prompt_version, saved.total, saved.model_dump_json(), _now()),
            )
        return saved

    def get_evaluation(self, evaluation_id: str) -> EvaluationResult:
        with self._conn() as c:
            row = c.execute("SELECT payload FROM evaluations WHERE evaluation_id = ?", (evaluation_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"evaluation {evaluation_id}")
        return EvaluationResult.model_validate_json(row["payload"])

    def latest_evaluation(self, submission_id: str) -> EvaluationResult | None:
        with self._conn() as c:
            row = c.execute("SELECT payload FROM evaluations WHERE submission_id = ? "
                            "ORDER BY created_at DESC, rowid DESC LIMIT 1", (submission_id,)).fetchone()
        return EvaluationResult.model_validate_json(row["payload"]) if row else None

    def list_evaluations(self, submission_id: str) -> list[EvaluationResult]:
        with self._conn() as c:
            rows = c.execute("SELECT payload FROM evaluations WHERE submission_id = ? ORDER BY created_at, rowid",
                             (submission_id,)).fetchall()
        return [EvaluationResult.model_validate_json(r["payload"]) for r in rows]

    # --- judge reports -----------------------------------------------------

    def save_judge_report(self, report: JudgeReport) -> JudgeReport:
        self.get_submission(report.submission_id)  # raises if unknown
        saved = report.model_copy(update={"judge_id": uuid.uuid4().hex[:12]})
        with self._conn() as c:
            c.execute(
                "INSERT INTO judge_reports VALUES (?, ?, ?, ?, ?, ?, ?)",
                (saved.judge_id, saved.submission_id, saved.evaluation_id, saved.model_id,
                 saved.recommended_total, saved.model_dump_json(), _now()),
            )
        return saved

    def get_judge_report(self, judge_id: str) -> JudgeReport:
        with self._conn() as c:
            row = c.execute("SELECT payload FROM judge_reports WHERE judge_id = ?", (judge_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"judge report {judge_id}")
        return JudgeReport.model_validate_json(row["payload"])

    def latest_judge_report(self, submission_id: str, evaluation_id: str | None = None) -> JudgeReport | None:
        """Latest report for the submission (optionally only for a given evaluation run)."""
        with self._conn() as c:
            if evaluation_id is None:
                row = c.execute("SELECT payload FROM judge_reports WHERE submission_id = ? "
                                "ORDER BY created_at DESC, rowid DESC LIMIT 1", (submission_id,)).fetchone()
            else:
                row = c.execute("SELECT payload FROM judge_reports WHERE submission_id = ? AND evaluation_id = ? "
                                "ORDER BY created_at DESC, rowid DESC LIMIT 1", (submission_id, evaluation_id)).fetchone()
        return JudgeReport.model_validate_json(row["payload"]) if row else None

    # --- teacher decisions (the only way to FINALIZED) ----------------------

    def recommended_marks(self, submission_id: str) -> dict[str, dict[str, float]]:
        """The marks the teacher's form starts from: the Judge's recommendation, else Agent 2's marks."""
        report = self.latest_judge_report(submission_id)
        if report is not None:
            return report.recommended_marks()
        evaluation = self.latest_evaluation(submission_id)
        if evaluation is None:
            raise NotFoundError(f"{submission_id} has no evaluation")
        return {q.question_id: {c.criterion_id: c.marks for c in q.criteria} for q in evaluation.questions}

    def finalize(self, decision: TeacherDecision) -> TeacherDecision:
        """Record the teacher's decision and mark the submission FINALIZED.

        Refused (FinalizationError) unless: the submission is PENDING_TEACHER_REVIEW; final marks
        cover exactly the rubric's criteria and stay within each criterion's maximum; and every mark
        that differs from the recommendation has a matching override. TeacherDecision itself
        requires a reviewer name and a reason on every override.
        """
        sub = self.get_submission(decision.submission_id)
        if sub.status != SubmissionStatus.PENDING_TEACHER_REVIEW:
            raise FinalizationError(f"{sub.submission_id} is {sub.status.value}; only PENDING_TEACHER_REVIEW can be finalised")
        try:
            rubric = self.get_rubric(sub.assignment_id)
        except NotFoundError:
            raise FinalizationError(f"no rubric stored for assignment {sub.assignment_id}") from None

        expected = {(q.question_id, c.criterion_id): c.max_marks for q in rubric.questions for c in q.criteria}
        given = {(q, c): m for q, crits in decision.final_marks.items() for c, m in crits.items()}
        if set(given) != set(expected):
            missing = sorted(f"{q}/{c}" for q, c in set(expected) - set(given))
            extra = sorted(f"{q}/{c}" for q, c in set(given) - set(expected))
            raise FinalizationError(f"final marks must cover the rubric exactly (missing {missing}, unknown {extra})")
        over = sorted(f"{q}/{c}" for (q, c), m in given.items() if m > expected[(q, c)] + 1e-6)
        if over:
            raise FinalizationError(f"marks above the criterion maximum: {over}")

        try:
            recommended = self.recommended_marks(sub.submission_id)
        except NotFoundError:
            raise FinalizationError(f"{sub.submission_id} has no evaluation to review") from None
        overrides = {(o.question_id, o.criterion_id): o for o in decision.overrides}
        for (q, c), m in sorted(given.items()):
            rec = recommended.get(q, {}).get(c, 0.0)
            if abs(m - rec) <= 1e-6:
                continue
            o = overrides.get((q, c))
            if o is None:
                raise FinalizationError(f"{q}/{c}: {m:g} differs from the recommended {rec:g}; an override with a reason is required")
            if abs(o.to_marks - m) > 1e-6 or abs(o.from_marks - rec) > 1e-6:
                raise FinalizationError(f"{q}/{c}: the override says {o.from_marks:g} -> {o.to_marks:g} but the change is {rec:g} -> {m:g}")

        report = self.latest_judge_report(sub.submission_id)
        evaluation = self.latest_evaluation(sub.submission_id)
        with self._conn() as c:
            cur = c.execute("UPDATE submissions SET status = ? WHERE submission_id = ? AND status = ?",
                            (SubmissionStatus.FINALIZED.value, sub.submission_id, SubmissionStatus.PENDING_TEACHER_REVIEW.value))
            if cur.rowcount != 1:
                raise FinalizationError(f"{sub.submission_id} changed status while finalising")
            c.execute("INSERT INTO teacher_decisions VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (sub.submission_id, decision.reviewer, decision.total,
                       report.judge_id if report else None, evaluation.evaluation_id if evaluation else None,
                       decision.model_dump_json(), decision.approved_at.isoformat()))
        return decision

    def get_decision(self, submission_id: str) -> TeacherDecision | None:
        with self._conn() as c:
            row = c.execute("SELECT payload FROM teacher_decisions WHERE submission_id = ?", (submission_id,)).fetchone()
        return TeacherDecision.model_validate_json(row["payload"]) if row else None

    # --- audit log -----------------------------------------------------------

    def log_step(self, submission_id: str, step: str, status: str, *, run_id: str | None = None,
                 model_id: str | None = None, prompt_version: str | None = None, duration_ms: int | None = None,
                 detail: str | None = None, error: str | None = None) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO audit_log (submission_id, run_id, step, status, model_id, prompt_version, "
                      "duration_ms, detail, error, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (submission_id, run_id, step, status, model_id, prompt_version, duration_ms,
                       detail, error[:2000] if error else None, _now()))

    def audit_log(self, submission_id: str, run_id: str | None = None) -> list[dict]:
        with self._conn() as c:
            if run_id is None:
                rows = c.execute("SELECT * FROM audit_log WHERE submission_id = ? ORDER BY id", (submission_id,)).fetchall()
            else:
                rows = c.execute("SELECT * FROM audit_log WHERE submission_id = ? AND run_id = ? ORDER BY id",
                                 (submission_id, run_id)).fetchall()
        return [dict(r) for r in rows]

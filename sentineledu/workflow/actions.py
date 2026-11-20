"""Teacher actions around the workflow (ARCHITECTURE §3.1, DECISIONS D-15).

These are what the dashboard (Phase 8) calls. None of them runs an agent:
- `submit_correction`: save the teacher's corrected transcription as a new extraction version and
  move the submission back to EXTRACTED, so the next workflow run resumes at evaluation.
- `request_reevaluation`: send a graded submission back to EXTRACTED (history is kept).
- `approve`: build a TeacherDecision from the recommendation plus the teacher's changes and
  finalise through `Storage.finalize`, which enforces the rules.
"""
from typing import Mapping, Sequence

from sentineledu.schemas.extraction import ExtractedSubmission
from sentineledu.schemas.review import MarkOverride, TeacherDecision
from sentineledu.schemas.submission import SubmissionStatus
from sentineledu.services.extraction_review import AnswerEdit, apply_correction
from sentineledu.services.storage import IllegalTransitionError, Storage

S = SubmissionStatus
_CORRECTABLE = {S.EXTRACTED, S.NEEDS_EXTRACTION_REVIEW, S.EVALUATED, S.PENDING_TEACHER_REVIEW}


def submit_correction(storage: Storage, submission_id: str, edits: Sequence[AnswerEdit], teacher: str, *,
                      unmatched_text: str | None = None) -> ExtractedSubmission:
    sub = storage.get_submission(submission_id)
    if sub.status not in _CORRECTABLE:
        raise IllegalTransitionError(f"{submission_id} is {sub.status.value}; its transcription cannot be corrected now")
    corrected = apply_correction(storage.get_extraction(submission_id), edits, teacher, unmatched_text=unmatched_text)
    saved = storage.save_extraction(corrected)
    storage.set_status(submission_id, S.EXTRACTED)
    return saved


def request_reevaluation(storage: Storage, submission_id: str) -> None:
    """Send a graded submission back for a new evaluation run (earlier runs are kept)."""
    status = storage.get_submission(submission_id).status
    if status not in (S.EVALUATED, S.PENDING_TEACHER_REVIEW):
        raise IllegalTransitionError(f"{submission_id} is {status.value}; only a graded submission can be re-evaluated")
    storage.set_status(submission_id, S.EXTRACTED)


def approve(storage: Storage, submission_id: str, reviewer: str, *,
            changes: Mapping[tuple[str, str], tuple[float, str]] | None = None,
            teacher_comment: str = "") -> TeacherDecision:
    """Finalise with the recommended marks, except `changes`: {(question_id, criterion_id): (marks, reason)}."""
    recommended = storage.recommended_marks(submission_id)
    final = {q: dict(crits) for q, crits in recommended.items()}
    overrides = []
    for (qid, cid), (marks, reason) in (changes or {}).items():
        before = recommended.get(qid, {}).get(cid, 0.0)
        final.setdefault(qid, {})[cid] = marks
        if abs(marks - before) > 1e-6:
            overrides.append(MarkOverride(question_id=qid, criterion_id=cid, from_marks=before, to_marks=marks, reason=reason))
    decision = TeacherDecision(submission_id=submission_id, reviewer=reviewer, final_marks=final,
                               overrides=overrides, teacher_comment=teacher_comment)
    return storage.finalize(decision)

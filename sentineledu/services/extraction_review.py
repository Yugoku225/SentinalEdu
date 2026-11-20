"""Extraction gate and teacher corrections (ARCHITECTURE §4.1, SENTINELEDU_TASKS.md 3.6–3.7)."""
from dataclasses import dataclass, field
from typing import Sequence

from pydantic import BaseModel

from sentineledu.schemas.extraction import ILLEGIBLE_MARKER, ExtractedAnswer, ExtractedSubmission


@dataclass
class GateResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def extraction_gate(extraction: ExtractedSubmission, expected_question_ids: Sequence[str],
                    threshold: float) -> GateResult:
    """Decide whether the extraction can go straight to grading or needs the teacher first.

    A teacher-corrected extraction always passes: the teacher has already reviewed it.
    """
    if extraction.engine == "teacher":
        return GateResult(ok=True)

    reasons = []
    if extraction.engine == "trocr":
        reasons.append("Read by the offline TrOCR fallback: answers must be assigned to questions.")
    for a in extraction.answers:
        if a.legibility == "unreadable":
            reasons.append(f"{a.question_id}: answer is unreadable.")
        elif a.legibility != "blank" and a.confidence < threshold:
            reasons.append(f"{a.question_id}: low extraction confidence ({a.confidence:.2f} < {threshold:.2f}).")
    found = {a.question_id for a in extraction.answers}
    missing = sorted(set(extraction.missing_question_ids) | (set(expected_question_ids) - found))
    if missing:
        reasons.append(f"Question(s) not found on the pages: {', '.join(missing)}.")
    if extraction.unmatched_text.strip():
        reasons.append("Some writing could not be matched to a question.")
    return GateResult(ok=not reasons, reasons=reasons)


class AnswerEdit(BaseModel):
    question_id: str
    answer_text: str


def _legibility_after_edit(text: str) -> str:
    if not text.strip():
        return "blank"
    return "partial" if ILLEGIBLE_MARKER in text else "clear"


def apply_correction(extraction: ExtractedSubmission, edits: Sequence[AnswerEdit], teacher: str, *,
                     unmatched_text: str | None = None) -> ExtractedSubmission:
    """Return a new extraction (next version, engine "teacher") with the teacher's edits applied.

    The original object is not modified; storage keeps every version.
    Edited answers get confidence 1.0 because the teacher has verified them. Editing a question
    that was missing adds it.
    """
    if not teacher.strip():
        raise ValueError("teacher name is required to save a correction")
    answers = {a.question_id: a for a in extraction.answers}
    edited_ids = []
    for edit in edits:
        prev = answers.get(edit.question_id)
        base = prev.model_dump() if prev else {"question_id": edit.question_id}
        answers[edit.question_id] = ExtractedAnswer.model_validate({
            **base,
            "answer_text": edit.answer_text.strip(),
            "legibility": _legibility_after_edit(edit.answer_text),
            "confidence": 1.0,
        })
        edited_ids.append(edit.question_id)

    order = [a.question_id for a in extraction.answers] + [q for q in edited_ids if q not in
                                                           {a.question_id for a in extraction.answers}]
    issues = list(extraction.issues)
    if edited_ids:
        issues.append(f"Teacher {teacher.strip()} corrected: {', '.join(edited_ids)}.")
    return extraction.model_copy(update={
        "engine": "teacher",
        "answers": [answers[q] for q in order],
        "unmatched_text": extraction.unmatched_text if unmatched_text is None else unmatched_text,
        "missing_question_ids": [q for q in extraction.missing_question_ids if q not in edited_ids],
        "issues": issues,
        "version": extraction.version + 1,
        "corrected_by": teacher.strip(),
    }, deep=True)

"""Rubric loading, validation and the legacy-compatible one-criterion rubric (TASKS 4.2, D-06).

Validation lives in the Pydantic models (schemas/rubric.py). This module turns their errors into
a message a teacher can act on, reads and writes rubric JSON files, and builds a default
full/partial/none rubric from a plain model answer so legacy-style answer keys still work.
"""
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sentineledu.schemas.rubric import Criterion, Question, Rubric, RubricLevel

DEFAULT_LEVELS = ("full", "partial", "none")


class RubricError(ValueError):
    """The rubric is not usable. The message lists every problem found."""


def _readable(err: ValidationError) -> str:
    lines = []
    for e in err.errors():
        where = ".".join(str(p) for p in e["loc"]) or "rubric"
        msg = e["msg"]
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
        lines.append(f"- {where}: {msg}")
    return "Invalid rubric:\n" + "\n".join(lines)


def validate_rubric(data: Rubric | dict[str, Any] | str | bytes) -> Rubric:
    """Return a validated Rubric or raise RubricError with a readable list of problems."""
    if isinstance(data, Rubric):
        return data
    try:
        if isinstance(data, (str, bytes)):
            return Rubric.model_validate_json(data)
        return Rubric.model_validate(data)
    except ValidationError as e:
        raise RubricError(_readable(e)) from None
    except ValueError as e:  # malformed JSON
        raise RubricError(f"Invalid rubric: {e}") from None


def load_rubric(path: Path | str) -> Rubric:
    path = Path(path)
    if not path.is_file():
        raise RubricError(f"Rubric file not found: {path}")
    return validate_rubric(path.read_text(encoding="utf-8"))


def save_rubric(rubric: Rubric, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rubric.model_dump_json(indent=2), encoding="utf-8")
    return path


def _round_half(x: float) -> float:
    """Nearest 0.5, so a 7-mark question gets a 3.5-mark partial level."""
    return round(x * 2) / 2


def from_simple_model_answer(
    model_answer: str,
    max_marks: float,
    *,
    assignment_id: str = "legacy",
    question_id: str = "Q1",
    prompt: str = "",
    title: str | None = None,
) -> Rubric:
    """A one-criterion, three-level rubric (full / partial / none) for a plain model answer.

    This is how legacy answer keys (`data/model_answers.json`) become rubrics without the teacher
    writing criteria. The LLM still only picks a level; marks come from the levels.
    """
    if not model_answer.strip():
        raise RubricError("Invalid rubric: the model answer is empty")
    if max_marks <= 0:
        raise RubricError("Invalid rubric: max_marks must be positive")
    partial = _round_half(max_marks / 2) or max_marks / 2
    levels = [
        RubricLevel(level_id="full", label="Full marks", marks=max_marks,
                    descriptor="Covers every key point of the model answer correctly."),
        RubricLevel(level_id="partial", label="Partial marks", marks=partial,
                    descriptor="Covers roughly half of the key points, or has a minor error."),
        RubricLevel(level_id="none", label="No marks", marks=0,
                    descriptor="Blank, off-topic, or none of the key points are present."),
    ]
    criterion = Criterion(criterion_id="C1", description="Agreement with the model answer",
                          max_marks=max_marks, levels=levels)
    question = Question(question_id=question_id, prompt=prompt or f"Question {question_id}",
                        max_marks=max_marks, model_answer=model_answer.strip(), criteria=[criterion])
    return Rubric(assignment_id=assignment_id, title=title or f"Assignment {assignment_id}", questions=[question])


def rubric_from_legacy_keys(keys: dict[str, str], max_marks: float = 10, *, assignment_id: str = "legacy") -> Rubric:
    """Turn a `data/model_answers.json` style mapping (name -> model answer) into one rubric."""
    questions = [
        from_simple_model_answer(text, max_marks, assignment_id=assignment_id, question_id=name).questions[0]
        for name, text in keys.items()
    ]
    if not questions:
        raise RubricError("Invalid rubric: no model answers given")
    return Rubric(assignment_id=assignment_id, title=f"Assignment {assignment_id}", questions=questions)


def rubric_summary(rubric: Rubric) -> str:
    """One line per criterion, for logs and the dashboard."""
    lines = [f"{rubric.title} ({rubric.assignment_id}, v{rubric.version}) — {rubric.max_total:g} marks"]
    for q in rubric.questions:
        lines.append(f"  {q.question_id} [{q.max_marks:g}]: {q.prompt}")
        for c in q.criteria:
            levels = ", ".join(f"{lv.level_id}={lv.marks:g}" for lv in c.levels)
            lines.append(f"    {c.criterion_id} [{c.max_marks:g}]: {c.description} ({levels})")
    return "\n".join(lines)


def load_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

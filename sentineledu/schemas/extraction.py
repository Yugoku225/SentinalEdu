"""Output of Agent 1 (Document Understanding)."""
from typing import Literal

from pydantic import BaseModel, Field

ILLEGIBLE_MARKER = "[[ILLEGIBLE]]"
UNKNOWN_QUESTION_ID = "UNKNOWN"

Legibility = Literal["clear", "partial", "unreadable", "blank"]
Engine = Literal["gemini", "trocr", "pdf_text", "teacher"]


class ExtractedAnswer(BaseModel):
    question_id: str  # matches Rubric.questions[].question_id, or UNKNOWN_QUESTION_ID
    question_text_detected: str | None = None
    answer_text: str  # verbatim; illegible spans replaced by ILLEGIBLE_MARKER
    legibility: Legibility
    confidence: float = Field(ge=0, le=1)
    contains_diagram: bool = False
    contains_math: bool = False
    page_refs: list[int] = []


class ExtractedSubmission(BaseModel):
    submission_id: str
    engine: Engine
    answers: list[ExtractedAnswer]
    unmatched_text: str = ""
    missing_question_ids: list[str] = []
    issues: list[str] = []
    version: int = Field(1, ge=1)
    corrected_by: str | None = None
    model_id: str | None = None  # Gemini model that produced it (None for trocr/teacher)
    prompt_version: str | None = None

    def answer_for(self, question_id: str) -> ExtractedAnswer | None:
        return next((a for a in self.answers if a.question_id == question_id), None)

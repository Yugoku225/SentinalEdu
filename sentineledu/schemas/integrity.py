"""Academic-integrity flags (ARCHITECTURE §5.1, DECISIONS D-11). Informational only: they never change marks."""
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

IntegrityCode = Literal["EXACT_DUPLICATE", "TEXT_DUPLICATE", "NEAR_DUPLICATE", "SIMILAR"]


class IntegrityFlag(BaseModel):
    assignment_id: str
    submission_id: str  # the submission this row is shown on
    other_submission_id: str  # the submission it resembles (each pair is stored once per side)
    code: IntegrityCode
    severity: Literal["info", "review", "high"]
    question_id: str | None = None  # None for whole-submission checks
    similarity: float | None = None
    margin: float | None = None  # similarity minus the closer of the two answers' similarity to the reference
    message: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def mirrored(self) -> "IntegrityFlag":
        return self.model_copy(update={"submission_id": self.other_submission_id,
                                        "other_submission_id": self.submission_id})

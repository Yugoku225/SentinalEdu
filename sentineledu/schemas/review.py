"""The teacher's decision. This is the only object that can finalise marks."""
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


class MarkOverride(BaseModel):
    question_id: str
    criterion_id: str
    from_marks: float = Field(ge=0)
    to_marks: float = Field(ge=0)
    reason: str

    @field_validator("reason")
    @classmethod
    def _reason_required(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("every mark override needs a reason")
        return v.strip()


class TeacherDecision(BaseModel):
    submission_id: str
    reviewer: str
    final_marks: dict[str, dict[str, float]]  # question_id -> criterion_id -> marks
    overrides: list[MarkOverride] = []
    teacher_comment: str = ""
    approved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("reviewer")
    @classmethod
    def _reviewer_required(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reviewer name is required to finalise marks")
        return v.strip()

    @field_validator("final_marks")
    @classmethod
    def _non_negative(cls, v: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        for qid, crits in v.items():
            for cid, m in crits.items():
                if m < 0:
                    raise ValueError(f"negative marks for {qid}/{cid}")
        return v

    @property
    def total(self) -> float:
        return sum(m for crits in self.final_marks.values() for m in crits.values())

"""Teacher-defined rubric: questions, criteria and discrete marking levels."""
from pydantic import BaseModel, Field, model_validator

_EPS = 1e-6


def _duplicates(ids: list[str]) -> list[str]:
    return sorted({i for i in ids if ids.count(i) > 1})


class RubricLevel(BaseModel):
    level_id: str = Field(min_length=1)  # e.g. "full", "partial", "none"
    label: str
    marks: float = Field(ge=0)
    descriptor: str


class Criterion(BaseModel):
    criterion_id: str = Field(min_length=1)
    description: str
    max_marks: float = Field(gt=0)
    key_concepts: list[str] = []
    levels: list[RubricLevel] = Field(min_length=2)

    @model_validator(mode="after")
    def _check_levels(self) -> "Criterion":
        dup = _duplicates([lv.level_id for lv in self.levels])
        if dup:
            raise ValueError(f"criterion {self.criterion_id}: duplicate level_id {dup}")
        marks = [lv.marks for lv in self.levels]
        if min(marks) > _EPS:
            raise ValueError(f"criterion {self.criterion_id}: needs a 0-mark level")
        if abs(max(marks) - self.max_marks) > _EPS:
            raise ValueError(
                f"criterion {self.criterion_id}: highest level is worth {max(marks)} "
                f"but max_marks is {self.max_marks}"
            )
        return self

    def level(self, level_id: str) -> RubricLevel | None:
        return next((lv for lv in self.levels if lv.level_id == level_id), None)

    @property
    def lowest_level(self) -> RubricLevel:
        return min(self.levels, key=lambda lv: lv.marks)


class Question(BaseModel):
    question_id: str = Field(min_length=1)  # "Q1", "Q2a", ...
    prompt: str
    max_marks: float = Field(gt=0)
    model_answer: str
    criteria: list[Criterion] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_criteria(self) -> "Question":
        dup = _duplicates([c.criterion_id for c in self.criteria])
        if dup:
            raise ValueError(f"question {self.question_id}: duplicate criterion_id {dup}")
        total = sum(c.max_marks for c in self.criteria)
        if abs(total - self.max_marks) > _EPS:
            raise ValueError(
                f"question {self.question_id}: criteria add up to {total} but max_marks is {self.max_marks}"
            )
        return self

    def criterion(self, criterion_id: str) -> Criterion | None:
        return next((c for c in self.criteria if c.criterion_id == criterion_id), None)


class Rubric(BaseModel):
    assignment_id: str = Field(min_length=1)
    title: str
    questions: list[Question] = Field(min_length=1)
    version: int = Field(1, ge=1)

    @model_validator(mode="after")
    def _check_questions(self) -> "Rubric":
        dup = _duplicates([q.question_id for q in self.questions])
        if dup:
            raise ValueError(f"duplicate question_id {dup}")
        return self

    @property
    def max_total(self) -> float:
        return sum(q.max_marks for q in self.questions)

    def question(self, question_id: str) -> Question | None:
        return next((q for q in self.questions if q.question_id == question_id), None)

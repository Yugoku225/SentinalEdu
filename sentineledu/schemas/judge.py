"""Output of Agent 3 (Judge & Feedback).

`JudgeDraft` is what the LLM returns: verdicts that may *propose* a rubric level, feedback and
flags, never a number. `JudgeReport` is the validated form: `accepted`, `recommended_marks` and
`recommended_total` are set by services/scoring_rules.validate_judge only (DECISIONS D-06).
The Judge never finalises anything; its recommendation is the default in the teacher's form.
"""
from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["info", "review", "high"]
Verdict = Literal["agree", "propose_adjustment", "flag"]


class Flag(BaseModel):
    code: str  # e.g. SBERT_LEVEL_DISAGREEMENT, NEAR_DUPLICATE
    severity: Severity
    question_id: str | None = None
    criterion_id: str | None = None
    message: str


# --- what the LLM returns ------------------------------------------------------------------

class VerdictDraft(BaseModel):
    criterion_id: str
    verdict: Verdict
    proposed_level_id: str | None = None  # only with verdict == "propose_adjustment"
    justification: str = ""
    evidence_quotes: list[str] = []  # exact spans from the student's answer


class QuestionJudgeDraft(BaseModel):
    question_id: str
    verdicts: list[VerdictDraft] = []
    strengths: list[str] = []
    improvements: list[str] = []
    next_steps: list[str] = []
    overlooked_concepts: list[str] = []  # concepts the evaluator missed (present or absent)


class FlagDraft(BaseModel):
    code: str
    severity: Severity = "review"
    question_id: str | None = None
    criterion_id: str | None = None
    message: str


class JudgeDraft(BaseModel):
    questions: list[QuestionJudgeDraft] = []
    summary_feedback: str = ""
    flags: list[FlagDraft] = []

    def question(self, question_id: str) -> QuestionJudgeDraft | None:
        return next((q for q in self.questions if q.question_id == question_id), None)


# --- after the validator -------------------------------------------------------------------

class CriterionVerdict(BaseModel):
    criterion_id: str
    verdict: Verdict
    proposed_level_id: str | None = None
    justification: str
    evidence_quotes: list[str] = []
    accepted: bool = False  # set by the validator only: True when a proposal was applied
    recommended_level_id: str = ""  # set by the validator only
    recommended_marks: float = Field(ge=0)  # set by the validator only


class QuestionFeedback(BaseModel):
    question_id: str
    strengths: list[str] = []
    improvements: list[str] = []
    next_steps: list[str] = []
    overlooked_concepts: list[str] = []


class JudgeReport(BaseModel):
    submission_id: str
    verdicts: dict[str, list[CriterionVerdict]]  # question_id -> verdicts
    feedback: list[QuestionFeedback]
    summary_feedback: str
    flags: list[Flag] = []
    recommended_total: float = Field(ge=0)
    max_total: float = Field(gt=0)
    evaluation_total: float = Field(0, ge=0)  # Agent 2's total, for a quick comparison
    prompt_version: str = ""
    model_id: str = "none"  # "none" when the LLM was not used (checks-only report)
    evaluation_id: str | None = None
    judge_id: str | None = None  # set by storage

    def feedback_for(self, question_id: str) -> QuestionFeedback | None:
        return next((f for f in self.feedback if f.question_id == question_id), None)

    def recommended_marks(self) -> dict[str, dict[str, float]]:
        """question_id -> criterion_id -> marks, the shape TeacherDecision.final_marks uses."""
        return {qid: {v.criterion_id: v.recommended_marks for v in vs} for qid, vs in self.verdicts.items()}

    def flags_with(self, code: str) -> list[Flag]:
        return [f for f in self.flags if f.code == code]

"""Output of Agent 2 (Answer Evaluation).

`EvaluationDraft` is what the LLM returns: a rubric *level* per criterion plus quotes and
citations, never a number. `EvaluationResult` is the validated form: `marks`,
`evidence_verified` and the totals are set by services/scoring_rules.py only
(SENTINELEDU_DECISIONS.md D-06).
"""
from pydantic import BaseModel, Field


# --- what the LLM returns ------------------------------------------------------------------

class CriterionDraft(BaseModel):
    criterion_id: str
    selected_level_id: str
    evidence_quotes: list[str] = []  # exact spans copied from the student's answer
    reference_chunk_ids: list[str] = []  # ids of the provided reference chunks that support the decision
    concepts_present: list[str] = []
    missing_concepts: list[str] = []
    incorrect_statements: list[str] = []
    rationale: str = ""


class QuestionDraft(BaseModel):
    question_id: str
    criteria: list[CriterionDraft] = []


class EvaluationDraft(BaseModel):
    questions: list[QuestionDraft] = []

    def question(self, question_id: str) -> QuestionDraft | None:
        return next((q for q in self.questions if q.question_id == question_id), None)


# --- after the validator -------------------------------------------------------------------

class CriterionEvaluation(BaseModel):
    criterion_id: str
    selected_level_id: str
    marks: float = Field(ge=0)  # set by the validator only
    evidence_quotes: list[str] = []
    evidence_verified: bool = False  # set by the validator only
    reference_chunk_ids: list[str] = []
    concepts_present: list[str] = []
    missing_concepts: list[str] = []
    incorrect_statements: list[str] = []
    rationale: str = ""


class QuestionEvaluation(BaseModel):
    question_id: str
    criteria: list[CriterionEvaluation]
    total: float = Field(ge=0)
    max_marks: float = Field(gt=0)
    sbert_sim_model_answer: float
    sbert_sim_best_chunk: float
    validator_flags: list[str] = []

    def criterion(self, criterion_id: str) -> CriterionEvaluation | None:
        return next((c for c in self.criteria if c.criterion_id == criterion_id), None)


class EvaluationResult(BaseModel):
    submission_id: str
    questions: list[QuestionEvaluation]
    total: float = Field(ge=0)
    max_total: float = Field(gt=0)
    prompt_version: str
    model_id: str
    extraction_version: int = Field(1, ge=1)  # which ExtractedSubmission version was graded
    evaluation_id: str | None = None  # set by storage

    def question(self, question_id: str) -> QuestionEvaluation | None:
        return next((q for q in self.questions if q.question_id == question_id), None)

    @property
    def all_flags(self) -> list[str]:
        return [f"{q.question_id}:{f}" for q in self.questions for f in q.validator_flags]

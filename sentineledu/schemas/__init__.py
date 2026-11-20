"""Pydantic contracts exchanged between agents, services and the UI (ARCHITECTURE §6)."""
from sentineledu.schemas.evaluation import (
    CriterionDraft,
    CriterionEvaluation,
    EvaluationDraft,
    EvaluationResult,
    QuestionDraft,
    QuestionEvaluation,
)
from sentineledu.schemas.extraction import (
    ILLEGIBLE_MARKER,
    UNKNOWN_QUESTION_ID,
    ExtractedAnswer,
    ExtractedSubmission,
)
from sentineledu.schemas.integrity import IntegrityFlag
from sentineledu.schemas.judge import (
    CriterionVerdict,
    Flag,
    FlagDraft,
    JudgeDraft,
    JudgeReport,
    QuestionFeedback,
    QuestionJudgeDraft,
    VerdictDraft,
)
from sentineledu.schemas.review import MarkOverride, TeacherDecision
from sentineledu.schemas.rubric import Criterion, Question, Rubric, RubricLevel
from sentineledu.schemas.submission import Submission, SubmissionStatus

__all__ = [
    "ILLEGIBLE_MARKER", "UNKNOWN_QUESTION_ID", "ExtractedAnswer", "ExtractedSubmission",
    "RubricLevel", "Criterion", "Question", "Rubric",
    "CriterionDraft", "QuestionDraft", "EvaluationDraft",
    "CriterionEvaluation", "QuestionEvaluation", "EvaluationResult",
    "Flag", "CriterionVerdict", "QuestionFeedback", "JudgeReport",
    "VerdictDraft", "QuestionJudgeDraft", "FlagDraft", "JudgeDraft",
    "IntegrityFlag",
    "MarkOverride", "TeacherDecision",
    "Submission", "SubmissionStatus",
]

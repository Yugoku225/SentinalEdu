"""A student's uploaded submission and its lifecycle status (ARCHITECTURE §3.1)."""
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class SubmissionStatus(str, Enum):
    UPLOADED = "UPLOADED"
    EXTRACTED = "EXTRACTED"
    NEEDS_EXTRACTION_REVIEW = "NEEDS_EXTRACTION_REVIEW"
    EVALUATED = "EVALUATED"
    PENDING_TEACHER_REVIEW = "PENDING_TEACHER_REVIEW"
    FINALIZED = "FINALIZED"
    FAILED = "FAILED"


class Submission(BaseModel):
    submission_id: str
    assignment_id: str
    student_ref: str | None = None
    files: list[str]  # paths relative to the storage directory, in page order
    file_sha256: str  # hash over all files in order; identical uploads give identical hashes
    status: SubmissionStatus = SubmissionStatus.UPLOADED
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

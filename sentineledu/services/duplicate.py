"""Duplicate detection within an assignment (TASKS 7.1 / 7.2, ARCHITECTURE §5.1, DECISIONS D-11).

Three checks, all informational: they produce IntegrityFlag rows for the teacher and never change marks.

- EXACT_DUPLICATE: byte-identical files (the submission's SHA-256 over all pages). Checked before
  OCR, so the earlier submission's extraction is reused and Gemini is not paid twice.
- TEXT_DUPLICATE: identical normalised text across all answers, different files (e.g. a re-scan or
  a photo of someone else's typed copy).
- NEAR_DUPLICATE / SIMILAR: per question, SBERT cosine between two students' answers. A pair is
  flagged only if it is at least `integrity_margin` closer to each other than either answer is to
  the reference material (the model answer AND the assignment's indexed reference chunks), so two
  students who both copied the textbook, or who both wrote the correct answer, are not flagged.
  Answers shorter than `integrity_min_tokens` words are skipped.
"""
import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sentineledu.config import Settings, get_settings
from sentineledu.schemas.extraction import ILLEGIBLE_MARKER, ExtractedSubmission
from sentineledu.schemas.integrity import IntegrityFlag
from sentineledu.schemas.rubric import Rubric
from sentineledu.schemas.submission import Submission
from sentineledu.services import embeddings
from sentineledu.services.rag import RagError, RagStore
from sentineledu.services.scoring_rules import is_attempted, quote_match_ratio
from sentineledu.services.storage import Storage

log = logging.getLogger("sentineledu.duplicate")

EXACT_DUPLICATE = "EXACT_DUPLICATE"
TEXT_DUPLICATE = "TEXT_DUPLICATE"
NEAR_DUPLICATE = "NEAR_DUPLICATE"
SIMILAR = "SIMILAR"
SCAN_CODES = {TEXT_DUPLICATE, NEAR_DUPLICATE, SIMILAR}


# --- text normalisation ----------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Same rules as the legacy `src.nlp_processing.clean_text_basic` (re-implemented here because
    importing that module loads NLTK and the spellchecker), then lower-cased."""
    text = text.replace(ILLEGIBLE_MARKER, " ")
    text = re.sub(r"[\n\t]+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,?!]", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def token_count(text: str) -> int:
    return len(clean_text(text).split())


def submission_text(extraction: ExtractedSubmission, question_ids: Sequence[str]) -> str:
    parts = []
    for qid in question_ids:
        a = extraction.answer_for(qid)
        parts.append(f"{qid}: {clean_text(a.answer_text) if a else ''}")
    return "\n".join(parts)


def text_sha256(extraction: ExtractedSubmission, question_ids: Sequence[str]) -> str:
    return hashlib.sha256(submission_text(extraction, question_ids).encode()).hexdigest()


# --- 7.1 exact duplicates --------------------------------------------------------------------

def exact_file_check(storage: Storage, submission: Submission) -> tuple[list[IntegrityFlag], ExtractedSubmission | None]:
    """Flags for byte-identical uploads, plus the extraction to reuse (from the oldest duplicate that has one)."""
    dups = storage.file_duplicates(submission)
    flags = []
    for other in dups:
        f = IntegrityFlag(assignment_id=submission.assignment_id, submission_id=submission.submission_id,
                          other_submission_id=other.submission_id, code=EXACT_DUPLICATE, severity="high",
                          similarity=1.0, message=f"The uploaded files are byte-identical to submission {other.submission_id}"
                          + (f" ({other.student_ref})" if other.student_ref else "") + ".")
        flags += [f, f.mirrored().model_copy(update={"message": f"The uploaded files are byte-identical to submission "
                                                                 f"{submission.submission_id}"
                                                                 + (f" ({submission.student_ref})" if submission.student_ref else "") + "."})]
    reuse = None
    for other in dups:
        ext = storage.latest_extraction(other.submission_id)
        if ext is not None:
            reuse = ext.model_copy(update={
                "submission_id": submission.submission_id,
                "issues": [*ext.issues, f"Reused the extraction of submission {other.submission_id} "
                                        f"(v{ext.version}): the files are identical, so Gemini was not called."]})
            break
    return flags, reuse


def text_duplicate_check(storage: Storage, submission: Submission, extraction: ExtractedSubmission, rubric: Rubric,
                         cohort: dict[str, ExtractedSubmission], settings: Settings) -> list[IntegrityFlag]:
    qids = [q.question_id for q in rubric.questions]
    if token_count(" ".join(a.answer_text for a in extraction.answers)) < settings.integrity_min_tokens:
        return []  # near-empty submissions match each other trivially
    mine = text_sha256(extraction, qids)
    same_files = {s.submission_id for s in storage.file_duplicates(submission)}
    flags = []
    for other_id, other in cohort.items():
        if other_id in same_files or text_sha256(other, qids) != mine:
            continue
        f = IntegrityFlag(assignment_id=submission.assignment_id, submission_id=submission.submission_id,
                          other_submission_id=other_id, code=TEXT_DUPLICATE, severity="high", similarity=1.0,
                          message=f"All answers are word-for-word identical to submission {other_id}, from different files.")
        flags += [f, f.mirrored().model_copy(update={"message": f"All answers are word-for-word identical to submission "
                                                                 f"{submission.submission_id}, from different files."})]
    return flags


# --- 7.2 near duplicates ---------------------------------------------------------------------

@dataclass
class PairResult:
    code: str | None  # NEAR_DUPLICATE, SIMILAR or None
    similarity: float
    margin: float
    skipped: str | None = None  # reason the pair was not compared


def classify_pair(similarity: float, ref_a: float, ref_b: float, tokens_a: int, tokens_b: int,
                  settings: Settings) -> PairResult:
    """The §5.1 rule for one pair of answers to the same question."""
    margin = similarity - max(ref_a, ref_b)
    if min(tokens_a, tokens_b) < settings.integrity_min_tokens:
        return PairResult(None, similarity, margin, skipped=f"answer shorter than {settings.integrity_min_tokens} words")
    if margin < settings.integrity_margin:
        return PairResult(None, similarity, margin)
    if similarity >= settings.near_dup_threshold:
        return PairResult(NEAR_DUPLICATE, similarity, margin)
    if similarity >= settings.similar_threshold:
        return PairResult(SIMILAR, similarity, margin)
    return PairResult(None, similarity, margin)


def reference_similarity(answer_vec: np.ndarray, answer_text: str, question, assignment_id: str,
                         rag: RagStore | None, settings: Settings, model_vec: np.ndarray) -> float:
    """How close an answer is to the reference material, the larger of:
    - SBERT cosine to the model answer;
    - SBERT cosine to each of the top retrieved reference chunks;
    - lexical containment: the best `difflib` match of the answer inside a retrieved chunk joined
      with its neighbours (the same sliding window that verifies evidence quotes). This catches answers
      copied from the textbook, which semantic similarity to a whole 500-character chunk underrates.
    """
    best = embeddings.cosine(answer_vec, model_vec)
    if rag is None or not rag.has_index(assignment_id):
        return best
    try:
        chunks = rag.retrieve(assignment_id, f"{question.prompt}\n{answer_text}", settings.rag_top_k)
    except RagError as e:
        log.warning("retrieval failed during the duplicate scan: %s", e)
        return best
    if chunks:
        vecs = embeddings.encode([c.text for c in chunks], settings)
        best = max(best, max(embeddings.cosine(answer_vec, v) for v in vecs))
        best = max(best, max(quote_match_ratio(answer_text, rag.neighbourhood(assignment_id, c.chunk_id)) for c in chunks))
    return best


def near_duplicate_check(submission: Submission, extraction: ExtractedSubmission, rubric: Rubric,
                         cohort: dict[str, ExtractedSubmission], settings: Settings,
                         rag: RagStore | None = None) -> list[IntegrityFlag]:
    flags: list[IntegrityFlag] = []
    for q in rubric.questions:
        mine = extraction.answer_for(q.question_id)
        if not is_attempted(mine) or token_count(mine.answer_text) < settings.integrity_min_tokens:
            continue
        others = {sid: e.answer_for(q.question_id) for sid, e in cohort.items()}
        others = {sid: a for sid, a in others.items()
                  if is_attempted(a) and token_count(a.answer_text) >= settings.integrity_min_tokens}
        if not others:
            continue
        ids = list(others)
        vecs = embeddings.encode([clean_text(mine.answer_text)] + [clean_text(others[i].answer_text) for i in ids]
                                 + [q.model_answer], settings)
        my_vec, other_vecs, model_vec = vecs[0], vecs[1:-1], vecs[-1]
        ref_mine = reference_similarity(my_vec, mine.answer_text, q, submission.assignment_id, rag, settings, model_vec)
        for sid, vec in zip(ids, other_vecs):
            ref_other = reference_similarity(vec, others[sid].answer_text, q, submission.assignment_id, rag, settings, model_vec)
            r = classify_pair(embeddings.cosine(my_vec, vec), ref_mine, ref_other,
                              token_count(mine.answer_text), token_count(others[sid].answer_text), settings)
            if r.code is None:
                continue
            word = "nearly identical" if r.code == NEAR_DUPLICATE else "very similar"
            f = IntegrityFlag(assignment_id=submission.assignment_id, submission_id=submission.submission_id,
                              other_submission_id=sid, code=r.code, severity="high" if r.code == NEAR_DUPLICATE else "review",
                              question_id=q.question_id, similarity=round(r.similarity, 4), margin=round(r.margin, 4),
                              message=f"{q.question_id}: answer is {word} to submission {sid}'s (similarity {r.similarity:.2f}, "
                                      f"{r.margin:.2f} closer to each other than to the reference).")
            flags += [f, f.mirrored().model_copy(update={"message": f.message.replace(f"submission {sid}'s",
                                                                                      f"submission {submission.submission_id}'s")})]
    return flags


def cohort_extractions(storage: Storage, submission: Submission) -> dict[str, ExtractedSubmission]:
    """Latest extraction of every other submission in the assignment that has one."""
    out = {}
    for other in storage.list_submissions(submission.assignment_id):
        if other.submission_id == submission.submission_id:
            continue
        ext = storage.latest_extraction(other.submission_id)
        if ext is not None:
            out[other.submission_id] = ext
    return out


def integrity_scan(storage: Storage, submission_id: str, *, settings: Settings | None = None,
                   rag: RagStore | None = None) -> list[IntegrityFlag]:
    """Text-duplicate and near-duplicate checks for one submission against its cohort; stores and returns
    the flags shown on this submission. Re-running replaces earlier results for this submission's pairs."""
    settings = settings or get_settings()
    sub = storage.get_submission(submission_id)
    rubric = storage.get_rubric(sub.assignment_id)
    extraction = storage.get_extraction(submission_id)
    cohort = cohort_extractions(storage, sub)
    text_flags = text_duplicate_check(storage, sub, extraction, rubric, cohort, settings)
    identical = {f.other_submission_id for f in text_flags if f.submission_id == submission_id}
    identical |= {s.submission_id for s in storage.file_duplicates(sub)}  # already EXACT_DUPLICATE
    near_cohort = {sid: e for sid, e in cohort.items() if sid not in identical}  # one flag per pair is enough
    flags = text_flags + near_duplicate_check(sub, extraction, rubric, near_cohort, settings, rag)
    storage.replace_integrity_flags(submission_id, SCAN_CODES, flags)
    return [f for f in flags if f.submission_id == submission_id]

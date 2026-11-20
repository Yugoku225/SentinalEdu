"""Agent 1 — Document Understanding (ARCHITECTURE §4.1, SENTINELEDU_TASKS.md 3.4).

Gemini reads the pages and splits them into answers per question. Python then normalises the
result (question ids, merged multi-page answers, legibility consistency, missing questions) and
falls back to the offline TrOCR reader if Gemini is unavailable.

CLI:  python -m sentineledu.agents.document_agent page1.jpg [page2.jpg ...] --questions Q1="..." Q2="..."
"""
import logging
import re
from pathlib import Path
from typing import Protocol, Sequence

from pydantic import BaseModel

from sentineledu.agents.base import build_agent
from sentineledu.agents.runner import LLMCallError, ResponseCache, run_structured
from sentineledu.config import Settings, get_settings
from sentineledu.schemas.extraction import ILLEGIBLE_MARKER, ExtractedAnswer, ExtractedSubmission
from sentineledu.services.documents import PreparedInput, prepare_inputs
from sentineledu.services.ocr_fallback import FallbackUnavailableError, extract_with_trocr

log = logging.getLogger("sentineledu.document_agent")

PROMPT_VERSION = "document_v1"
INSTRUCTIONS = (Path(__file__).parent / "prompts" / f"{PROMPT_VERSION}.md").read_text()

_LEGIBILITY_RANK = {"clear": 0, "partial": 1, "unreadable": 2}


class ExpectedQuestion(Protocol):
    question_id: str
    prompt: str


class DocumentDraft(BaseModel):
    """What Gemini returns. Submission id, engine and version are filled in by Python."""
    answers: list[ExtractedAnswer]
    unmatched_text: str = ""
    issues: list[str] = []


class DocumentExtractionError(RuntimeError):
    """Neither Gemini nor the TrOCR fallback could read the submission."""


def build_document_agent(settings: Settings | None = None):
    settings = settings or get_settings()
    return build_agent(name="document-agent", instructions=INSTRUCTIONS, output_schema=DocumentDraft,
                       model_id=settings.gemini_ocr_model, settings=settings)


def build_prompt(questions: Sequence[ExpectedQuestion], prepared: PreparedInput, file_names: Sequence[str]) -> str:
    lines = ["Assignment questions (use these question_id values exactly):"]
    lines += [f"- {q.question_id}: {q.prompt}" for q in questions]
    lines.append("")
    if prepared.is_text_only:
        lines += [
            "This submission is a typed PDF. Its text is below, between <document> tags.",
            "Split it into answers using the same rules. Typed text is normally legibility \"clear\".",
            "<document>", prepared.text, "</document>",
        ]
    else:
        lines.append(f"The submission has {prepared.page_count} page(s). Uploaded files, in page order: "
                     + ", ".join(file_names) + ".")
        lines.append("Read every page and return the answers.")
    return "\n".join(lines)


# --- normalisation ---------------------------------------------------------------------------

def _canonical(label: str) -> str:
    """'Q2(a)', 'q 2a', 'Ans 2a', 'Question 2 a' -> '2a'."""
    s = label.lower().strip()
    s = re.sub(r"^(question|ques|answer|ans|q)\.?\s*", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def _id_resolver(expected_ids: Sequence[str]):
    exact = {q: q for q in expected_ids}
    lower = {q.lower(): q for q in expected_ids}
    canon: dict[str, str | None] = {}
    for q in expected_ids:
        c = _canonical(q)
        canon[c] = None if c in canon else q  # ambiguous canonical forms resolve to nothing

    def resolve(label: str) -> str | None:
        return exact.get(label) or lower.get(label.lower().strip()) or canon.get(_canonical(label))
    return resolve


def _consistent_legibility(answer: ExtractedAnswer) -> str:
    text = answer.answer_text.strip()
    if not text:
        return "blank"
    if not text.replace(ILLEGIBLE_MARKER, "").strip(" .,;:-\n"):
        return "unreadable"
    if answer.legibility in ("clear", "blank"):
        return "partial" if ILLEGIBLE_MARKER in text else "clear"
    return answer.legibility


def _merge(parts: list[ExtractedAnswer]) -> ExtractedAnswer:
    written = [p for p in parts if p.legibility != "blank"]
    if not written:
        return parts[0]
    return ExtractedAnswer(
        question_id=parts[0].question_id,
        question_text_detected=next((p.question_text_detected for p in parts if p.question_text_detected), None),
        answer_text="\n".join(p.answer_text.strip() for p in written),
        legibility=max((p.legibility for p in written), key=_LEGIBILITY_RANK.__getitem__),
        confidence=min(p.confidence for p in written),
        contains_diagram=any(p.contains_diagram for p in parts),
        contains_math=any(p.contains_math for p in parts),
        page_refs=sorted({r for p in parts for r in p.page_refs}),
    )


def normalise_draft(draft: DocumentDraft, *, submission_id: str, expected_ids: Sequence[str], engine: str,
                    page_count: int, notes: Sequence[str] = (), model_id: str | None = None) -> ExtractedSubmission:
    resolve = _id_resolver(expected_ids)
    grouped: dict[str, list[ExtractedAnswer]] = {}
    unmatched = [draft.unmatched_text.strip()] if draft.unmatched_text.strip() else []

    for raw in draft.answers:
        answer = raw.model_copy(update={
            "page_refs": sorted({p for p in raw.page_refs if 1 <= p <= max(page_count, 1)}),
        })
        answer = answer.model_copy(update={"legibility": _consistent_legibility(answer)})
        qid = resolve(raw.question_id)
        if qid is None:
            if answer.answer_text.strip():
                unmatched.append(f"[labelled '{raw.question_id}'] {answer.answer_text.strip()}")
            continue
        grouped.setdefault(qid, []).append(answer.model_copy(update={"question_id": qid}))

    answers = [_merge(grouped[q]) for q in expected_ids if q in grouped]
    return ExtractedSubmission(
        submission_id=submission_id,
        engine=engine,
        answers=answers,
        unmatched_text="\n\n".join(unmatched),
        missing_question_ids=[q for q in expected_ids if q not in grouped],
        issues=[*draft.issues, *notes],
        model_id=model_id,
        prompt_version=PROMPT_VERSION if model_id else None,
    )


# --- entry point -----------------------------------------------------------------------------

def _offline_extract(submission_id: str, paths: Sequence[Path], prepared: PreparedInput,
                     expected_ids: list[str], settings: Settings, reason: str) -> ExtractedSubmission:
    if prepared.is_text_only:
        return ExtractedSubmission(
            submission_id=submission_id, engine="pdf_text", answers=[], unmatched_text=prepared.text,
            missing_question_ids=expected_ids, issues=[reason, *prepared.notes,
                                                       "Typed PDF text was not split into questions."],
        )
    try:
        result = extract_with_trocr(submission_id, paths, expected_question_ids=expected_ids, settings=settings)
    except FallbackUnavailableError as e:
        raise DocumentExtractionError(f"{reason} {e}") from e
    return result.model_copy(update={"issues": [reason, *result.issues, *prepared.notes]})


def extract_submission(
    submission_id: str,
    paths: Sequence[Path | str],
    questions: Sequence[ExpectedQuestion],
    *,
    agent=None,
    settings: Settings | None = None,
    cache: ResponseCache | None = None,
    use_cache: bool = True,
    enhance_images: bool = False,
) -> ExtractedSubmission:
    """Read a submission's pages and return answers per question (not yet saved)."""
    settings = settings or get_settings()
    paths = [Path(p) for p in paths]
    expected_ids = [q.question_id for q in questions]
    prepared = prepare_inputs(paths, enhance_images=enhance_images)

    if settings.offline_mode:
        return _offline_extract(submission_id, paths, prepared, expected_ids, settings,
                                "OFFLINE_MODE is on, so Gemini was not used.")
    if cache is None and use_cache:
        cache = ResponseCache(settings.cache_dir)
    agent = agent or build_document_agent(settings)
    try:
        result = run_structured(
            agent, build_prompt(questions, prepared, [p.name for p in paths]), DocumentDraft,
            prompt_version=PROMPT_VERSION, images=prepared.images, files=prepared.files, cache=cache,
        )
    except LLMCallError as e:
        log.warning("Gemini extraction failed for %s, using TrOCR fallback: %s", submission_id, e)
        return _offline_extract(submission_id, paths, prepared, expected_ids, settings,
                                f"Gemini could not read the submission ({str(e)[:200]}).")

    notes = list(prepared.notes)
    if result.fallback_used:
        notes.append(f"Read by fallback model {result.model_id}.")
    return normalise_draft(result.output, submission_id=submission_id, expected_ids=expected_ids,
                           engine="pdf_text" if prepared.is_text_only else "gemini",
                           page_count=prepared.page_count, notes=notes, model_id=result.model_id)


if __name__ == "__main__":
    import argparse
    from dataclasses import dataclass

    @dataclass
    class _Q:
        question_id: str
        prompt: str

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Extract answers from a submission (no grading).")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--questions", nargs="+", required=True, metavar='ID="prompt"')
    ap.add_argument("--enhance", action="store_true", help="apply legacy OpenCV clean-up to images")
    args = ap.parse_args()
    qs = [_Q(*q.split("=", 1)) for q in args.questions]
    out = extract_submission("cli", args.files, qs, enhance_images=args.enhance)
    print(out.model_dump_json(indent=2))

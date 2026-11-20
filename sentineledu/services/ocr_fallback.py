"""Offline OCR fallback using the legacy fine-tuned TrOCR (DECISIONS D-14).

TrOCR reads lines but cannot tell which question an answer belongs to, so all text goes into
`unmatched_text` and the extraction gate always sends the result to the teacher for review.
"""
from pathlib import Path
from typing import Sequence

from sentineledu.config import Settings, get_settings
from sentineledu.schemas.extraction import ExtractedSubmission
from sentineledu.services.documents import detect_kind


class FallbackUnavailableError(RuntimeError):
    pass


def _legacy_extract(image_path: str, model_path: str) -> tuple[str, float]:
    # Imported lazily: loading torch + TrOCR takes several seconds.
    from src.ocr_pipeline import extract_handwritten_text
    return extract_handwritten_text(image_path, model_path=model_path)


def extract_with_trocr(submission_id: str, paths: Sequence[Path], *,
                       expected_question_ids: Sequence[str] = (),
                       settings: Settings | None = None) -> ExtractedSubmission:
    settings = settings or get_settings()
    paths = [Path(p) for p in paths]
    pdfs = [p.name for p in paths if detect_kind(p) == "pdf"]
    if pdfs:
        raise FallbackUnavailableError(
            f"TrOCR fallback only handles images; cannot read PDF page(s) {pdfs}. Retry when Gemini is available.")
    if not settings.trocr_model_path.exists():
        raise FallbackUnavailableError(f"TrOCR model not found at {settings.trocr_model_path}")

    texts, confidences = [], []
    for page, path in enumerate(paths, start=1):
        text, conf = _legacy_extract(str(path), str(settings.trocr_model_path))
        if text.strip():
            texts.append(f"[page {page}]\n{text.strip()}")
            confidences.append(conf)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    issues = ["Read by the offline TrOCR fallback, which cannot split answers by question: "
              "assign the text to questions in the review screen.",
              f"Average TrOCR line confidence: {avg_conf:.2f}."]
    if not texts:
        issues.append("TrOCR found no text on any page.")
    return ExtractedSubmission(
        submission_id=submission_id,
        engine="trocr",
        answers=[],
        unmatched_text="\n\n".join(texts),
        missing_question_ids=list(expected_question_ids),
        issues=issues,
    )

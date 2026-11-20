"""Turn uploaded files into what the Document Understanding agent receives (SENTINELEDU_TASKS.md 3.2).

Routing:
- Images -> Gemini image parts. The format is detected from the bytes, not the extension
  (data/sample_images/sample1.png is actually a TIFF). Formats Gemini doesn't accept are
  converted to PNG. Optional OpenCV clean-up reuses the legacy preprocessing.
- PDFs with a real text layer (typed) -> their text, so no OCR is needed (engine "pdf_text").
- Other PDFs (scanned/handwritten) -> sent to Gemini as a PDF file part.
"""
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

from agno.media import File, Image
from PIL import Image as PILImage

GEMINI_IMAGE_MIME = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
MAX_INLINE_BYTES = 20 * 1024 * 1024  # Gemini inline request limit
MAX_IMAGE_SIDE = 3072
MIN_TEXT_LAYER_CHARS = 20  # fewer letters than this means the PDF is scanned, not typed

FileKind = Literal["image", "pdf"]


class UnsupportedFileError(ValueError):
    pass


@dataclass
class PreparedInput:
    images: list[Image] = field(default_factory=list)
    files: list[File] = field(default_factory=list)
    text: str = ""  # text layer from typed PDFs
    page_count: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def is_text_only(self) -> bool:
        return bool(self.text) and not self.images and not self.files

    @property
    def has_pdf_scan(self) -> bool:
        return bool(self.files)


def detect_kind(path: Path) -> FileKind:
    head = Path(path).read_bytes()[:8]
    if head.startswith(b"%PDF"):
        return "pdf"
    try:
        with PILImage.open(path) as im:
            im.verify()
        return "image"
    except Exception as e:
        raise UnsupportedFileError(f"{Path(path).name}: not an image or PDF") from e


def _enhance(img: PILImage.Image) -> PILImage.Image:
    """Legacy OpenCV clean-up: gentle CLAHE + deskew (src/ocr_pipeline.py)."""
    import cv2
    import numpy as np

    from src.ocr_pipeline import _apply_deskew, _get_skew_angle, create_binary_for_segmentation, prepare_clean_gray

    gray = prepare_clean_gray(cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR))
    angle = _get_skew_angle(create_binary_for_segmentation(gray))
    return PILImage.fromarray(_apply_deskew(gray, angle))


def prepare_image(path: Path, *, enhance: bool = False) -> tuple[Image, list[str]]:
    notes = []
    with PILImage.open(path) as im:
        fmt = im.format
        im.load()
        img = im
        n_frames = getattr(im, "n_frames", 1)
        if n_frames > 1:
            notes.append(f"{Path(path).name}: multi-frame image, only the first frame was used")
    changed = False
    if enhance:
        img, changed = _enhance(img), True
    if max(img.size) > MAX_IMAGE_SIDE:
        img = img.copy()
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        changed = True
        notes.append(f"{Path(path).name}: downscaled to {img.size[0]}x{img.size[1]}")
    if fmt in GEMINI_IMAGE_MIME and not changed:
        data, mime = Path(path).read_bytes(), GEMINI_IMAGE_MIME[fmt]
    else:
        buf = io.BytesIO()
        img.convert("RGB" if img.mode not in ("L", "RGB") else img.mode).save(buf, format="PNG")
        data, mime = buf.getvalue(), "image/png"
        if fmt not in GEMINI_IMAGE_MIME:
            notes.append(f"{Path(path).name}: converted {fmt} to PNG")
    return Image(content=data, mime_type=mime), notes


def pdf_text_layer(path: Path) -> tuple[str, int]:
    """(text, page_count). Text is empty when the PDF has no meaningful text layer."""
    import PyPDF2

    from src.pdf_utils import extract_text_from_pdf

    with open(path, "rb") as f:
        pages = len(PyPDF2.PdfReader(f).pages)
    text = extract_text_from_pdf(str(path))
    letters = len(re.findall(r"[A-Za-z]", text))
    return (text if letters >= MIN_TEXT_LAYER_CHARS else ""), pages


def prepare_inputs(paths: Sequence[Path], *, enhance_images: bool = False) -> PreparedInput:
    out = PreparedInput()
    texts: list[str] = []
    typed_pdfs: list[Path] = []
    for path in map(Path, paths):
        if detect_kind(path) == "image":
            image, notes = prepare_image(path, enhance=enhance_images)
            out.images.append(image)
            out.notes += notes
            out.page_count += 1
        else:
            text, pages = pdf_text_layer(path)
            out.page_count += pages
            if text:
                texts.append(text)
                typed_pdfs.append(path)
            else:
                out.files.append(File(content=path.read_bytes(), mime_type="application/pdf",
                                      filename=path.name))
    if texts and (out.images or out.files):
        # Mixed typed + scanned upload: send the typed PDFs as files too, so every page goes
        # through the same OCR path. (Agno sends images before files, so mixed uploads may reach
        # Gemini out of page order; the prompt lists the upload order.)
        out.files += [File(content=p.read_bytes(), mime_type="application/pdf", filename=p.name) for p in typed_pdfs]
        out.notes.append("typed PDF pages were OCR'd with the scanned pages for consistency")
    elif texts:
        out.text = "\n\n".join(texts)

    size = sum(len(m.content) for m in [*out.images, *out.files])
    if size > MAX_INLINE_BYTES:
        raise UnsupportedFileError(f"submission is {size / 1e6:.1f} MB; the limit is "
                                   f"{MAX_INLINE_BYTES / 1e6:.0f} MB. Upload fewer or smaller pages.")
    return out

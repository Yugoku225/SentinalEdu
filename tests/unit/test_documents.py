from pathlib import Path

import pytest
from PIL import Image as PILImage

from sentineledu.services.documents import (
    MAX_IMAGE_SIDE,
    UnsupportedFileError,
    detect_kind,
    pdf_text_layer,
    prepare_inputs,
)

ROOT = Path(__file__).resolve().parents[2]
TYPED_PDF = ROOT / "tests" / "fixtures" / "legacy" / "sample1_key.pdf"  # has a text layer
TIFF_NAMED_PNG = ROOT / "data" / "sample_images" / "sample1.png"     # really a TIFF


def make_image(path, fmt, size=(200, 100)):
    PILImage.new("RGB", size, "white").save(path, format=fmt)
    return path


@pytest.fixture
def scanned_pdf(tmp_path):
    """An image-only PDF, like a scanned handwritten page."""
    p = tmp_path / "scan.pdf"
    PILImage.new("RGB", (300, 400), "white").save(p, format="PDF")
    return p


def test_detect_kind_uses_bytes_not_extension(tmp_path, scanned_pdf):
    assert detect_kind(TIFF_NAMED_PNG) == "image"
    assert detect_kind(scanned_pdf) == "pdf"
    bad = tmp_path / "notes.png"
    bad.write_text("hello")
    with pytest.raises(UnsupportedFileError):
        detect_kind(bad)


@pytest.mark.parametrize("fmt, mime", [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")])
def test_supported_images_sent_unchanged(tmp_path, fmt, mime):
    p = make_image(tmp_path / f"page.{fmt.lower()}", fmt)
    out = prepare_inputs([p])
    assert out.images[0].mime_type == mime
    assert out.images[0].content == p.read_bytes()
    assert out.page_count == 1 and not out.files and not out.text


def test_tiff_converted_to_png():
    out = prepare_inputs([TIFF_NAMED_PNG])
    assert out.images[0].mime_type == "image/png"
    assert out.images[0].content.startswith(b"\x89PNG")
    assert any("converted TIFF to PNG" in n for n in out.notes)


def test_large_image_downscaled(tmp_path):
    p = make_image(tmp_path / "big.png", "PNG", size=(MAX_IMAGE_SIDE * 2, 1000))
    out = prepare_inputs([p])
    import io
    with PILImage.open(io.BytesIO(out.images[0].content)) as im:
        assert max(im.size) == MAX_IMAGE_SIDE
    assert any("downscaled" in n for n in out.notes)


def test_typed_pdf_uses_text_layer():
    out = prepare_inputs([TYPED_PDF])
    assert out.is_text_only
    assert "quick brown fox" in out.text
    assert out.page_count == 1


def test_scanned_pdf_sent_as_file(scanned_pdf):
    text, pages = pdf_text_layer(scanned_pdf)
    assert text == "" and pages == 1
    out = prepare_inputs([scanned_pdf])
    assert out.files[0].mime_type == "application/pdf"
    assert not out.is_text_only


def test_mixed_upload_keeps_every_page(tmp_path, scanned_pdf):
    img = make_image(tmp_path / "p.jpg", "JPEG")
    out = prepare_inputs([img, TYPED_PDF, scanned_pdf])
    assert len(out.images) == 1
    assert len(out.files) == 2          # typed PDF is sent too, not dropped
    assert out.text == ""
    assert out.page_count == 3


def test_enhance_reuses_legacy_preprocessing(tmp_path):
    p = make_image(tmp_path / "p.png", "PNG")
    out = prepare_inputs([p], enhance_images=True)
    assert out.images[0].mime_type == "image/png"
    with PILImage.open(__import__("io").BytesIO(out.images[0].content)) as im:
        assert im.mode == "L"  # legacy CLAHE output is grayscale


def test_oversized_submission_rejected(tmp_path, monkeypatch):
    import sentineledu.services.documents as docs
    monkeypatch.setattr(docs, "MAX_INLINE_BYTES", 10)
    with pytest.raises(UnsupportedFileError, match="limit"):
        prepare_inputs([make_image(tmp_path / "p.png", "PNG")])

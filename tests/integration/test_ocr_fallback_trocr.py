"""Real TrOCR fallback on the repo sample (slow: loads the 1.2 GB model). Run with: pytest -m legacy"""
from pathlib import Path

import pytest

from sentineledu.services.extraction_review import extraction_gate
from sentineledu.services.ocr_fallback import extract_with_trocr

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "sample_images" / "sample1.png"


@pytest.mark.legacy
def test_trocr_fallback_reads_sample_and_forces_review(offline_settings):
    from sentineledu.config import Settings
    settings = Settings(_env_file=None, offline_mode=True)  # real TROCR_MODEL_PATH
    out = extract_with_trocr("S1", [SAMPLE], expected_question_ids=["Q1"], settings=settings)
    assert out.engine == "trocr"
    assert "quick brown" in out.unmatched_text.lower()
    assert out.missing_question_ids == ["Q1"]
    gate = extraction_gate(out, ["Q1"], threshold=0.6)
    assert not gate.ok and any("TrOCR" in r for r in gate.reasons)


@pytest.mark.legacy
def test_trocr_runs_after_faiss_is_loaded():
    """Regression (2026-11-21): importing faiss before torch inference segfaulted the process on macOS.
    Runs in a subprocess so a crash fails this test instead of killing pytest."""
    import subprocess
    import sys
    from sentineledu.config import PROJECT_ROOT
    code = (
        "import sentineledu.services.rag\n"  # loads faiss the way the dashboard and CLI do
        "from pathlib import Path\n"
        "from sentineledu.config import Settings\n"
        "from sentineledu.services.ocr_fallback import extract_with_trocr\n"
        "s = Settings(_env_file=None, offline_mode=True)\n"
        "out = extract_with_trocr('x', [Path('tests/fixtures/images/printed_sample.png')], expected_question_ids=['Q1'], settings=s)\n"
        "print('ENGINE', out.engine)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "ENGINE trocr" in r.stdout

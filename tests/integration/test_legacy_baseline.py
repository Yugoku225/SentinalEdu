"""Regression check: the unchanged legacy evaluator must reproduce the saved baseline.

Slow (loads TrOCR, SBERT and DistilBERT). Run with: pytest -m legacy
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tests" / "fixtures" / "legacy_baseline.json"


@pytest.mark.legacy
def test_legacy_evaluator_matches_baseline(tmp_path):
    out = tmp_path / "current.json"
    subprocess.run([sys.executable, "scripts/legacy_baseline.py", "--out", str(out)],
                   cwd=ROOT, check=True, capture_output=True)
    expected = json.loads(BASELINE.read_text())["results"]
    actual = json.loads(out.read_text())["results"]

    assert actual["raw_ocr"] == expected["raw_ocr"]
    assert actual["nlp_cleaned"] == expected["nlp_cleaned"]
    assert actual["grading"] == expected["grading"]
    assert actual["feedback"] == expected["feedback"]
    for key in ("similarity_score", "ocr_confidence", "answer_deviation"):
        assert actual[key] == pytest.approx(expected[key], abs=0.01)

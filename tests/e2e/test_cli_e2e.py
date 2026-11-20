"""End-to-end through the CLI with real Gemini (SENTINELEDU_TASKS.md 9.3).

    pytest -m live tests/e2e/

Runs three submissions through `python -m sentineledu.cli assess` in a subprocess with a fresh storage
directory, then checks structure, status and bounds, not exact marks. Uses the gold set when
tests/fixtures/gold/manifest.json exists, otherwise the synthetic pages.
"""
import json
import os
import subprocess
import sys

import pytest

from sentineledu.config import PROJECT_ROOT
from sentineledu.services.rubric import load_rubric
from sentineledu.workflow.steps import STEP_ORDER

pytestmark = pytest.mark.live
RUBRIC_PATH = PROJECT_ROOT / "data" / "rubrics" / "sample_rubric.json"
RUBRIC = load_rubric(RUBRIC_PATH)
GOLD = PROJECT_ROOT / "tests" / "fixtures" / "gold" / "manifest.json"
SYNTHETIC = PROJECT_ROOT / "tests" / "fixtures" / "synthetic" / "manifest.json"
MANIFEST = GOLD if GOLD.exists() else SYNTHETIC
ITEMS = json.loads(MANIFEST.read_text())["items"][:3]
ALLOWED_END = {"PENDING_TEACHER_REVIEW", "NEEDS_EXTRACTION_REVIEW"}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from sentineledu.config import Settings
    try:
        if Settings().offline_mode:
            pytest.skip("OFFLINE_MODE is on")
    except ValueError:
        pytest.skip("GEMINI_API_KEY not configured")
    e = dict(os.environ)
    e["STORAGE_DIR"] = str(tmp_path_factory.mktemp("e2e_storage"))
    return e


def cli(env, *args):
    r = subprocess.run([sys.executable, "-m", "sentineledu.cli", *args], cwd=PROJECT_ROOT, env=env,
                       capture_output=True, text=True, timeout=900)
    return r


@pytest.mark.parametrize("item", ITEMS, ids=[i["id"] for i in ITEMS])
def test_cli_assess_end_to_end(env, item):
    files = [str(MANIFEST.parent / f) for f in item["files"]]
    r = cli(env, "assess", "--rubric", str(RUBRIC_PATH), "--student", item["id"], "--file", *files)
    run = json.loads(r.stdout)  # stdout must be pure JSON even when errors are logged
    if r.returncode != 0 and "RESOURCE_EXHAUSTED" in r.stderr:
        # The free tier allows 20 requests/day per model. The run must still have stopped cleanly, resumable later.
        assert run["status"] in {"EXTRACTED", "EVALUATED", "UPLOADED", "FAILED"} and run["stopped_at"], run
        pytest.skip(f"Gemini quota exhausted at step {run['stopped_at']}; re-run when the quota resets")
    assert r.returncode == 0, r.stderr[-3000:]

    assert run["status"] in ALLOWED_END, run
    assert run["assignment_id"] == RUBRIC.assignment_id
    steps = [s["step"] for s in run["steps"]]
    assert steps == STEP_ORDER[:len(steps)]  # executed in order, nothing skipped over
    assert all(s["status"] in {"ok", "skipped", "stopped", "fallback"} for s in run["steps"]), run["steps"]
    assert "document_understanding" in r.stderr  # per-step timings printed
    if run["status"] == "PENDING_TEACHER_REVIEW":
        assert steps == STEP_ORDER and run["stopped_at"] is None
        assert run["max_total"] == RUBRIC.max_total
        assert 0 <= run["evaluation_total"] <= RUBRIC.max_total
        assert 0 <= run["recommended_total"] <= RUBRIC.max_total
        assert abs(run["recommended_total"] - run["evaluation_total"]) <= RUBRIC.max_total
    else:
        assert run["stopped_at"] == "extraction_gate" and run["reason"]

    status = json.loads(cli(env, "status", run["submission_id"]).stdout)
    assert status["submission"]["status"] == run["status"]
    assert len([a for a in status["audit_log"] if a["run_id"] == run["run_id"]]) == len(steps)
    assert status["final_total"] is None  # never finalised without a teacher

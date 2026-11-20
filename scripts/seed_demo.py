"""Seed the demo class (SENTINELEDU_TASKS.md 9.6, MASTER_PLAN §9).

Creates the assignment, its rubric and reference PDF, then uploads and assesses five submissions
through the real workflow (Gemini):

  S01  clean answers                              -> PENDING_TEACHER_REVIEW
  S02  smudged, one answer unreadable             -> NEEDS_EXTRACTION_REVIEW (correct it in the UI)
  S03  answer continues onto a second page        -> PENDING_TEACHER_REVIEW
  S04  original answers with unusual analogies    -> PENDING_TEACHER_REVIEW
  S05  reworded copy of S04                       -> PENDING_TEACHER_REVIEW, near-duplicate flag on both
  (--exact-duplicate adds S06: S01's file again   -> EXACT_DUPLICATE flag, no second OCR call)

Usage:
  python scripts/seed_demo.py --reset            # wipe the database first (keeps the Gemini response cache)
  python scripts/seed_demo.py --storage-dir /tmp/sentinel_demo --reset
Then: streamlit run streamlit_app.py  (see docs/demo_script.md)

The pages are synthetic (rendered from handwriting fonts); swap in real pages for a real demo.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentineledu.config import Settings  # noqa: E402
from sentineledu.services.rag import RagStore  # noqa: E402
from sentineledu.services.rubric import load_rubric  # noqa: E402
from sentineledu.services.storage import Storage  # noqa: E402
from sentineledu.workflow.assessment_workflow import run_assessment  # noqa: E402
from sentineledu.workflow.steps import WorkflowDeps  # noqa: E402

SYN = ROOT / "tests" / "fixtures" / "synthetic"
DEMO = ROOT / "data" / "demo"
RUBRIC = ROOT / "data" / "rubrics" / "sample_rubric.json"
REFERENCE = DEMO / "photosynthesis_notes.pdf"
SUBMISSIONS = [
    ("S01", [SYN / "syn_01_clean_p1.jpg"], "clean answers"),
    ("S02", [SYN / "syn_02_messy_illegible_p1.jpg"], "one answer smudged (correct it in Extraction Review)"),
    ("S03", [SYN / "syn_04_two_pages_p1.jpg", SYN / "syn_04_two_pages_p2.jpg"], "answer continues onto page 2"),
    ("S04", [DEMO / "demo_s04_original_p1.jpg"], "original wording"),
    ("S05", [DEMO / "demo_s05_copied_p1.jpg"], "reworded copy of S04"),
]
RESET_TARGETS = ["sentinel.db", "uploads", "indexes", "references"]  # never "cache": it saves Gemini quota


def reset(root: Path) -> None:
    for name in RESET_TARGETS:
        target = root / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    print(f"Reset {root}: removed {', '.join(RESET_TARGETS)} (kept cache/)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--storage-dir", type=Path, help="defaults to STORAGE_DIR from .env (storage/)")
    ap.add_argument("--reset", action="store_true", help="delete the database, uploads, indexes and references first")
    ap.add_argument("--exact-duplicate", action="store_true", help="also upload S01's file again as S06")
    args = ap.parse_args()

    settings = Settings(**({"storage_dir": args.storage_dir} if args.storage_dir else {}))
    if args.reset:
        reset(settings.storage_dir)
    storage = Storage(settings)
    rubric = load_rubric(RUBRIC)
    if storage.list_submissions(rubric.assignment_id) and not args.reset:
        print(f"Assignment {rubric.assignment_id} already has submissions in {settings.storage_dir}. "
              "Re-run with --reset for a clean demo, or use --storage-dir.", file=sys.stderr)
        return 2

    start = time.monotonic()
    saved = storage.save_rubric(rubric)
    deps = WorkflowDeps(storage=storage, settings=settings, rag=RagStore(settings))
    ref_dir = settings.storage_dir / "references" / rubric.assignment_id
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref = shutil.copy(REFERENCE, ref_dir / REFERENCE.name)
    chunks = deps.rag.ingest(rubric.assignment_id, [ref])
    print(f"Assignment {saved.assignment_id}: rubric v{saved.version} ({saved.max_total:g} marks), "
          f"reference {REFERENCE.name} -> {chunks} chunks")

    plan = list(SUBMISSIONS) + ([("S06", [SYN / "syn_01_clean_p1.jpg"], "same file as S01")] if args.exact_duplicate else [])
    print(f"\n{'student':8} {'status':26} {'marks':>11} {'secs':>6}  note")
    for student, files, note in plan:
        sub = storage.create_submission(rubric.assignment_id, files, student_ref=student)
        t = time.monotonic()
        run = run_assessment(sub.submission_id, deps)
        marks = (f"{run.evaluation_total:g}->{run.recommended_total:g}/{run.max_total:g}"
                 if run.recommended_total is not None else "-")
        flags = sorted({f.code for f in storage.integrity_flags(sub.submission_id)})
        print(f"{student:8} {run.status.value:26} {marks:>11} {time.monotonic() - t:6.1f}  {note}"
              + (f"  [{', '.join(flags)}]" if flags else "") + (f"  stopped: {run.reason}" if run.reason else ""))

    print(f"\nSeeded in {time.monotonic() - start:.0f} s. Next: streamlit run streamlit_app.py  (talk track: docs/demo_script.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

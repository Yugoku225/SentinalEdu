"""Command-line runner (TASKS 6.5).

  python -m sentineledu.cli assess --assignment photosynthesis_basics --file page1.jpg [--file page2.jpg] --student S01 \
      [--rubric data/rubrics/sample_rubric.json] [--reference notes.pdf]
  python -m sentineledu.cli resume <submission_id>          # re-run after a correction, or retry
  python -m sentineledu.cli status <submission_id>          # status, totals and the audit trail
  python -m sentineledu.cli rubric-import data/rubrics/sample_rubric.json

The summary JSON goes to stdout; per-step timings go to stderr.
"""
import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path

from sentineledu.config import Settings
from sentineledu.services.rag import RagStore
from sentineledu.services.rubric import RubricError, load_rubric
from sentineledu.services.storage import NotFoundError, Storage
from sentineledu.workflow.assessment_workflow import AssessmentRun, assess_files, run_assessment
from sentineledu.workflow.steps import WorkflowDeps


def logs_to_stderr() -> None:
    """Agno's loggers write to stdout; move them to stderr so stdout stays pure JSON (found in the e2e run)."""
    import agno.utils.log as agno_log

    for name in (agno_log.LOGGER_NAME, agno_log.TEAM_LOGGER_NAME, agno_log.WORKFLOW_LOGGER_NAME):
        for handler in logging.getLogger(name).handlers:
            if isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) in (sys.stdout, sys.__stdout__):
                handler.setStream(sys.stderr)


def _import_rubric(storage: Storage, path: Path, assignment_id: str | None) -> str:
    rubric = load_rubric(path)
    if assignment_id and assignment_id != rubric.assignment_id:
        raise RubricError(f"--assignment {assignment_id!r} does not match the rubric's assignment_id {rubric.assignment_id!r}")
    try:
        stored = storage.get_rubric(rubric.assignment_id)
        same = stored.model_dump(exclude={"version"}) == rubric.model_dump(exclude={"version"})
    except NotFoundError:
        same = False
    if not same:
        saved = storage.save_rubric(rubric)
        print(f"Stored rubric {saved.assignment_id} v{saved.version}", file=sys.stderr)
    return rubric.assignment_id


def _print_timings(run: AssessmentRun) -> None:
    print(f"\nSubmission {run.submission_id} ({run.assignment_id}) -> {run.status.value}   run {run.run_id}", file=sys.stderr)
    print(f"{'step':24} {'status':9} {'ms':>8}  model", file=sys.stderr)
    for s in run.steps:
        print(f"{s.step:24} {s.status:9} {s.duration_ms or 0:8d}  {s.model_id or ''}", file=sys.stderr)
    print(f"{'total':24} {'':9} {run.duration_ms:8d}", file=sys.stderr)
    if run.reason:
        print(f"Stopped at {run.stopped_at}: {run.reason}", file=sys.stderr)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sentineledu", description="SentinelEdu assessment workflow")
    sub = ap.add_subparsers(dest="command", required=True)

    a = sub.add_parser("assess", help="store a submission and run the workflow")
    a.add_argument("--assignment", help="assignment id (defaults to the rubric's)")
    a.add_argument("--file", action="extend", nargs="+", required=True, type=Path, help="page files in order")
    a.add_argument("--student", help="anonymised student reference")
    a.add_argument("--rubric", type=Path, help="rubric JSON to store first (new version only if it changed)")
    a.add_argument("--reference", nargs="*", type=Path, default=[], help="reference PDF/TXT to index for this assignment")
    a.add_argument("--no-cache", action="store_true", help="do not reuse cached Gemini answers")

    r = sub.add_parser("resume", help="re-run the workflow for a stored submission")
    r.add_argument("submission_id")
    r.add_argument("--no-cache", action="store_true")

    st = sub.add_parser("status", help="show a submission's status and audit trail")
    st.add_argument("submission_id")

    ri = sub.add_parser("rubric-import", help="validate and store a rubric")
    ri.add_argument("path", type=Path)

    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logs_to_stderr()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    settings = Settings()
    storage = Storage(settings)
    try:
        if args.command == "rubric-import":
            print(json.dumps({"assignment_id": _import_rubric(storage, args.path, None)}), file=_OUT)
            return 0

        if args.command == "status":
            s = storage.get_submission(args.submission_id)
            out = {"submission": json.loads(s.model_dump_json()), "audit_log": storage.audit_log(s.submission_id)}
            ev = storage.latest_evaluation(s.submission_id)
            rep = storage.latest_judge_report(s.submission_id)
            dec = storage.get_decision(s.submission_id)
            out |= {"evaluation_total": ev.total if ev else None, "recommended_total": rep.recommended_total if rep else None,
                    "final_total": dec.total if dec else None}
            print(json.dumps(out, indent=2, default=str), file=_OUT)
            return 0

        deps = WorkflowDeps(storage=storage, settings=settings, rag=RagStore(settings), use_cache=not args.no_cache)
        if args.command == "assess":
            assignment = args.assignment
            if args.rubric:
                assignment = _import_rubric(storage, args.rubric, assignment)
            if not assignment:
                ap.error("give --assignment or --rubric")
            if args.reference:
                n = deps.rag.ingest(assignment, args.reference)
                print(f"Reference index for {assignment}: {n} chunks", file=sys.stderr)
            run = assess_files(assignment, args.file, deps, student_ref=args.student)
        else:
            run = run_assessment(args.submission_id, deps)
    except (RubricError, NotFoundError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    _print_timings(run)
    print(run.model_dump_json(indent=2), file=_OUT)
    return 0 if run.stopped_at is None or run.status.value == "NEEDS_EXTRACTION_REVIEW" else 1


_OUT = sys.stdout


def main(argv: list[str] | None = None) -> int:
    """stdout carries only the JSON result. Anything else printed while working (the legacy TrOCR code
    prints progress with print()) is sent to stderr. Found in the e2e run, 2026-11-21."""
    global _OUT
    _OUT = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        return _main(argv)


if __name__ == "__main__":
    sys.exit(main())

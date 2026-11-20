"""Seeded-error benchmark for the Judge (SENTINELEDU_TASKS.md 5.5).

1. Grade the manifest's submissions with Agent 2 (from the transcriptions) to get correct evaluations.
2. Inject one error into each of 5 copies (wrong level up, wrong level down, concept marked missing
   though present, marks inflated past the rubric, evidence stripped).
3. Run the deterministic checks + Judge on each and count how many injected errors were
   detected: flagged by a check, flagged by the judge, or corrected by an accepted proposal.
4. Also run the Judge on the unmodified evaluations to measure false alarms.

Usage:
  python scripts/eval_judge.py tests/fixtures/synthetic/manifest.json --rubric data/rubrics/sample_rubric.json \
      --reference tests/fixtures/reference/photosynthesis_notes.txt --json docs/phase5_seeded_errors.json
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentineledu.agents.evaluation_agent import build_context, evaluate  # noqa: E402
from sentineledu.agents.judge_agent import JudgeError, judge  # noqa: E402
from sentineledu.config import Settings  # noqa: E402
from sentineledu.schemas.evaluation import EvaluationResult  # noqa: E402
from sentineledu.schemas.extraction import ExtractedAnswer, ExtractedSubmission  # noqa: E402
from sentineledu.schemas.rubric import Rubric  # noqa: E402
from sentineledu.services.rag import RagStore  # noqa: E402
from sentineledu.services.rubric import load_rubric  # noqa: E402
from sentineledu.services.scoring_rules import check_recommendations_within_bounds  # noqa: E402


def extraction_from_transcription(item: dict, qids: list[str]) -> ExtractedSubmission:
    answers = [ExtractedAnswer(question_id=q, answer_text=item["transcription"].get(q, ""),
                               legibility="clear" if item["transcription"].get(q, "").strip() else "blank", confidence=1.0)
               for q in qids]
    return ExtractedSubmission(submission_id=item["id"], engine="teacher", answers=answers)


def _set_level(ev: EvaluationResult, rubric: Rubric, qid: str, cid: str, level_id: str, *, keep_quotes=True) -> None:
    ce = ev.question(qid).criterion(cid)
    lv = rubric.question(qid).criterion(cid).level(level_id)
    ce.selected_level_id, ce.marks = lv.level_id, lv.marks
    if not keep_quotes:
        ce.evidence_quotes = []
    ce.evidence_verified = bool(ce.evidence_quotes)
    qe = ev.question(qid)
    qe.total = sum(c.marks for c in qe.criteria)
    ev.total = sum(q.total for q in ev.questions)


def _pick(evals: dict[str, EvaluationResult], rubric: Rubric, want: str) -> tuple[str, str, str] | None:
    """The first (submission, question, criterion) where an error of kind `want` can be injected."""
    return next(_spots(evals, rubric, want), None)


def _spots(evals: dict[str, EvaluationResult], rubric: Rubric, want: str):
    """Every (submission, question, criterion) where an error of kind `want` can be injected, in order."""
    for sid, ev in evals.items():
        for qe in ev.questions:
            q = rubric.question(qe.question_id)
            for ce in qe.criteria:
                c = q.criterion(ce.criterion_id)
                order = [lv.level_id for lv in sorted(c.levels, key=lambda l: l.marks)]
                i = order.index(ce.selected_level_id)
                attempted = qe.validator_flags != ["NOT_ATTEMPTED"]
                if not attempted:
                    continue
                if want == "level_up" and i < len(order) - 1 and ce.marks >= 0:
                    yield sid, qe.question_id, ce.criterion_id
                if want == "level_down" and i > 0:
                    yield sid, qe.question_id, ce.criterion_id
                if want == "concept_missing" and ce.marks > 0 and ce.concepts_present:
                    yield sid, qe.question_id, ce.criterion_id
                if want == "inflated" and ce.marks > 0:
                    yield sid, qe.question_id, ce.criterion_id
                if want == "no_evidence" and ce.marks > 0 and ce.evidence_verified:
                    yield sid, qe.question_id, ce.criterion_id


def inject_errors(evals: dict[str, EvaluationResult], rubric: Rubric, extractions: dict[str, ExtractedSubmission]) -> list[dict]:
    """Five seeded errors, each on a copy of a correct evaluation. Returns [{kind, sid, qid, cid, evaluation}]."""
    seeds = []
    used: set[tuple[str, str, str]] = set()

    def take(kind):
        # First pass: the first suitable criterion of each submission (spreads errors across submissions).
        for sid, ev in evals.items():
            spot = _pick({sid: ev}, rubric, kind)
            if spot and spot not in used:
                used.add(spot)
                return spot
        # Second pass: any suitable criterion not used yet.
        for spot in _spots(evals, rubric, kind):
            if spot not in used:
                used.add(spot)
                return spot
        return None

    for kind in ["level_up", "level_down", "concept_missing", "inflated", "no_evidence"]:
        spot = take(kind)
        if spot is None:
            seeds.append({"kind": kind, "skipped": "no suitable criterion"})
            continue
        sid, qid, cid = spot
        ev = copy.deepcopy(evals[sid])
        ce = ev.question(qid).criterion(cid)
        c = rubric.question(qid).criterion(cid)
        order = [lv.level_id for lv in sorted(c.levels, key=lambda l: l.marks)]
        i = order.index(ce.selected_level_id)
        original = ce.selected_level_id
        if kind == "level_up":  # over-generous: one level higher than deserved (quotes kept, so the validator can't catch it)
            _set_level(ev, rubric, qid, cid, order[i + 1])
            if not ce.evidence_quotes:  # a 0-level had no quotes; quote the answer's first words so it looks legitimate
                words = extractions[sid].answer_for(qid).answer_text.split()
                ce.evidence_quotes = [" ".join(words[:6])]
                ce.evidence_verified = True
            ce.rationale = "Fully satisfied."
            ce.missing_concepts = []
        elif kind == "level_down":  # unfair: one level lower than deserved
            _set_level(ev, rubric, qid, cid, order[i - 1])
            ce.rationale = "Not enough detail."
        elif kind == "concept_missing":  # level kept, but a present concept is listed as missing
            moved = ce.concepts_present.pop(0)
            ce.missing_concepts = [moved] + ce.missing_concepts
            ce.rationale = f"{moved} is not mentioned."
        elif kind == "inflated":  # marks edited past the rubric (level unchanged)
            ce.marks = c.max_marks + 1
            qe = ev.question(qid)
            qe.total = sum(x.marks for x in qe.criteria)
            ev.total = sum(q.total for q in ev.questions)
        elif kind == "no_evidence":  # marks kept, evidence removed
            ce.evidence_quotes, ce.evidence_verified = [], False
        seeds.append({"kind": kind, "sid": sid, "qid": qid, "cid": cid, "from_level": original,
                      "to_level": ce.selected_level_id, "evaluation": ev})
    return seeds


def detected(seed: dict, report, rubric: Rubric) -> tuple[bool, str]:
    qid, cid, kind = seed["qid"], seed["cid"], seed["kind"]
    verdict = next(v for v in report.verdicts[qid] if v.criterion_id == cid)
    check_hit = [f for f in report.flags if f.question_id == qid and f.criterion_id == cid
                 and f.code in ("OUT_OF_BOUNDS", "NO_EVIDENCE", "SUM_MISMATCH", "SBERT_LEVEL_DISAGREEMENT")]
    if check_hit:
        return True, f"deterministic check {check_hit[0].code}"
    if verdict.accepted and verdict.recommended_level_id == seed["from_level"]:
        return True, f"judge corrected to {verdict.recommended_level_id!r}"
    if verdict.verdict == "propose_adjustment" and verdict.recommended_level_id != seed["to_level"]:
        return True, f"judge proposed {verdict.proposed_level_id!r} (accepted={verdict.accepted})"
    if verdict.verdict in ("propose_adjustment", "flag"):
        return True, f"judge {verdict.verdict}: {verdict.justification[:80]}"
    fb = report.feedback_for(qid)
    if kind == "concept_missing" and fb and fb.overlooked_concepts:
        return True, f"judge listed overlooked concepts {fb.overlooked_concepts}"
    return False, f"judge agreed: {verdict.justification[:80]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--rubric", type=Path, required=True)
    ap.add_argument("--reference", nargs="*", type=Path, default=[])
    ap.add_argument("--model")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    rubric = load_rubric(args.rubric)
    manifest = json.loads(args.manifest.read_text())
    qids = [q.question_id for q in rubric.questions]
    settings = Settings(**({"gemini_reasoning_model": args.model} if args.model else {}))
    store = RagStore(settings)
    if args.reference:
        store.ingest(rubric.assignment_id, args.reference, replace=True)

    extractions = {i["id"]: extraction_from_transcription(i, qids) for i in manifest["items"]}
    contexts = {sid: build_context(ext, rubric, rag=store, settings=settings) for sid, ext in extractions.items()}
    print("Grading with Agent 2 ...")
    evals = {sid: evaluate(ext, rubric, context=contexts[sid], settings=settings) for sid, ext in extractions.items()}

    print("Judging the unmodified evaluations (false-alarm check) ...")
    baseline = []
    for sid, ev in evals.items():
        t = time.monotonic()
        try:
            rep = judge(extractions[sid], rubric, ev, context=contexts[sid], settings=settings)
        except JudgeError as e:
            print(f"  !! {sid}: {str(e)[:160]}")
            baseline.append({"sid": sid, "error": str(e)[:300], "bounds_problems": [], "feedback_for_all": False})
            continue
        disagreements = [v for vs in rep.verdicts.values() for v in vs if v.verdict != "agree"]
        baseline.append({"sid": sid, "seconds": round(time.monotonic() - t, 1), "evaluator_total": ev.total,
                         "recommended_total": rep.recommended_total, "disagreements": len(disagreements),
                         "accepted_changes": sum(v.accepted for v in disagreements),
                         "bounds_problems": check_recommendations_within_bounds(rep, ev, rubric),
                         "feedback_for_all": all((f.strengths or f.improvements) for f in rep.feedback)})

    print("Injecting errors and judging ...")
    seeds = inject_errors(evals, rubric, extractions)
    results = []
    for seed in seeds:
        if "skipped" in seed:
            results.append(seed)
            continue
        ev = seed.pop("evaluation")
        t = time.monotonic()
        try:
            rep = judge(extractions[seed["sid"]], rubric, ev, context=contexts[seed["sid"]], settings=settings)
        except JudgeError as e:
            print(f"  !! {seed['kind']}: {str(e)[:160]}")
            results.append({**seed, "detected": False, "how": f"NOT RUN: {str(e)[:120]}", "bounds_problems": []})
            continue
        hit, how = detected(seed, rep, rubric)
        results.append({**seed, "detected": hit, "how": how, "seconds": round(time.monotonic() - t, 1),
                        "evaluator_total": ev.total, "recommended_total": rep.recommended_total,
                        "bounds_problems": check_recommendations_within_bounds(rep, ev, rubric)})

    n = sum(1 for r in results if "detected" in r)
    hits = sum(1 for r in results if r.get("detected"))
    print("\n| # | Injected error | Where | Detected | How |")
    print("|---|---|---|---|---|")
    for i, r in enumerate(results, 1):
        if "skipped" in r:
            print(f"| {i} | {r['kind']} | – | skipped | {r['skipped']} |")
        else:
            print(f"| {i} | {r['kind']} ({r['from_level']}→{r['to_level']}) | {r['sid']} {r['qid']}/{r['cid']} | "
                  f"{'YES' if r['detected'] else 'no'} | {r['how']} |")
    print(f"\nDetection rate: {hits}/{n}")
    print("\nUnmodified evaluations:")
    for b in baseline:
        if "error" in b:
            print(f"  {b['sid']:28} NOT RUN: {b['error'][:100]}")
            continue
        print(f"  {b['sid']:28} evaluator {b['evaluator_total']:g} -> recommended {b['recommended_total']:g}  "
              f"disagreements={b['disagreements']} accepted={b['accepted_changes']} bounds_ok={not b['bounds_problems']} "
              f"feedback_ok={b['feedback_for_all']}  {b['seconds']}s")
    bounds_bad = [b for b in baseline if b["bounds_problems"]] + [r for r in results if r.get("bounds_problems")]
    not_run = sum('error' in b for b in baseline) + sum(r.get('how', '').startswith('NOT RUN') for r in results)
    print(f"\nExit gate: recommended marks within rubric bounds in {len(baseline) + n - len(bounds_bad) - not_run}/{len(baseline) + n - not_run} completed runs; "
          f"feedback for every answered question: {all(b['feedback_for_all'] for b in baseline)}")
    if not_run:
        print(f"!! {not_run} run(s) did not complete (Gemini unavailable or quota exhausted). Re-run later; completed calls are cached.")
    if manifest.get("synthetic"):
        print("\nNOTE: synthetic set; see docs/results.md for caveats.")
    if args.json:
        args.json.write_text(json.dumps({"baseline": baseline, "seeded": results, "detection": f"{hits}/{n}"}, indent=2))
    return 0 if hits >= 4 and not bounds_bad else 1


if __name__ == "__main__":
    sys.exit(main())

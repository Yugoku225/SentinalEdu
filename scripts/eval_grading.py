"""Measure grading accuracy against teacher marks (SENTINELEDU_TASKS.md 4.8).

Compares, per submission and per question:
- SentinelEdu Agent 2 (evaluation_agent + scoring_rules), graded from the manifest's
  transcription by default (isolates grading from OCR) or from real OCR with --from-ocr;
- the legacy grader (src/similarity.py + src/scoring.py + agent_verifier rescue) on the same
  text, with --legacy.

Metrics: mean absolute error on question marks and totals, exact-level agreement per criterion
(when the manifest gives per-criterion marks), Cohen's kappa on levels (when >= 10 criteria are
compared and both sides use more than one level), and the Phase 4 exit gate checks
(every awarded criterion has verified evidence; totals match the rubric).

Manifest: the gold/synthetic manifest format plus, per item,
  "teacher_marks": {"Q1": {"C1": 2, "C2": 1}, "Q2": 2, ...}   (per criterion or per question)

Usage:
  python scripts/eval_grading.py tests/fixtures/synthetic/manifest.json --rubric data/rubrics/sample_rubric.json \
      --reference tests/fixtures/reference/photosynthesis_notes.txt --legacy --json docs/phase4_synthetic_grading.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentineledu.agents.evaluation_agent import evaluate  # noqa: E402
from sentineledu.config import Settings  # noqa: E402
from sentineledu.schemas.extraction import ExtractedAnswer, ExtractedSubmission  # noqa: E402
from sentineledu.schemas.rubric import Rubric  # noqa: E402
from sentineledu.services.rag import RagStore  # noqa: E402
from sentineledu.services.rubric import load_rubric  # noqa: E402
from sentineledu.services.scoring_rules import check_marks_are_rubric_derived  # noqa: E402

MIN_KAPPA_N = 10


def extraction_from_transcription(item: dict, qids: list[str]) -> ExtractedSubmission:
    expect = item.get("expect", {})
    answers = []
    for qid in qids:
        text = item["transcription"].get(qid, "")
        legibility = "blank" if not text.strip() else ("partial" if qid in expect.get("illegible", []) else "clear")
        answers.append(ExtractedAnswer(question_id=qid, answer_text=text, legibility=legibility, confidence=1.0))
    return ExtractedSubmission(submission_id=item["id"], engine="teacher", answers=answers, corrected_by="gold set")


def extraction_from_ocr(item: dict, base: Path, rubric: Rubric, settings: Settings) -> ExtractedSubmission:
    from sentineledu.agents.document_agent import extract_submission
    return extract_submission(item["id"], [base / f for f in item["files"]], rubric.questions, settings=settings)


def teacher_marks(item: dict, rubric: Rubric) -> dict[str, dict[str, float | None]] | None:
    """question_id -> criterion_id -> marks (None when only a question total was given)."""
    given = item.get("teacher_marks")
    if not given:
        return None
    out: dict[str, dict[str, float | None]] = {}
    for q in rubric.questions:
        v = given.get(q.question_id)
        if isinstance(v, dict):
            out[q.question_id] = {c.criterion_id: float(v.get(c.criterion_id, 0)) for c in q.criteria}
        elif v is None:
            out[q.question_id] = {c.criterion_id: None for c in q.criteria}
        else:
            out[q.question_id] = ({q.criteria[0].criterion_id: float(v)} if len(q.criteria) == 1
                                  else {c.criterion_id: None for c in q.criteria})
            out[q.question_id]["__total__"] = float(v)
    return out


def level_for_marks(rubric: Rubric, qid: str, cid: str, marks: float | None) -> str | None:
    if marks is None:
        return None
    c = rubric.question(qid).criterion(cid)
    return next((lv.level_id for lv in c.levels if abs(lv.marks - marks) < 1e-6), None)


def legacy_grade(text: str, model_answer: str, max_marks: float) -> float:
    """The legacy pipeline's grading stages (NLP clean -> SBERT -> threshold -> rescue), without OCR."""
    from src.agent_verifier import rescue_grade_if_needed
    from src.nlp_processing import process_student_answer
    from src.scoring import calculate_grade
    from src.similarity import compute_similarity

    if not text.strip():
        return 0.0
    nlp = process_student_answer(text)
    sim = compute_similarity(nlp["final_processed_string"], model_answer)
    grades = calculate_grade(sim, max_marks=max_marks)
    grades = rescue_grade_if_needed(grades, student_raw_text=nlp["regex_cleaned_text"], model_answer=model_answer)
    return float(min(grades["awarded_marks"], max_marks))


def kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa between two label lists (None when the sample is too small to mean anything)."""
    if len(a) < MIN_KAPPA_N or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    n = len(a)
    labels = set(a) | set(b)
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(lb) / n) * (b.count(lb) / n) for lb in labels)
    return None if expected == 1.0 else (observed - expected) / (1 - expected)


def run(manifest_path: Path, rubric: Rubric, reference: list[Path], *, from_ocr: bool, legacy: bool,
        model: str | None) -> dict:
    manifest = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    qids = [q.question_id for q in rubric.questions]
    settings = Settings(**({"gemini_reasoning_model": model} if model else {}))
    store = RagStore(settings)
    if reference:
        store.ingest(rubric.assignment_id, reference, replace=True)

    rows, gate_problems = [], []
    abs_err_q, abs_err_total, abs_err_q_legacy, abs_err_total_legacy = [], [], [], []
    levels_teacher, levels_model = [], []
    awarded, awarded_verified = 0, 0
    for item in manifest["items"]:
        start = time.monotonic()
        ext = (extraction_from_ocr(item, base, rubric, settings) if from_ocr
               else extraction_from_transcription(item, qids))
        result = evaluate(ext, rubric, rag=store, settings=settings)
        seconds = time.monotonic() - start
        gate_problems += [f"{item['id']}: {p}" for p in check_marks_are_rubric_derived(result, rubric)]
        for qe in result.questions:
            for ce in qe.criteria:
                if ce.marks > 0:
                    awarded += 1
                    awarded_verified += ce.evidence_verified

        marks = teacher_marks(item, rubric)
        row = {"item": item["id"], "seconds": round(seconds, 1), "model": result.model_id,
               "sentinel_total": result.total, "max_total": result.max_total,
               "teacher_total": None, "legacy_total": None, "questions": {},
               "flags": result.all_flags}
        teacher_total = 0.0
        for q in rubric.questions:
            qe = result.question(q.question_id)
            qrow = {"sentinel": qe.total, "levels": {c.criterion_id: c.selected_level_id for c in qe.criteria}}
            if marks:
                tm = marks[q.question_id]
                t_total = tm.get("__total__", sum(v or 0 for k, v in tm.items() if k != "__total__"))
                qrow["teacher"] = t_total
                teacher_total += t_total
                abs_err_q.append(abs(qe.total - t_total))
                for c in q.criteria:
                    t_level = level_for_marks(rubric, q.question_id, c.criterion_id, tm.get(c.criterion_id))
                    if t_level is not None:
                        levels_teacher.append(t_level)
                        levels_model.append(qe.criterion(c.criterion_id).selected_level_id)
            if legacy:
                ans = ext.answer_for(q.question_id)
                lg = legacy_grade(ans.answer_text if ans else "", q.model_answer, q.max_marks)
                qrow["legacy"] = lg
                if marks:
                    abs_err_q_legacy.append(abs(lg - qrow["teacher"]))
            row["questions"][q.question_id] = qrow
        if marks:
            row["teacher_total"] = teacher_total
            abs_err_total.append(abs(result.total - teacher_total))
        if legacy:
            row["legacy_total"] = sum(qr["legacy"] for qr in row["questions"].values())
            if marks:
                abs_err_total_legacy.append(abs(row["legacy_total"] - teacher_total))
        rows.append(row)

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    summary = {
        "manifest": str(manifest_path), "synthetic": manifest.get("synthetic", False),
        "marks_by": manifest.get("marks_by"), "items": len(rows), "graded_from": "ocr" if from_ocr else "transcription",
        "model": settings.gemini_reasoning_model, "reference_docs": [p.name for p in reference],
        "sentinel": {"mae_question": mean(abs_err_q), "mae_total": mean(abs_err_total),
                     "level_agreement": (sum(a == b for a, b in zip(levels_teacher, levels_model)) / len(levels_teacher)
                                         if levels_teacher else None),
                     "levels_compared": len(levels_teacher), "kappa": kappa(levels_teacher, levels_model)},
        "legacy": ({"mae_question": mean(abs_err_q_legacy), "mae_total": mean(abs_err_total_legacy)} if legacy else None),
        "exit_gate": {"awarded_criteria": awarded, "awarded_with_verified_evidence": awarded_verified,
                      "rubric_consistency_problems": gate_problems},
        "avg_seconds": mean([r["seconds"] for r in rows]),
    }
    return {"summary": summary, "rows": rows}


def fmt(x, digits=2):
    return "n/a" if x is None else f"{x:.{digits}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--rubric", type=Path, required=True)
    ap.add_argument("--reference", nargs="*", type=Path, default=[])
    ap.add_argument("--from-ocr", action="store_true", help="run the document agent on the pages instead of using the transcription")
    ap.add_argument("--legacy", action="store_true", help="also score with the legacy SBERT-threshold grader")
    ap.add_argument("--model", help="Gemini reasoning model id")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    rubric = load_rubric(args.rubric)
    out = run(args.manifest, rubric, args.reference, from_ocr=args.from_ocr, legacy=args.legacy, model=args.model)
    s = out["summary"]
    print(f"\nGraded from {s['graded_from']} with {s['model']}; reference docs: {s['reference_docs'] or 'none'}")
    for r in out["rows"]:
        per_q = "  ".join(f"{q}: S={v['sentinel']:g}" + (f"/T={v['teacher']:g}" if 'teacher' in v else "")
                          + (f"/L={v['legacy']:g}" if 'legacy' in v else "") for q, v in r["questions"].items())
        print(f"  {r['item']:28} {r['seconds']:6.1f}s  total S={r['sentinel_total']:g}"
              + (f" T={r['teacher_total']:g}" if r["teacher_total"] is not None else "")
              + (f" L={r['legacy_total']:g}" if r["legacy_total"] is not None else "")
              + f"/{r['max_total']:g}   {per_q}" + (f"   flags={r['flags']}" if r["flags"] else ""))

    print("\n| Grader | MAE per question | MAE per total | Exact level agreement | Cohen's κ |")
    print("|---|---|---|---|---|")
    se = s["sentinel"]
    agree = f"{se['level_agreement']:.0%} ({se['levels_compared']} criteria)" if se["level_agreement"] is not None else "n/a"
    print(f"| SentinelEdu Agent 2 | {fmt(se['mae_question'])} | {fmt(se['mae_total'])} | {agree} | {fmt(se['kappa'])} |")
    if s["legacy"]:
        print(f"| Legacy SBERT threshold | {fmt(s['legacy']['mae_question'])} | {fmt(s['legacy']['mae_total'])} | n/a (no levels) | n/a |")
    g = s["exit_gate"]
    print(f"\nExit gate: {g['awarded_with_verified_evidence']}/{g['awarded_criteria']} awarded criteria have verified evidence; "
          f"rubric consistency problems: {len(g['rubric_consistency_problems'])}")
    for p in g["rubric_consistency_problems"]:
        print("  !!", p)
    if s["synthetic"]:
        print("\nNOTE: synthetic set. Teacher marks here were assigned by:", s["marks_by"] or "unknown")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
    return 0 if not g["rubric_consistency_problems"] else 1


if __name__ == "__main__":
    sys.exit(main())

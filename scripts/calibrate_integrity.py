"""Calibrate the near-duplicate thresholds (SENTINELEDU_TASKS.md 7.3, DECISIONS D-11).

Input: a calibration JSON with typed answers per student, planted copied pairs, and control pairs
that must NOT be flagged (e.g. both students copied the textbook). Every pair of students is compared
per question, with the same similarity, reference-margin and short-answer rules as the live scan
(`services/duplicate.py`). The script then reports precision/recall for the current settings and a
grid of thresholds, and recommends the setting that finds every planted pair with the fewest false
positives (closest to the current defaults on ties).

Usage:
  python scripts/calibrate_integrity.py tests/fixtures/integrity/calibration.json --json docs/phase7_calibration.json
"""
import argparse
import itertools
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentineledu.config import PROJECT_ROOT, Settings  # noqa: E402
from sentineledu.services import embeddings  # noqa: E402
from sentineledu.services.duplicate import classify_pair, clean_text, reference_similarity, token_count  # noqa: E402
from sentineledu.services.rag import RagStore  # noqa: E402
from sentineledu.services.rubric import load_rubric  # noqa: E402

GRID = {
    "near_dup_threshold": [0.88, 0.90, 0.92, 0.94],
    "similar_threshold": [0.80, 0.82, 0.85, 0.88],
    "integrity_margin": [0.05, 0.08, 0.10, 0.12, 0.15],
}


def pair_table(cal: dict, settings: Settings) -> list[dict]:
    rubric = load_rubric(PROJECT_ROOT / cal["rubric"])
    rag = RagStore(settings, root=Path(tempfile.mkdtemp()) / "indexes")
    if cal.get("reference"):
        rag.ingest("calibration", [PROJECT_ROOT / r for r in cal["reference"]])
    students = cal["students"]
    planted = {(p["question_id"], *sorted((p["a"], p["b"]))) for p in cal["planted"]}
    controls = {(c["question_id"], *sorted((c["a"], c["b"]))) for c in cal.get("controls", [])}
    rows = []
    for q in rubric.questions:
        ids = [s for s in sorted(students) if students[s].get(q.question_id, "").strip()]
        if len(ids) < 2:
            continue
        texts = [students[s][q.question_id] for s in ids]
        vecs = embeddings.encode([clean_text(t) for t in texts] + [q.model_answer], settings)
        model_vec = vecs[-1]
        refs = [reference_similarity(v, t, q, "calibration", rag, settings, model_vec) for v, t in zip(vecs[:-1], texts)]
        for (i, a), (j, b) in itertools.combinations(enumerate(ids), 2):
            key = (q.question_id, a, b)
            rows.append({"question_id": q.question_id, "a": a, "b": b,
                         "similarity": embeddings.cosine(vecs[i], vecs[j]), "ref_a": refs[i], "ref_b": refs[j],
                         "tokens_a": token_count(texts[i]), "tokens_b": token_count(texts[j]),
                         "planted": key in planted, "control": key in controls})
    return rows


def evaluate(rows: list[dict], settings: Settings) -> dict:
    flagged = [r for r in rows if classify_pair(r["similarity"], r["ref_a"], r["ref_b"], r["tokens_a"], r["tokens_b"],
                                                settings).code]
    tp = sum(r["planted"] for r in flagged)
    planted = sum(r["planted"] for r in rows)
    return {"flagged": len(flagged), "true_positives": tp, "false_positives": len(flagged) - tp,
            "planted": planted, "recall": tp / planted if planted else None,
            "precision": tp / len(flagged) if flagged else None,
            "controls_flagged": [f"{r['question_id']}:{r['a']}-{r['b']}" for r in flagged if r["control"]],
            "false_positive_pairs": [f"{r['question_id']}:{r['a']}-{r['b']}" for r in flagged if not r["planted"]]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("calibration", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    cal = json.loads(args.calibration.read_text())
    base = Settings(_env_file=None, offline_mode=True)
    rows = pair_table(cal, base)

    print("\nPer-pair numbers (planted pairs, controls, and the 5 most similar other pairs):")
    print(f"{'pair':16} {'kind':8} {'sim':>6} {'margin':>7} {'words':>9}")
    others = sorted((r for r in rows if not r["planted"] and not r["control"]), key=lambda r: -r["similarity"])[:5]
    for r in [r for r in rows if r["planted"]] + [r for r in rows if r["control"]] + others:
        kind = "PLANTED" if r["planted"] else "control" if r["control"] else "other"
        margin = r["similarity"] - max(r["ref_a"], r["ref_b"])
        print(f"{r['question_id']}:{r['a']}-{r['b']:9} {kind:8} {r['similarity']:6.3f} {margin:7.3f} {r['tokens_a']:4d}/{r['tokens_b']:<4d}")

    current = evaluate(rows, base)
    grid = []
    for near, similar, margin in itertools.product(*GRID.values()):
        if similar > near:
            continue
        s = base.model_copy(update={"near_dup_threshold": near, "similar_threshold": similar, "integrity_margin": margin})
        grid.append({"near_dup_threshold": near, "similar_threshold": similar, "integrity_margin": margin, **evaluate(rows, s)})
    full = [g for g in grid if g["recall"] == 1.0]
    distance = lambda g: (abs(g["near_dup_threshold"] - base.near_dup_threshold) + abs(g["similar_threshold"] - base.similar_threshold)
                          + abs(g["integrity_margin"] - base.integrity_margin))
    best = min(full, key=lambda g: (g["false_positives"], distance(g))) if full else None

    def line(name, r, near, sim, mar):
        prec = f"{r['precision']:.2f}" if r["precision"] is not None else "n/a"
        print(f"| {name} | {near} | {sim} | {mar} | {r['true_positives']}/{r['planted']} | {r['false_positives']} | {prec} | "
              f"{', '.join(r['controls_flagged']) or 'none'} |")

    pairs = len(rows)
    print(f"\n{pairs} pairs compared ({sum(r['planted'] for r in rows)} planted, {sum(r['control'] for r in rows)} controls).")
    print("| Setting | NEAR_DUP | SIMILAR | MARGIN | Planted found | False positives | Precision | Controls flagged |")
    print("|---|---|---|---|---|---|---|---|")
    line("current defaults", current, base.near_dup_threshold, base.similar_threshold, base.integrity_margin)
    if best:
        line("recommended", best, best["near_dup_threshold"], best["similar_threshold"], best["integrity_margin"])
    ok_settings = [g for g in full if g["false_positives"] <= 1 and not g["controls_flagged"]]
    print(f"\n{len(ok_settings)}/{len(grid)} grid settings find every planted pair with <= 1 false positive and no control flagged.")
    if current["false_positive_pairs"]:
        print("False positives at the current defaults:", ", ".join(current["false_positive_pairs"]))
    if cal.get("synthetic"):
        print("\nNOTE: synthetic calibration set. Re-run on planted pairs in the real gold set (task 1.6).")
    if args.json:
        args.json.write_text(json.dumps({"rows": rows, "current": current, "recommended": best,
                                         "grid_ok": len(ok_settings), "grid_size": len(grid)}, indent=2))
    return 0 if current["recall"] == 1.0 and current["false_positives"] <= 1 and not current["controls_flagged"] else 1


if __name__ == "__main__":
    sys.exit(main())

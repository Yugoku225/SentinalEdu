"""Run the unchanged legacy evaluator on a fixed sample and save its outputs as JSON.

Used by Phase 1 (task 1.5) to confirm the new environment reproduces the old
results, and by Phase 9 (task 9.7) as a regression check.

Usage:
    python scripts/legacy_baseline.py --out tests/fixtures/legacy_baseline.json
"""
import argparse
import json
import os
import platform
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # the legacy code resolves my_trocr_model/ relative to the cwd

DEFAULT_IMAGE = "data/sample_images/sample1.png"
DEFAULT_KEY = "tests/fixtures/legacy/sample1_key.pdf"


def package_versions():
    import importlib.metadata as md
    names = ["torch", "transformers", "sentence-transformers", "numpy",
             "opencv-python", "nltk", "pyspellchecker", "PyPDF2", "streamlit"]
    return {n: md.version(n) for n in names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--max-marks", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from src.pdf_utils import extract_text_from_pdf
    from src.pipeline_runner import run_full_assessment

    model_answer = extract_text_from_pdf(args.key)
    results = run_full_assessment(args.image, model_answer, max_marks=args.max_marks)

    baseline = {
        "inputs": {"image": args.image, "key": args.key, "max_marks": args.max_marks,
                   "model_answer": model_answer},
        "environment": {"python": platform.python_version(), "packages": package_versions()},
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(baseline, f, indent=2, default=float)
    print("Saved baseline to " + args.out)


if __name__ == "__main__":
    main()

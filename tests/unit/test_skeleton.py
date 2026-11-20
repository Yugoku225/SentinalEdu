from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_sentineledu_package_imports():
    import sentineledu
    import sentineledu.agents, sentineledu.schemas, sentineledu.services, sentineledu.workflow  # noqa: F401


def test_legacy_files_still_present():
    for rel in ["app.py", "main.py", "src/pipeline_runner.py", "src/ocr_pipeline.py"]:
        assert (ROOT / rel).is_file(), rel

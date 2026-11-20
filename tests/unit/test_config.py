from pathlib import Path

import pytest
from pydantic import ValidationError

from sentineledu.config import PROJECT_ROOT, Settings

ENV_KEYS = ["GEMINI_API_KEY", "GEMINI_OCR_MODEL", "OFFLINE_MODE", "STORAGE_DIR"]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_defaults():
    s = Settings(_env_file=None, gemini_api_key="k")
    assert s.gemini_ocr_model == "gemini-3.5-flash"
    assert s.gemini_fallback_model == "gemini-3.7-flash"
    assert s.embedding_model == "all-MiniLM-L6-v2"
    assert s.rag_top_k == 4
    assert s.storage_dir == PROJECT_ROOT / "storage"
    assert s.trocr_model_path == PROJECT_ROOT / "my_trocr_model"


def test_missing_key_fails_fast_with_clear_message():
    with pytest.raises(ValidationError, match="GEMINI_API_KEY is not set"):
        Settings(_env_file=None)


def test_blank_key_counts_as_missing():
    with pytest.raises(ValidationError, match="GEMINI_API_KEY"):
        Settings(_env_file=None, gemini_api_key="   ")


def test_offline_mode_needs_no_key():
    s = Settings(_env_file=None, offline_mode=True)
    with pytest.raises(RuntimeError, match="unavailable"):
        s.api_key()


def test_reads_env_file_and_ignores_blank_values(tmp_path):
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=abc123\nGEMINI_OCR_MODEL=\nRAG_TOP_K=6\n")
    s = Settings(_env_file=env)
    assert s.api_key() == "abc123"
    assert s.gemini_ocr_model == "gemini-3.5-flash"  # blank value -> default
    assert s.rag_top_k == 6


def test_key_is_not_leaked_in_repr():
    s = Settings(_env_file=None, gemini_api_key="super-secret")
    assert "super-secret" not in repr(s)


def test_threshold_order_validated():
    with pytest.raises(ValidationError, match="SIMILAR_THRESHOLD"):
        Settings(_env_file=None, offline_mode=True, near_dup_threshold=0.8, similar_threshold=0.9)


def test_env_example_matches_settings_fields():
    example = (PROJECT_ROOT / ".env.example").read_text()
    keys = {line.split("=", 1)[0].strip().lower() for line in example.splitlines()
            if "=" in line and not line.lstrip().startswith("#")}
    assert keys == set(Settings.model_fields)


def test_absolute_paths_kept(tmp_path: Path):
    s = Settings(_env_file=None, offline_mode=True, storage_dir=tmp_path)
    assert s.storage_dir == tmp_path
    assert s.cache_dir == tmp_path / "cache" / "llm"

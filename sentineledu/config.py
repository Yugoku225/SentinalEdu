"""Application settings, loaded from environment variables and the project's .env file."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # a blank "GEMINI_OCR_MODEL=" falls back to the default
        extra="ignore",
    )

    # --- Gemini (chosen in Phase 3, see SENTINELEDU_DECISIONS.md D-03) ---
    gemini_api_key: SecretStr | None = None
    gemini_ocr_model: str = "gemini-3.5-flash"
    gemini_reasoning_model: str = "gemini-3.5-flash"
    gemini_fallback_model: str = "gemini-3.7-flash"
    gemini_thinking_level: Literal["low", "high"] | None = "low"
    gemini_timeout_s: float = 120.0

    # --- Local models ---
    embedding_model: str = "all-MiniLM-L6-v2"
    trocr_model_path: Path = Path("my_trocr_model")

    # --- Behaviour ---
    offline_mode: bool = False
    extraction_review_threshold: float = Field(0.6, ge=0, le=1)
    near_dup_threshold: float = Field(0.92, ge=0, le=1)
    similar_threshold: float = Field(0.85, ge=0, le=1)
    integrity_margin: float = Field(0.10, ge=0, le=1)  # pair must be this much closer to each other than to the reference
    integrity_min_tokens: int = Field(15, ge=1)  # shorter answers are not compared (they converge naturally)
    rag_top_k: int = Field(4, ge=1)
    storage_dir: Path = Path("storage")

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        if not self.offline_mode and not (self.gemini_api_key and self.gemini_api_key.get_secret_value().strip()):
            raise ValueError(
                "GEMINI_API_KEY is not set. Add it to the .env file in the project root "
                "(see .env.example), or set OFFLINE_MODE=true to run without Gemini."
            )
        if self.similar_threshold > self.near_dup_threshold:
            raise ValueError("SIMILAR_THRESHOLD must be <= NEAR_DUP_THRESHOLD.")
        # Relative paths are relative to the project root, not the current directory.
        if not self.trocr_model_path.is_absolute():
            self.trocr_model_path = PROJECT_ROOT / self.trocr_model_path
        if not self.storage_dir.is_absolute():
            self.storage_dir = PROJECT_ROOT / self.storage_dir
        return self

    @property
    def cache_dir(self) -> Path:
        return self.storage_dir / "cache" / "llm"

    def api_key(self) -> str:
        if self.gemini_api_key is None:
            raise RuntimeError("Gemini is unavailable: OFFLINE_MODE is on and no GEMINI_API_KEY is set.")
        return self.gemini_api_key.get_secret_value().strip()


@lru_cache
def get_settings() -> Settings:
    return Settings()

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture
def offline_settings(tmp_path):
    """Settings that never read the real .env and need no API key."""
    from sentineledu.config import Settings
    return Settings(_env_file=None, offline_mode=True, storage_dir=tmp_path / "storage")


@pytest.fixture
def live_settings():
    from sentineledu.config import Settings
    try:
        settings = Settings()
    except ValueError:
        pytest.skip("GEMINI_API_KEY not configured")
    if settings.offline_mode:
        pytest.skip("OFFLINE_MODE is on")
    return settings

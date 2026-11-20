"""Print the configured Gemini model IDs and check that each one responds (SENTINELEDU_TASKS.md 2.3).

Usage:  python scripts/check_gemini_models.py [--list]
  --list   also list every Gemini model this API key can see
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google import genai  # noqa: E402

from sentineledu.config import get_settings  # noqa: E402


def main() -> int:
    settings = get_settings()
    client = genai.Client(api_key=settings.api_key())

    if "--list" in sys.argv:
        print("Models visible to this key:")
        for m in sorted(client.models.list(), key=lambda m: m.name):
            if "generateContent" in (m.supported_actions or []) and "gemini" in m.name:
                print("  ", m.name.removeprefix("models/"))
        print()

    roles = {
        "GEMINI_OCR_MODEL": settings.gemini_ocr_model,
        "GEMINI_REASONING_MODEL": settings.gemini_reasoning_model,
        "GEMINI_FALLBACK_MODEL": settings.gemini_fallback_model,
    }
    failures = 0
    checked: dict[str, str] = {}
    for role, model_id in roles.items():
        if model_id not in checked:
            start = time.monotonic()
            try:
                reply = client.models.generate_content(model=model_id, contents="Reply with exactly: OK")
                checked[model_id] = f"OK    {time.monotonic() - start:5.1f}s  reply={reply.text.strip()[:10]!r}"
            except Exception as e:
                checked[model_id] = f"FAIL  {time.monotonic() - start:5.1f}s  {str(e)[:90]}"
        failures += checked[model_id].startswith("FAIL")
        print(f"{role:24} {model_id:20} {checked[model_id]}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

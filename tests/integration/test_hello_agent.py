"""Live Gemini test (SENTINELEDU_TASKS.md 2.5). Run with: pytest -m live"""
from pathlib import Path

import pytest

from sentineledu.agents.hello_agent import HelloOut, build_hello_agent, summarise_image, summarise_text
from sentineledu.agents.runner import ResponseCache

IMAGE = Path(__file__).resolve().parents[1] / "fixtures" / "images" / "printed_sample.png"
TEXT = "Photosynthesis converts light energy into chemical energy in plants."


@pytest.mark.live
def test_text_returns_typed_object(live_settings):
    result = summarise_text(TEXT, agent=build_hello_agent(live_settings))
    assert isinstance(result.output, HelloOut)  # a Pydantic object, not a string
    assert 7 <= result.output.word_count <= 11
    assert result.output.summary.strip()


@pytest.mark.live
def test_image_input_reads_visible_text(live_settings):
    result = summarise_image(IMAGE, agent=build_hello_agent(live_settings))
    assert isinstance(result.output, HelloOut)
    assert result.output.word_count > 20  # the sample page has ~80 printed words
    assert any(w in result.output.summary.lower() for w in ("fox", "dog", "test", "text"))


@pytest.mark.live
def test_second_identical_call_is_served_from_cache(live_settings, tmp_path):
    cache = ResponseCache(tmp_path)
    agent = build_hello_agent(live_settings)
    first = summarise_text(TEXT, agent=agent, cache=cache)
    second = summarise_text(TEXT, agent=agent, cache=cache)
    assert not first.cached and second.cached
    assert second.output == first.output

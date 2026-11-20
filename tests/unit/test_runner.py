import pytest
from pydantic import BaseModel

from fakes import FakeAgent, ProviderError
from sentineledu.agents.hello_agent import HelloOut, summarise_text
from sentineledu.agents.runner import (
    LLMCallError,
    ResponseCache,
    StructuredOutputError,
    is_retryable,
    run_structured,
)

OK = {"summary": "s", "word_count": 3}


class NoSleep:
    def __init__(self):
        self.delays = []

    def __call__(self, s):
        self.delays.append(s)


def run(agent, cache=None, sleep=None, **kw):
    return run_structured(agent, "prompt", HelloOut, prompt_version="t1", cache=cache,
                          sleep=sleep or NoSleep(), **kw)


def test_dict_content_is_validated_into_schema():
    result = run(FakeAgent(OK))
    assert isinstance(result.output, HelloOut)
    assert result.output.word_count == 3
    assert result.attempts == 1 and not result.cached
    assert result.input_tokens == 10


def test_json_string_content_is_parsed():
    assert run(FakeAgent('{"summary": "s", "word_count": 1}')).output.word_count == 1


def test_retries_transient_errors_with_exponential_backoff():
    sleep = NoSleep()
    agent = FakeAgent(ProviderError("503 UNAVAILABLE high demand"),
                      ProviderError("429 RESOURCE_EXHAUSTED"), OK)
    result = run(agent, sleep=sleep)
    assert result.attempts == 3
    assert sleep.delays == [2.0, 4.0]


def test_non_retryable_error_fails_immediately():
    agent = FakeAgent(ProviderError("404 NOT_FOUND model retired"), OK)
    with pytest.raises(LLMCallError, match="non-retryable"):
        run(agent)
    assert len(agent.calls) == 1


def test_error_text_is_never_returned_as_an_answer():
    # Agno returns status=error with the error message as content; it must not be parsed as output.
    agent = FakeAgent(*[ProviderError("503 UNAVAILABLE")] * 4)
    with pytest.raises(LLMCallError, match="after 4 attempts"):
        run(agent)


def test_raised_exceptions_are_handled():
    agent = FakeAgent(TimeoutError("request timed out"), OK)
    assert run(agent).attempts == 2
    with pytest.raises(LLMCallError, match="non-retryable"):
        run(FakeAgent(ValueError("bad argument")))


def test_schema_mismatch_retries_then_raises():
    agent = FakeAgent({"summary": "s"}, OK)  # first response missing word_count
    assert run(agent).attempts == 2
    with pytest.raises(LLMCallError):
        run(FakeAgent({"summary": "s"}, {"summary": "s"}), max_attempts=2)


def test_cache_hit_skips_model_call(tmp_path):
    cache = ResponseCache(tmp_path)
    first = run(FakeAgent(OK), cache=cache)
    agent2 = FakeAgent()  # no responses queued: any call would fail
    second = run(agent2, cache=cache)
    assert second.cached and second.output == first.output
    assert agent2.calls == []


def test_cache_key_depends_on_prompt_version_model_and_media(tmp_path):
    img_a, img_b = tmp_path / "a.png", tmp_path / "b.png"
    img_a.write_bytes(b"A")
    img_b.write_bytes(b"B")

    class Img:
        def __init__(self, p):
            self.filepath, self.content = p, None

    k = lambda **kw: ResponseCache.key(**{"prompt_version": "v1", "model_id": "m", "schema": HelloOut,
                                          "prompt": "p", "media": [Img(img_a)], **kw})
    base = k()
    assert base == k()
    assert base != k(prompt_version="v2")
    assert base != k(model_id="m2")
    assert base != k(media=[Img(img_b)])
    assert base != k(prompt="p2")


def test_stale_cache_entry_is_ignored(tmp_path):
    class Other(BaseModel):
        x: int

    cache = ResponseCache(tmp_path)
    key = ResponseCache.key(prompt_version="v", model_id="m", schema=HelloOut, prompt="p")
    cache.put(key, Other(x=1), "m")
    assert cache.get(key, HelloOut) is None
    (tmp_path / f"{key}.json").write_text('{"summary": "old format", "word_count": 1}')
    assert cache.get(key, HelloOut) is None


def test_is_retryable():
    assert is_retryable("503 UNAVAILABLE")
    assert is_retryable("429 RESOURCE_EXHAUSTED")
    assert not is_retryable("404 NOT_FOUND")
    assert not is_retryable("400 INVALID_ARGUMENT")


def test_hello_agent_with_fake_fixture():
    agent = FakeAgent.from_fixture("hello_text")
    result = summarise_text("Photosynthesis converts light energy into chemical energy in plants.", agent=agent)
    assert result.output.word_count == 9
    assert "Photosynthesis" in agent.calls[0]["input"]
    assert agent.calls[0]["output_schema"] is HelloOut


def test_structured_output_error_is_llm_call_error():
    assert issubclass(StructuredOutputError, LLMCallError)


def test_reports_fallback_model_when_agno_switched():
    class FallbackAgent(FakeAgent):
        def run(self, input, **kwargs):
            out = super().run(input, **kwargs)
            self.fallback_events.append("gemini-3.5-flash")  # what base.build_agent's callback does
            return out

    agent = FallbackAgent(OK, model_id="gemini-3.7-flash")
    agent.fallback_events = []
    result = run(agent)
    assert result.fallback_used
    assert result.model_id == "gemini-3.5-flash"

    plain = run(FakeAgent(OK, model_id="gemini-3.7-flash"))
    assert not plain.fallback_used and plain.model_id == "gemini-3.7-flash"


def test_build_agent_wires_fallback_for_errors_and_rate_limits(offline_settings):
    from sentineledu.agents.base import build_agent
    settings = offline_settings.model_copy(update={"gemini_api_key": __import__("pydantic").SecretStr("k")})
    agent = build_agent(name="t", instructions="x", output_schema=HelloOut, settings=settings)
    primary, fallback = settings.gemini_reasoning_model, settings.gemini_fallback_model
    assert agent.model.id == primary
    cfg = agent.fallback_config
    assert [m.id for m in cfg.on_error] == [fallback]
    assert [m.id for m in cfg.on_rate_limit] == [fallback]
    assert agent.telemetry is False
    assert agent.model.temperature == 0
    cfg.callback(primary, fallback, RuntimeError("503"))
    assert agent.fallback_events == [fallback]

    same = build_agent(name="t", instructions="x", output_schema=HelloOut, model_id=fallback, settings=settings)
    assert same.fallback_config is None  # never fall back to the same model


def test_cache_remembers_which_model_actually_answered(tmp_path):
    class FallbackAgent(FakeAgent):
        def run(self, input, **kwargs):
            out = super().run(input, **kwargs)
            self.fallback_events.append("gemini-3.5-flash")
            return out

    cache = ResponseCache(tmp_path)
    agent = FallbackAgent(OK, model_id="gemini-3.7-flash")
    agent.fallback_events = []
    run(agent, cache=cache)
    hit = run(FakeAgent(model_id="gemini-3.7-flash"), cache=cache)
    assert hit.cached and hit.model_id == "gemini-3.5-flash" and hit.fallback_used

"""Run an Agno agent and return a validated Pydantic object.

Adds what Agno does not do for us:
- Agno does not raise on provider errors; it returns RunOutput(status=error) with the
  error text as content. We check the status so an error message is never mistaken
  for an answer.
- Retry with exponential backoff on transient errors (429/500/503/timeouts) only.
- On-disk cache keyed by (prompt_version, model_id, schema, prompt, media bytes), so the
  same submission is never sent to Gemini twice.
- Logs latency and token usage per call.
"""
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Sequence, TypeVar

from pydantic import BaseModel, ValidationError

log = logging.getLogger("sentineledu.llm")

T = TypeVar("T", bound=BaseModel)

_RETRYABLE_MARKERS = ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE",
                      "INTERNAL", "DEADLINE_EXCEEDED", "high demand", "timed out", "timeout",
                      "ConnectError", "ConnectionError", "RemoteProtocolError")


class LLMCallError(RuntimeError):
    """The model call failed (after retries, or with a non-retryable error)."""


class StructuredOutputError(LLMCallError):
    """The model answered, but not in the requested schema."""


@dataclass
class LLMResult(Generic[T]):
    output: T
    model_id: str
    cached: bool
    attempts: int
    duration_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    fallback_used: bool = False


def is_retryable(message: str) -> bool:
    return any(marker.lower() in message.lower() for marker in _RETRYABLE_MARKERS)


def _media_digest(item: Any) -> str:
    content = getattr(item, "content", None)
    if isinstance(content, (bytes, bytearray)):
        return hashlib.sha256(content).hexdigest()
    filepath = getattr(item, "filepath", None)
    if filepath:
        return hashlib.sha256(Path(filepath).read_bytes()).hexdigest()
    return hashlib.sha256(str(getattr(item, "url", item)).encode()).hexdigest()


class ResponseCache:
    """One JSON file per cached response under `directory`."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    @staticmethod
    def key(*, prompt_version: str, model_id: str, schema: type[BaseModel], prompt: str,
            media: Sequence[Any] = ()) -> str:
        payload = {
            "prompt_version": prompt_version,
            "model_id": model_id,
            "schema": f"{schema.__module__}.{schema.__qualname__}",
            "schema_fields": sorted(schema.model_fields),
            "prompt": prompt,
            "media": [_media_digest(m) for m in media],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def get(self, key: str, schema: type[T]) -> tuple[T, str] | None:
        """(output, model_id that actually produced it), or None."""
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text())
            return schema.model_validate(entry["output"]), entry["model_id"]
        except (ValidationError, KeyError, TypeError, json.JSONDecodeError):
            log.warning("Ignoring stale cache entry %s", path.name)
            return None

    def put(self, key: str, value: BaseModel, model_id: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        entry = {"model_id": model_id, "output": value.model_dump(mode="json")}
        (self.directory / f"{key}.json").write_text(json.dumps(entry, indent=2))


def _coerce(content: Any, schema: type[T]) -> T:
    if isinstance(content, schema):
        return content
    try:
        if isinstance(content, BaseModel):
            return schema.model_validate(content.model_dump())
        if isinstance(content, dict):
            return schema.model_validate(content)
        if isinstance(content, str):
            return schema.model_validate_json(content)
    except ValidationError as e:
        raise StructuredOutputError(f"response did not match {schema.__name__}: {e}") from e
    raise StructuredOutputError(f"unexpected response type {type(content).__name__}")


def run_structured(
    agent: Any,
    prompt: str,
    schema: type[T],
    *,
    prompt_version: str,
    images: Sequence[Any] | None = None,
    files: Sequence[Any] | None = None,
    cache: ResponseCache | None = None,
    max_attempts: int = 4,
    base_delay_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> LLMResult[T]:
    """Run `agent` on `prompt` (+ media) and return its output validated as `schema`."""
    images, files = list(images or []), list(files or [])
    model_id = getattr(getattr(agent, "model", None), "id", "unknown")
    name = getattr(agent, "name", None) or "agent"

    key = None
    if cache is not None:
        key = ResponseCache.key(prompt_version=prompt_version, model_id=model_id, schema=schema,
                                prompt=prompt, media=images + files)
        hit = cache.get(key, schema)
        if hit is not None:
            output, produced_by = hit
            log.info("[%s] cache hit (%s, produced by %s)", name, key[:12], produced_by)
            return LLMResult(output=output, model_id=produced_by, cached=True, attempts=0, duration_s=0.0,
                             fallback_used=produced_by != model_id)

    last_error = ""
    for attempt in range(1, max_attempts + 1):
        start = time.monotonic()
        events = getattr(agent, "fallback_events", None)
        events_before = len(events) if events is not None else 0
        try:
            run = agent.run(prompt, images=images or None, files=files or None, output_schema=schema)
        except Exception as e:  # Agno normally returns errors, but be defensive
            last_error = f"{type(e).__name__}: {e}"
            if not is_retryable(last_error):
                raise LLMCallError(f"[{name}] non-retryable error: {last_error[:500]}") from e
        else:
            duration = time.monotonic() - start
            status = str(getattr(getattr(run, "status", None), "value", getattr(run, "status", ""))).lower()
            if status != "error":
                try:
                    output = _coerce(run.content, schema)
                except StructuredOutputError as e:
                    last_error = str(e)  # retry: a second attempt usually parses
                else:
                    metrics = getattr(run, "metrics", None)
                    fallback_used = events is not None and len(events) > events_before
                    # Agno reports the primary model even when a fallback answered.
                    used_model = events[-1] if fallback_used else (getattr(run, "model", None) or model_id)
                    result = LLMResult(
                        output=output, model_id=used_model, cached=False, attempts=attempt,
                        duration_s=duration,
                        input_tokens=getattr(metrics, "input_tokens", None),
                        output_tokens=getattr(metrics, "output_tokens", None),
                        fallback_used=fallback_used,
                    )
                    log.info("[%s] %s%s ok in %.1fs (attempt %d, tokens in=%s out=%s)", name, used_model,
                             " (fallback)" if fallback_used else "", duration, attempt,
                             result.input_tokens, result.output_tokens)
                    if cache is not None and key is not None:
                        cache.put(key, output, used_model)
                    return result
            else:
                last_error = str(run.content)
                if not is_retryable(last_error):
                    raise LLMCallError(f"[{name}] non-retryable model error: {last_error[:500]}")

        if attempt < max_attempts:
            delay = base_delay_s * 2 ** (attempt - 1)
            log.warning("[%s] attempt %d/%d failed (%s); retrying in %.0fs", name, attempt,
                        max_attempts, last_error[:120], delay)
            sleep(delay)

    raise LLMCallError(f"[{name}] failed after {max_attempts} attempts: {last_error[:500]}")

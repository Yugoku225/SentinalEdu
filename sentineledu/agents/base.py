"""Shared factory for SentinelEdu's Agno agents (Gemini model, typed output, no telemetry)."""
import os

from agno.agent import Agent
from agno.models.fallback import FallbackConfig
from agno.models.google import Gemini
from pydantic import BaseModel

from sentineledu.config import Settings, get_settings

# Agno sends anonymous usage telemetry by default. Student data never leaves except to Gemini.
os.environ.setdefault("AGNO_TELEMETRY", "false")


def gemini_model(model_id: str, settings: Settings) -> Gemini:
    return Gemini(
        id=model_id,
        api_key=settings.api_key(),
        temperature=0,
        thinking_level=settings.gemini_thinking_level,
        timeout=settings.gemini_timeout_s,
    )


def build_agent(
    *,
    name: str,
    instructions: str | list[str],
    output_schema: type[BaseModel],
    model_id: str | None = None,
    settings: Settings | None = None,
) -> Agent:
    """Create an agent that returns `output_schema` objects and falls back to a second Gemini model.

    Agno reports the primary model id even when a fallback answered, so fallback activations are
    recorded on `agent.fallback_events` (read by runner.run_structured).
    """
    settings = settings or get_settings()
    model_id = model_id or settings.gemini_reasoning_model
    events: list[str] = []
    fallback = None
    if settings.gemini_fallback_model and settings.gemini_fallback_model != model_id:
        # on_error only covers 5xx/network errors; 429 rate limits need on_rate_limit.
        fallback = FallbackConfig(
            on_error=[gemini_model(settings.gemini_fallback_model, settings)],
            on_rate_limit=[gemini_model(settings.gemini_fallback_model, settings)],
            callback=lambda primary_id, fallback_id, error: events.append(fallback_id),
        )
    agent = Agent(
        name=name,
        model=gemini_model(model_id, settings),
        fallback_config=fallback,
        instructions=instructions,
        output_schema=output_schema,
        markdown=False,
        telemetry=False,
    )
    agent.fallback_events = events
    return agent

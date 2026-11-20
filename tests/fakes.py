"""Offline stand-ins for Agno agents (SENTINELEDU_TASKS.md 2.7)."""
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).parent / "fixtures" / "llm"


class _Status(Enum):
    completed = "completed"
    error = "error"


@dataclass
class _Metrics:
    input_tokens: int = 10
    output_tokens: int = 5


@dataclass
class FakeRunOutput:
    content: Any
    status: _Status = _Status.completed
    model: str = "fake-model"
    metrics: _Metrics = field(default_factory=_Metrics)


@dataclass
class _Model:
    id: str = "fake-model"


class ProviderError:
    """Queue this to simulate Agno returning RunOutput(status=error)."""

    def __init__(self, message: str):
        self.message = message


class FakeAgent:
    """Returns queued responses in order. A response can be a dict/BaseModel/str (success),
    a ProviderError (Agno-style error status) or an Exception (raised)."""

    def __init__(self, *responses: Any, name: str = "fake-agent", model_id: str = "fake-model"):
        self.responses = list(responses)
        self.name = name
        self.model = _Model(model_id)
        self.calls: list[dict] = []

    @classmethod
    def from_fixture(cls, *names: str, **kwargs) -> "FakeAgent":
        return cls(*(json.loads((FIXTURES / f"{n}.json").read_text()) for n in names), **kwargs)

    def run(self, input: Any, **kwargs) -> FakeRunOutput:
        self.calls.append({"input": input, **kwargs})
        if not self.responses:
            raise AssertionError("FakeAgent called more times than responses were queued")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, ProviderError):
            return FakeRunOutput(content=response.message, status=_Status.error)
        return FakeRunOutput(content=response, model=self.model.id)

"""Phase 2 smoke-test agent: proves Agno + Gemini + typed output + image input work end to end.

Run directly:  python -m sentineledu.agents.hello_agent [image_path]
"""
import sys
from pathlib import Path

from agno.media import Image
from pydantic import BaseModel, Field

from sentineledu.agents.base import build_agent
from sentineledu.agents.runner import LLMResult, ResponseCache, run_structured
from sentineledu.config import Settings, get_settings

PROMPT_VERSION = "hello_v1"


class HelloOut(BaseModel):
    summary: str = Field(description="One-sentence summary of the input")
    word_count: int = Field(ge=0, description="Number of words in the input text (or in the text visible in the image)")


INSTRUCTIONS = [
    "You summarise the text you are given, or the text visible in an image.",
    "Count words exactly as they appear. Do not invent text that is not present.",
]


def build_hello_agent(settings: Settings | None = None):
    settings = settings or get_settings()
    return build_agent(name="hello-agent", instructions=INSTRUCTIONS, output_schema=HelloOut,
                       model_id=settings.gemini_ocr_model, settings=settings)


def summarise_text(text: str, *, agent=None, cache: ResponseCache | None = None) -> LLMResult[HelloOut]:
    agent = agent or build_hello_agent()
    return run_structured(agent, f"Summarise this text:\n\n{text}", HelloOut,
                          prompt_version=PROMPT_VERSION, cache=cache)


def summarise_image(image_path: str | Path, *, agent=None, cache: ResponseCache | None = None) -> LLMResult[HelloOut]:
    agent = agent or build_hello_agent()
    return run_structured(agent, "Summarise the text visible in this image.", HelloOut,
                          prompt_version=PROMPT_VERSION, images=[Image(filepath=str(image_path))], cache=cache)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    result = (summarise_image(sys.argv[1]) if len(sys.argv) > 1
              else summarise_text("Photosynthesis converts light energy into chemical energy in plants."))
    print(f"model={result.model_id} cached={result.cached} {result.duration_s:.1f}s")
    print(result.output.model_dump_json(indent=2))

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plyagents.tac.apply.base import Apply, Shortened
from plyagents.tac.configs.components import ComponentParams
from plyagents.tac.configs.model import ModelSpec
from plyagents.tac.utils.calls import Calls
from plyagents.tac.utils.options import read_options

SYSTEM = "You shorten the notes an agent keeps about one situation in a game. Answer with the shortened notes only."
PROMPT = """Shorten these notes to at most {keep} characters. Keep every point and shorten each in proportion; do not add, merge or reinterpret anything.

Notes:
{notes}"""


@dataclass(frozen=True, eq=True)
class CompressParams(ComponentParams):
    model: ModelSpec | None = None

    @classmethod
    def from_string(cls, params_string: str) -> CompressParams:
        options = read_options(params_string, {"model"}, "compress")
        return cls(ModelSpec.from_string(options["model"]) if "model" in options else None)


class CompressApply(Apply):
    # an LLM shortens the old entry to its share, every point in proportion (one call); an answer over the share is cut
    def __init__(self, model: ModelSpec) -> None:
        self.model = model

    async def shorten(self, old: str, keep: int, calls: Calls, role: str, about: dict[str, Any] | None = None) -> Shortened:
        if len(old) <= keep:
            return Shortened(old, False)
        if keep == 0:
            return Shortened("", False)
        text, call = await calls.indexed_text(role, self.model.config, SYSTEM, PROMPT.format(keep=keep, notes=old), about)
        text = text.strip()
        return Shortened(text[:keep], len(text) > keep, call)

    def describe(self) -> str:
        return f"compress: {self.model.to_string()} shortens the entry to (1 - alpha) B characters, then the lesson is appended"

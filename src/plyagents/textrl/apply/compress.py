from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from plyagents.textrl.apply.base import Apply, Shortened
from plyagents.textrl.configs.components import ComponentParams
from plyagents.textrl.configs.model import ModelSpec
from plyagents.textrl.enums import ReduceType
from plyagents.textrl.model.memory import MemoryWrite
from plyagents.textrl.utils.calls import Calls
from plyagents.textrl.utils.options import read_options

SYSTEM = "You shorten the notes an agent keeps about one situation in a game. Answer with the shortened notes only."
PROMPT = """Shorten these notes to at most {keep} characters. Keep every point and shorten each in proportion; keep any count of the lessons that support a point \
as it is written; do not add, merge or reinterpret anything.

Notes:
{notes}"""
# reduce=mean: the lessons of one key in a batch combined into one, with the old entry shortened in the same call when it must be. The mean in text: each
# point with the number of lessons that support it, so 3 against 1 is not written as 2 against 2 (the same instruction for every game: part of Apply)
SYSTEM_REDUCE = "You keep the notes an agent has about one situation in a game, and combine new lessons about it into one."
COMBINE = """Combine these {count} lessons about the same situation into one text{limit}. Keep every point they make and state how many of the {count} lessons \
support it ("k of {count} lessons: ..."); where they disagree, give each side with its count. Add nothing they do not say. Answer with the combined text only.

Lessons:
{lessons}"""
COMBINE_AND_SHORTEN = """First shorten the notes to at most {keep} characters: keep every point and shorten each in proportion; keep any count of the lessons \
that support a point as it is written; do not add, merge or reinterpret anything. Then combine the {count} new lessons about the same situation into one \
text{limit}: keep every point they make and state how many of the {count} lessons support it ("k of {count} lessons: ..."); where they disagree, give each side \
with its count. Add nothing they do not say.

Notes:
{notes}

New lessons:
{lessons}"""


class _Combined(BaseModel):
    kept: str
    combined: str


@dataclass(frozen=True, eq=True)
class CompressParams(ComponentParams):
    model: ModelSpec | None = None
    # what a write makes of several gradients of a key: join them (their sum), or combine them into their mean in one call
    reduce: ReduceType = ReduceType.JOIN

    @classmethod
    def from_string(cls, params_string: str) -> CompressParams:
        options = read_options(params_string, {"model", "reduce"}, "compress")
        reduce = ReduceType.from_value(options.get("reduce", cls.reduce.value))
        if reduce is None:
            raise ValueError(f"Invalid compress reduce {options['reduce']!r}; choose from {ReduceType.values()}")
        return cls(ModelSpec.from_string(options["model"]) if "model" in options else None, reduce)


def _numbered(parts: tuple[str, ...]) -> str:
    return "\n".join(f"{number}. {part}" for number, part in enumerate(parts, start=1))


class CompressApply(Apply):
    # an LLM shortens the old entry to its share, every point in proportion (one call); an answer over the share is cut. With reduce=mean the
    # gradients of a key are combined in that same call (one call when the entry needs shortening, else one call to combine)
    def __init__(self, model: ModelSpec, reduce: ReduceType = ReduceType.JOIN) -> None:
        self.model = model
        self.reduce_type = reduce

    async def shorten(self, old: str, keep: int, calls: Calls, role: str, about: dict[str, Any] | None = None) -> Shortened:
        if len(old) <= keep:
            return Shortened(old, False)
        if keep == 0:
            return Shortened("", False)
        answer = await calls.text(role, self.model.config, SYSTEM, PROMPT.format(keep=keep, notes=old), about)
        text = answer.output.strip()
        return Shortened(text[:keep], len(text) > keep, answer.index)

    async def reduce(self, parts: tuple[str, ...], share: int | None, calls: Calls, role: str, about: dict[str, Any]) -> str:
        if self.reduce_type is ReduceType.JOIN:
            return await super().reduce(parts, share, calls, role, about)
        limit = f" of at most {share} characters" if share is not None else ""
        return (await calls.text(role, self.model.config, SYSTEM_REDUCE, COMBINE.format(count=len(parts), limit=limit, lessons=_numbered(parts)), about)).output.strip()

    async def merge(
        self, key: str, old: str, parts: tuple[str, ...], alpha: float, budget: int | None, calls: Calls, role: str, steps: tuple[int, ...], writes_before: int
    ) -> MemoryWrite:
        keep = int((1 - alpha) * budget) if budget is not None else None
        # one call for both only when there are gradients to combine and the entry must be shortened by an LLM; otherwise as every Apply
        if self.reduce_type is ReduceType.JOIN or len(parts) == 1 or keep is None or keep == 0 or len(old) <= keep:
            return await super().merge(key, old, parts, alpha, budget, calls, role, steps, writes_before)
        share = budget - keep if budget is not None else None
        prompt = COMBINE_AND_SHORTEN.format(keep=keep, count=len(parts), limit=f" of at most {share} characters" if share is not None else "", notes=old, lessons=_numbered(parts))
        about = {"key": key, "steps": list(steps)}
        answer = await calls.structured(role, self.model.config, SYSTEM_REDUCE, prompt, _Combined, about)
        kept = answer.output.kept.strip()
        shortened = Shortened(kept[:keep], len(kept) > keep, answer.index)
        return self.assemble(key, old, parts, alpha, budget, keep, shortened, answer.output.combined.strip(), steps, writes_before)

    def describe(self) -> str:
        reduce = "; the lessons of a key are combined into their mean" if self.reduce_type is ReduceType.MEAN else ""
        return f"compress: {self.model.to_string()} shortens the entry to (1 - alpha) B characters, then the lesson is appended{reduce}"

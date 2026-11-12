from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plyagents.textrl.apply.base import Apply, Shortened, changed
from plyagents.textrl.configs.components import ComponentParams
from plyagents.textrl.configs.model import ModelSpec
from plyagents.textrl.model.memory import MemoryWrite
from plyagents.textrl.utils.calls import Calls
from plyagents.textrl.utils.options import read_options

SYSTEM = "You keep the notes an agent has about one situation in a game, and rewrite them with new lessons. Answer with the new notes only."
PROMPT = """Rewrite these notes so that they take in the new {lessons_word}. Change about {percent}% of the notes and keep the rest as it is{limit}; the notes may \
take any form that helps the agent, for example counts, cases or rules. Keep any count of the lessons that support a point unless the new {lessons_word} \
change it.{support}

Notes:
{notes}

New {lessons_word}:
{lessons}"""
# with several new lessons, their mean in text: each point with the number of lessons that support it (as compress[reduce=mean] combines them)
SUPPORT = ' State how many of the {count} new lessons support each point ("k of {count} lessons: ..."); where they disagree, give each side with its count.'


@dataclass(frozen=True, eq=True)
class RewriteParams(ComponentParams):
    model: ModelSpec | None = None

    @classmethod
    def from_string(cls, params_string: str) -> RewriteParams:
        options = read_options(params_string, {"model"}, "rewrite")
        return cls(ModelSpec.from_string(options["model"]) if "model" in options else None)


class RewriteApply(Apply):
    # the free write: an LLM rewrites the whole entry with the gradients of the key, told the step alpha as the share of the entry to change and the
    # budget B; nothing enforces the step, so the write records the realised one (changed). An empty entry with one gradient takes it unchanged, no call
    def __init__(self, model: ModelSpec) -> None:
        self.model = model

    async def shorten(self, old: str, keep: int, calls: Calls, role: str, about: dict[str, Any] | None = None) -> Shortened:
        # not used: merge rewrites the entry in one call
        return Shortened(old, False)

    async def merge(
        self, key: str, old: str, parts: tuple[str, ...], alpha: float, budget: int | None, calls: Calls, role: str, steps: tuple[int, ...], writes_before: int
    ) -> MemoryWrite:
        lesson = "\n".join(parts)
        if not old and len(parts) == 1:
            new, call = parts[0], None
        else:
            prompt = PROMPT.format(
                lessons_word="lessons" if len(parts) > 1 else "lesson",
                percent=round(100 * alpha),
                limit=f", in at most {budget} characters" if budget is not None else "",
                support=SUPPORT.format(count=len(parts)) if len(parts) > 1 else "",
                notes=old or "(none yet)",
                lessons="\n".join(f"{number}. {part}" for number, part in enumerate(parts, start=1)),
            )
            answer = await calls.text(role, self.model.config, SYSTEM, prompt, {"key": key, "steps": list(steps)})
            new, call = answer.output.strip(), answer.index
        over = budget is not None and len(new) > budget
        new = new[:budget] if over else new
        return MemoryWrite(key, old, lesson, alpha, "", new, over, steps, writes_before, budget, None, call, False, over, parts, changed(old, new))

    def describe(self) -> str:
        return f"rewrite: {self.model.to_string()} rewrites the entry with the lessons, told to change about alpha of it (the realised step is recorded)"

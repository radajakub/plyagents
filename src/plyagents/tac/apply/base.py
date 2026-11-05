from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from plyagents.tac.model.feedback import StepFeedback
from plyagents.tac.model.memory import KeyedMemory, MemoryUpdate, MemoryWrite
from plyagents.tac.model.step import TACStep
from plyagents.tac.schedulers.base import Scheduler
from plyagents.tac.utils.calls import Calls

# between the kept notes and a new lesson
SEPARATOR = "\n"


@dataclass(frozen=True)
class Shortened:
    # the old entry after shortening, and whether it had to be cut because the shortening overshot its share
    text: str
    cut: bool


class Apply(ABC):
    # the Apply operator: writes a game's lessons into a memory. The old entry is shortened to (1 - alpha) B, the lesson
    # appended, and the result kept within the budget B (characters); alpha is the scheduler's step for the key's k_s. Without a
    # budget the lesson is only appended. Subclasses decide how to shorten
    @abstractmethod
    async def shorten(self, old: str, keep: int, calls: Calls, role: str) -> Shortened:
        raise NotImplementedError

    # one line for the debug summary printed when the player is built
    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError

    async def update(
        self, memory: KeyedMemory, steps: list[TACStep], feedback: list[StepFeedback], texts: list[str | None], scheduler: Scheduler, budget: int | None, calls: Calls, role: str
    ) -> MemoryUpdate:
        # what one game teaches a memory: the text of each step (the actor's lesson, the critic's note; None writes nothing) written by
        # write, with the visits and the feedback; the actor's memory and the critic's are written the same way
        writes = await self.write(memory, [(step.key, text) for step, text in zip(steps, texts) if text], scheduler, budget, calls, role)
        return MemoryUpdate(tuple((step.key, step.action) for step in steps), tuple(feedback), writes)

    async def write(self, memory: KeyedMemory, lessons: list[tuple[str, str]], scheduler: Scheduler, budget: int | None, calls: Calls, role: str) -> tuple[MemoryWrite, ...]:
        # lessons as (key, lesson) in step order; the lessons of a key visited several times are joined into one write
        joined: dict[str, list[str]] = {}
        for key, lesson in lessons:
            joined.setdefault(key, []).append(lesson)
        writes = []
        for key, parts in joined.items():
            entry, lesson = memory.get(key), SEPARATOR.join(parts)
            alpha = scheduler.step_size(entry.writes)
            writes.append(await self._merge(key, entry.text, lesson, alpha, budget, calls, role))
        return tuple(writes)

    async def _merge(self, key: str, old: str, lesson: str, alpha: float, budget: int | None, calls: Calls, role: str) -> MemoryWrite:
        if budget is None:
            shortened = Shortened(old, False)
        else:
            # an empty entry needs no shortening, so no call
            shortened = await self.shorten(old, int((1 - alpha) * budget), calls, role) if old else Shortened("", False)
        new = SEPARATOR.join(part for part in (shortened.text, lesson) if part)
        over = budget is not None and len(new) > budget
        # the lesson is cut, not the kept notes: it was told its share (alpha B)
        return MemoryWrite(key, old, lesson, alpha, shortened.text, new[:budget] if over else new, shortened.cut or over)

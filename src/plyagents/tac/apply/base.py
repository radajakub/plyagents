from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from plyagents.tac.model.feedback import StepFeedback
from plyagents.tac.model.memory import KeyedMemory, MemoryUpdate, MemoryWrite
from plyagents.tac.model.step import TACStep
from plyagents.tac.schedulers.base import Scheduler
from plyagents.tac.utils.calls import Calls

# between the kept notes and a new lesson
SEPARATOR = "\n"


@dataclass(frozen=True)
class Shortened:
    # the old entry after shortening, whether it had to be cut because the shortening overshot its share, and the index of the call that
    # shortened it among the proposal's calls (None when no call was made)
    text: str
    cut: bool
    call: int | None = None


class Apply(ABC):
    # the Apply operator: writes a game's lessons into a memory. The old entry is shortened to (1 - alpha) B, the lesson
    # appended, and the result kept within the budget B (characters); alpha is the scheduler's step for the key's k_s. Without a
    # budget the lesson is only appended. Subclasses decide how to shorten
    # about: what a shortening call names it is for (LLMCall.about)
    @abstractmethod
    async def shorten(self, old: str, keep: int, calls: Calls, role: str, about: dict[str, Any] | None = None) -> Shortened:
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
        written = [(number, step.key, text) for number, (step, text) in enumerate(zip(steps, texts), start=1) if text]
        writes = await self.write(memory, [(key, text) for _, key, text in written], scheduler, budget, calls, role, [number for number, _, _ in written])
        return MemoryUpdate(tuple((step.key, step.action) for step in steps), tuple(feedback), writes)

    async def write(
        self, memory: KeyedMemory, lessons: list[tuple[str, str]], scheduler: Scheduler, budget: int | None, calls: Calls, role: str, steps: list[int] | None = None
    ) -> tuple[MemoryWrite, ...]:
        # lessons as (key, lesson) in step order, steps their step numbers when known; the lessons of a key visited several times are joined into one write
        joined: dict[str, list[tuple[int | None, str]]] = {}
        for index, (key, lesson) in enumerate(lessons):
            joined.setdefault(key, []).append((steps[index] if steps is not None else None, lesson))
        writes = []
        for key, parts in joined.items():
            entry, lesson = memory.get(key), SEPARATOR.join(text for _, text in parts)
            alpha = scheduler.step_size(entry.writes)
            numbers = tuple(number for number, _ in parts if number is not None)
            writes.append(await self._merge(key, entry.text, lesson, alpha, budget, calls, role, numbers, entry.writes))
        return tuple(writes)

    async def _merge(self, key: str, old: str, lesson: str, alpha: float, budget: int | None, calls: Calls, role: str, steps: tuple[int, ...], writes_before: int) -> MemoryWrite:
        keep = int((1 - alpha) * budget) if budget is not None else None
        if keep is None:
            shortened = Shortened(old, False)
        else:
            # an empty entry needs no shortening, so no call
            shortened = await self.shorten(old, keep, calls, role, {"key": key, "steps": list(steps)}) if old else Shortened("", False)
        new = SEPARATOR.join(part for part in (shortened.text, lesson) if part)
        over = budget is not None and len(new) > budget
        # the lesson is cut, not the kept notes: it was told its share (alpha B)
        return MemoryWrite(
            key, old, lesson, alpha, shortened.text, new[:budget] if over else new, shortened.cut or over, steps, writes_before, budget, keep, shortened.call, shortened.cut, over
        )

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import Any

from plyagents.textrl.model.feedback import StepFeedback
from plyagents.textrl.model.memory import Entry, KeyedMemory, MemoryUpdate, MemoryWrite
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.schedulers.base import Scheduler
from plyagents.textrl.utils.calls import Calls

# between the kept notes and a new lesson
SEPARATOR = "\n"


def lesson_share(entry: Entry, scheduler: Scheduler, budget: int | None) -> int | None:
    # the length one new lesson may take in the entry: what a write of it leaves beside the kept (1 - alpha) B; None without a budget
    return budget - int((1 - scheduler.share(entry.writes, entry.samples, 1)) * budget) if budget is not None else None


def changed(old: str, new: str) -> float:
    # the realised step of a write: 1 minus the similarity of the new entry to the old (difflib's ratio), 0 for an unchanged entry
    return 1.0 - SequenceMatcher(None, old, new, autojunk=False).ratio()


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
        self, memory: KeyedMemory, steps: list[TextRLStep], feedback: list[StepFeedback], texts: list[str | None], scheduler: Scheduler, budget: int | None, calls: Calls, role: str
    ) -> MemoryUpdate:
        # what one game teaches a memory: the text of each step (the actor's lesson, the critic's note; None writes nothing) written by
        # write, with the visits and the feedback; the actor's memory and the critic's are written the same way
        written = [(number, step.key, text) for number, (step, text) in enumerate(zip(steps, texts), start=1) if text]
        writes = await self.write(memory, [(key, text) for _, key, text in written], scheduler, budget, calls, role, [number for number, _, _ in written])
        return MemoryUpdate(tuple((step.key, step.action) for step in steps), tuple(feedback), writes)

    async def write(
        self, memory: KeyedMemory, lessons: list[tuple[str, str]], scheduler: Scheduler, budget: int | None, calls: Calls, role: str, steps: list[int] | None = None
    ) -> tuple[MemoryWrite, ...]:
        # lessons as (key, lesson) in step order, steps their step numbers when known. All lessons of a key (several visits, several games of a batch)
        # go into one write, with one step: the scheduler's share for n gradients
        grouped: dict[str, list[tuple[int | None, str]]] = {}
        for index, (key, lesson) in enumerate(lessons):
            grouped.setdefault(key, []).append((steps[index] if steps is not None else None, lesson))
        writes = []
        for key, items in grouped.items():
            entry, parts = memory.get(key), tuple(text for _, text in items)
            share = scheduler.share(entry.writes, entry.samples, len(parts))
            numbers = tuple(number for number, _ in items if number is not None)
            writes.append(await self.merge(key, entry.text, parts, share, budget, calls, role, numbers, entry.writes))
        return tuple(writes)

    async def merge(
        self, key: str, old: str, parts: tuple[str, ...], alpha: float, budget: int | None, calls: Calls, role: str, steps: tuple[int, ...], writes_before: int
    ) -> MemoryWrite:
        # one write: the old entry shortened to (1 - alpha) B, then the lesson; the lesson is the gradients joined (their sum), or what reduce makes
        # of several. A subclass may write differently (rewrite)
        keep = int((1 - alpha) * budget) if budget is not None else None
        about = {"key": key, "steps": list(steps)}
        if keep is None:
            shortened = Shortened(old, False)
        else:
            # an empty entry needs no shortening, so no call
            shortened = await self.shorten(old, keep, calls, role, about) if old else Shortened("", False)
        share = budget - keep if budget is not None and keep is not None else None
        lesson, reduce_call = (parts[0], None) if len(parts) == 1 else await self.reduce(parts, share, calls, role, about)
        return self.assemble(key, old, parts, alpha, budget, keep, shortened, lesson, steps, writes_before, reduce_call)

    @staticmethod
    def assemble(
        key: str,
        old: str,
        parts: tuple[str, ...],
        alpha: float,
        budget: int | None,
        keep: int | None,
        shortened: Shortened,
        lesson: str,
        steps: tuple[int, ...],
        writes_before: int,
        reduce_call: int | None = None,
    ) -> MemoryWrite:
        # the kept notes, then the lesson; the lesson is cut, not the kept notes: it was told its share (alpha B)
        new = SEPARATOR.join(part for part in (shortened.text, lesson) if part)
        over = budget is not None and len(new) > budget
        new = new[:budget] if over else new
        written = MemoryWrite(
            key, old, lesson, alpha, shortened.text, new, shortened.cut or over, steps, writes_before, budget, keep, shortened.call, shortened.cut, over, parts, changed(old, new)
        )
        return replace(written, reduce_call=reduce_call)

    async def reduce(self, parts: tuple[str, ...], share: int | None, calls: Calls, role: str, about: dict[str, Any]) -> tuple[str, int | None]:
        # the lesson of several gradients of one key, with the index of the call that made it (None: no call): joined in step order (their sum), as
        # the TAC wrote them before batches; a subclass may reduce them to their mean within the share
        return SEPARATOR.join(parts), None

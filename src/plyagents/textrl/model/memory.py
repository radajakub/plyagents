from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from plyagents.textrl.model.feedback import StepFeedback
from plyagents.textrl.model.update import ComponentUpdate


@dataclass
class Entry:
    # the memory of one key: its text, the visits n_s, the actions taken there n_{s,a}, the writes k_s, and the gradients written (the samples the
    # step-size rule per=sample counts; several per write when a batch reduces them)
    text: str = ""
    visits: int = 0
    actions: dict[str, int] = field(default_factory=dict)
    writes: int = 0
    samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "actions": dict(self.actions)}


class KeyedMemory:
    # the textual memory M (or M^V): one entry per key z, read by equality
    def __init__(self, entries: dict[str, Entry] | None = None) -> None:
        self.entries = entries if entries is not None else {}

    def get(self, key: str) -> Entry:
        return self.entries.get(key, Entry())

    def read(self, key: str, memory: str) -> dict[str, Any]:
        # a read of the entry of a key as a step or an assessment records it: which memory, the key, whether the entry existed, and the entry as it
        # was (text, visits, actions, writes); score is a retriever's similarity, None for the read by equality
        entry = self.get(key)
        return {"memory": memory, "key": key, "score": None, "found": key in self.entries, **entry.to_dict()}

    def to_dict(self) -> dict[str, Any]:
        return {key: entry.to_dict() for key, entry in self.entries.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KeyedMemory:
        return cls({key: Entry(entry["text"], entry["visits"], dict(entry["actions"]), entry["writes"], entry["samples"]) for key, entry in data.items()})


@dataclass(frozen=True)
class MemoryWrite:
    # one write of Apply: the entry before, the game's lesson for the key, the step size alpha, the old entry after shortening,
    # the new entry, and whether it had to be cut to the budget
    key: str
    old: str
    lesson: str
    alpha: float
    compressed: str
    new: str
    cut: bool
    # the steps (1-based) whose lessons were joined into this one, the writes the entry had before (k_s, which gave alpha), the budget B and the
    # share the old entry was shortened to ((1 - alpha) B, None without a budget or for a rewrite), the index of the call among the proposal's calls
    # (None without one), and which part was cut: the old entry (the shortening overshot its share) or the lesson (over its share)
    steps: tuple[int, ...] = ()
    writes_before: int | None = None
    budget: int | None = None
    keep: int | None = None
    call: int | None = None
    cut_old: bool | None = None
    cut_lesson: bool | None = None
    # the gradients this write reduced into the lesson, in step order (one per step with text), and how much of the entry changed: 1 minus the
    # similarity of the new entry to the old (difflib), the realised step
    parts: tuple[str, ...] = ()
    changed: float | None = None
    # the index of the call that reduced several gradients into the lesson (compress[reduce=mean] when the entry needed no shortening); None when
    # the lesson came from no call of its own (one gradient, joined, or made in the same call as the shortening: call)
    reduce_call: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryWrite:
        return cls(**{**data, "steps": tuple(data["steps"]), "parts": tuple(data["parts"])})


@dataclass(frozen=True)
class MemoryUpdate(ComponentUpdate):
    # what one game teaches a memory: the visits (key and action of every step, None for a failed move), the feedback it was
    # given per step, and the writes. More than commit needs: update.json is then the full trace of the game's learning. Per step as well, what the
    # block's backward pass wrote (the gradient, None: nothing) and the length it was given for it (its share alpha B, None without a budget)
    visits: tuple[tuple[str, str | None], ...]
    feedback: tuple[StepFeedback, ...]
    writes: tuple[MemoryWrite, ...]
    gradients: tuple[str | None, ...] = ()
    limits: tuple[int | None, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "visits": [list(visit) for visit in self.visits],
            "feedback": [step.to_dict() for step in self.feedback],
            "writes": [asdict(write) for write in self.writes],
            "gradients": list(self.gradients),
            "limits": list(self.limits),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryUpdate:
        return cls(
            tuple((key, action) for key, action in data["visits"]),
            tuple(StepFeedback.from_dict(step) for step in data["feedback"]),
            tuple(MemoryWrite.from_dict(write) for write in data["writes"]),
            tuple(data.get("gradients", ())),
            tuple(data.get("limits", ())),
        )

    def apply_to(self, memory: KeyedMemory) -> None:
        # commit: count the visits and actions, replace the written entries and count the writes; deterministic
        for key, action in self.visits:
            entry = memory.entries.setdefault(key, Entry())
            entry.visits += 1
            if action is not None:
                entry.actions[action] = entry.actions.get(action, 0) + 1
        for write in self.writes:
            entry = memory.entries.setdefault(write.key, Entry())
            entry.text = write.new
            entry.writes += 1
            entry.samples += len(write.parts)

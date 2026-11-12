from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.counts import KeyCounts, MoveCounts, counts_from_dict, counts_to_dict, result
from plyagents.textrl.model.feedback import Grade
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.utils.solved import Solved


class Reference:
    # the numbers kept beside the text components, never shown to any of them except as the feedback's history: per key and
    # move, the counts of every training visit so far (every visit of a game counts)
    def __init__(self, counts: dict[str, dict[str, MoveCounts]] | None = None) -> None:
        self.counts: dict[str, dict[str, MoveCounts]] = counts if counts is not None else {}

    def before(self, key: str) -> KeyCounts:
        # a copy: the counts are immutable, so the snapshot stays as it was after later commits
        return dict(self.counts.get(key, {}))

    def add(self, key: str, move: str, outcome: float) -> None:
        moves = self.counts.setdefault(key, {})
        moves[move] = moves.get(move, MoveCounts()).plus(outcome)

    def to_dict(self) -> dict[str, Any]:
        return {key: counts_to_dict(moves) for key, moves in self.counts.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Reference:
        return cls({key: counts_from_dict(moves) for key, moves in data.items()})


@dataclass(frozen=True)
class Values:
    # a value of the state and of each legal move (aligned with TextRLStep.legal, None where there is none)
    value: float | None
    action_values: tuple[float | None, ...] | None

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "action_values": list(self.action_values) if self.action_values is not None else None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Values:
        action_values = data["action_values"]
        return cls(data["value"], tuple(action_values) if action_values is not None else None)


def solved_values(solved: Solved, step: TextRLStep) -> Values | None:
    # V* and Q* of the step's true state; None where the solver has no entry for it
    avq = solved.get(step.player, step.os_state)
    if avq is None:
        return None
    return Values(avq.V(), tuple(avq.Q(number) for number in step.legal_numbers))


@dataclass(frozen=True)
class ReferenceStep:
    # one step of a training game with the numbers beside its text: the counts of every move at its key before the game (what
    # the components knew when they answered), the critic's numbers, V* and Q* when the game is solved, and the feedback's grade.
    # step is 1-based, as the feedback model's verdicts and the game's steps are numbered
    step: int
    key: str
    move: str | None
    legal: tuple[str, ...]
    counts: KeyCounts
    critic: Values
    solved: Values | None
    grade: Grade | None

    @classmethod
    def of(cls, number: int, step: TextRLStep, counts: KeyCounts, assessment: Assessment, solved: Values | None, grade: Grade | None) -> ReferenceStep:
        return cls(number, step.key, step.action, step.legal, counts, Values(assessment.value, assessment.action_values), solved, grade)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "key": self.key,
            "move": self.move,
            "legal": list(self.legal),
            "counts": counts_to_dict(self.counts),
            "critic": self.critic.to_dict(),
            "solved": self.solved.to_dict() if self.solved is not None else None,
            "grade": self.grade,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReferenceStep:
        solved = data["solved"]
        return cls(
            data["step"],
            data["key"],
            data["move"],
            tuple(data["legal"]),
            counts_from_dict(data["counts"]),
            Values.from_dict(data["critic"]),
            Values.from_dict(solved) if solved is not None else None,
            data["grade"],
        )


@dataclass(frozen=True)
class ReferenceUpdate:
    # one training game's record: its steps and the shaped outcome. Not a component's update: the player keeps it for every config
    steps: tuple[ReferenceStep, ...]
    outcome: float

    def apply_to(self, reference: Reference) -> None:
        # commit: each played step counts once toward its key and move
        for step in self.steps:
            if step.move is not None:
                reference.add(step.key, step.move, self.outcome)

    def to_dict(self) -> dict[str, Any]:
        return {"outcome": self.outcome, "result": result(self.outcome), "steps": [step.to_dict() for step in self.steps]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReferenceUpdate:
        return cls(tuple(ReferenceStep.from_dict(step) for step in data["steps"]), data["outcome"])

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

# the grades of a move, worst first
type Grade = Literal["mistake", "dubious", "neutral", "good", "strong"]
GRADES: tuple[Grade, ...] = ("mistake", "dubious", "neutral", "good", "strong")
# the learning blocks of the TextRL: each takes its own text of a verdict as its gradient
type BlockKind = Literal["actor", "critic"]


@dataclass(frozen=True)
class StepFeedback:
    # the feedback model's verdict on one step of the agent: a grade, why, a lesson for the actor's memory and a note for the
    # critic's; all empty when the feedback gives none
    grade: Grade | None = None
    reason: str = ""
    lesson: str | None = None
    critic_note: str | None = None

    def gradient(self, block: BlockKind) -> str | None:
        # the text the loss wrote for a block: the actor's lesson, the critic's note; None is a zero gradient
        return self.lesson if block == "actor" else self.critic_note

    def without_gradient(self, block: BlockKind) -> StepFeedback:
        return replace(self, lesson=None) if block == "actor" else replace(self, critic_note=None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepFeedback:
        return cls(data.get("grade"), data.get("reason", ""), data.get("lesson"), data.get("critic_note"))

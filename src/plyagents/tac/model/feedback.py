from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StepFeedback:
    # the feedback model's verdict on one step of the agent: a grade, why, a lesson for the actor's memory and a note for the
    # critic's; all empty when the feedback gives none
    grade: str | None = None
    reason: str = ""
    lesson: str | None = None
    critic_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepFeedback:
        return cls(data.get("grade"), data.get("reason", ""), data.get("lesson"), data.get("critic_note"))

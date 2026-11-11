from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Assessment:
    # the critic's view of one visited state: a number when it has one, and the text the feedback model reads
    value: float | None
    text: str
    # Q(s, a) of the step's legal actions, aligned with them; None for an action it has no estimate of, or for a critic without action values
    action_values: tuple[float | None, ...] | None = None
    # an LLM critic's: the entry of its memory it read, and its reasoning for the assessment
    context: str | None = None
    reasoning: str | None = None

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    # the text assesses each legal move (the LLM Q critic), not the situation: the feedback model labels it so, and its critic notes are about the move
    moves: bool = False
    # an LLM critic's: every entry of its memory it read (KeyedMemory.read), and, for the Q critic, the moves the model named that are not legal and
    # the legal moves it named more than once (only the first answer counts)
    read: tuple[dict[str, Any], ...] = ()
    unmatched: tuple[str, ...] = ()
    repeated: tuple[str, ...] = ()
    # the LLM Q critic's: the assessment of each legal move, aligned with them (None for a move it left out); the critic loss reads the move played's
    move_texts: tuple[str | None, ...] | None = None

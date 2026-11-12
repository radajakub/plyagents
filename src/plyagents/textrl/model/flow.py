from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.feedback import BlockKind, StepFeedback
from plyagents.textrl.model.step import TextRLStep

# What flows between the components when the TextRL learns from a game, in the order it flows (player.py draws the whole graph): the actor's forward
# pass leaves a TextRLStep per move (model/step.py), the critic's an Assessment per step (model/assessment.py); a loss gives a verdict per step (Given),
# a sweep the verdicts of all losses (Judged); a block's backward pass reads its Tape of each step and turns the verdict into the text written to
# its memory (a gradient), which Apply writes (model/memory.py); a batch is learned from as JudgedGames.


@dataclass(frozen=True)
class Given:
    # what a loss gave a game: a verdict per step, and for the update's record the verdicts it dropped (an answer for a step that does not exist, or
    # for one already answered) and the inputs it was shown besides the moves, by part
    verdicts: list[StepFeedback]
    dropped: list[dict[str, Any]] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Judged:
    # what a sweep gives a game: the critic's assessment of every step, the feedback per step for both blocks (the joint loss's, or the actor's and the
    # critic's merged), and for the record what each of the actor's and the critic's own losses gave, by "actor" and "critic"
    assessments: list[Assessment]
    given: Given
    parts: dict[str, Given] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgedGame:
    # one game of a batch as the blocks learn from it: its steps, the critic's assessments, the shaped outcome and the verdict of every step
    steps: list[TextRLStep]
    assessments: list[Assessment]
    outcome: float
    feedback: list[StepFeedback]


@dataclass(frozen=True)
class Tape:
    # what a block stored of one step in its forward pass, read back from the records: the situation, the entry it read (None: nothing), what it
    # produced (the move, the assessment) and its reasoning; block names it for the wording of a backward pass that calls an LLM and for the text of
    # the verdict that is its gradient. limit: the length the written text may take, the share alpha B of the entry it goes to; None without a budget.
    # action: a critic's, the move the actor played (the actor's move is its output); per_move: the entry is the played move's, not the situation's (Q)
    block: BlockKind
    situation: str
    read: str | None
    output: str
    reasoning: str | None = None
    limit: int | None = None
    action: str | None = None
    per_move: bool = False

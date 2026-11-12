from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.counts import KeyCounts
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.targets.outcome import discounted

# a correction of the critic's assessment by the critic loss, shown in the backward sweep after the step was judged
CHECKED = "A check of that assessment found: {note}"
HISTORY = "What the agent's earlier games showed in these situations (the moves it played there and how those games ended):"
# written for an empty set of inputs
NO_INPUTS = "none"
# with `track` (the exploration experiment's T-text): the lessons keep what was tried and how sure they are, so the actor can judge what is still uncertain
TRACK = " In each lesson, also keep track of which moves were tried in this situation and how sure the lesson is."


def parse_inputs(text: str, known: tuple[str, ...], loss: str) -> str:
    # "q+memory" -> "memory+q": the inputs in the fixed order of known, so equal sets give equal configs and paths
    names = set() if text == NO_INPUTS else set(text.split("+"))
    unknown = sorted(names - set(known))
    if unknown:
        raise ValueError(f"Unknown {loss} inputs {unknown}; known: {list(known)} or {NO_INPUTS!r}, joined by +")
    return "+".join(name for name in known if name in names) or NO_INPUTS


class ShownInputs:
    # the params of a loss shown inputs (written joined by +, or NO_INPUTS): the set of their names
    inputs: str

    @property
    def shown(self) -> set[str]:
        return set() if self.inputs == NO_INPUTS else set(self.inputs.split("+"))


def history_text(counts: KeyCounts) -> str | None:
    # "<door:1> in 3 games (2 loss, 1 win); <door:2> in 1 games (1 win)", the moves in the order first played; None when the key was never seen
    if not counts:
        return None
    parts = (
        f"{move} in {move_counts.visits} games ({', '.join(f'{count} {name}' for name, count in Counter(move_counts.results).most_common())})"
        for move, move_counts in counts.items()
    )
    return "; ".join(parts)


def history_block(history: Mapping[str, KeyCounts], keys: list[str]) -> str:
    # per step of the game, the moves played in its situation in the earlier games; put after the moves
    lines = [f"Step {number}: {history_text(history.get(key, {})) or 'no earlier games in this situation'}." for number, key in enumerate(keys, start=1)]
    return f"\n\n{HISTORY}\n" + "\n".join(lines)


def outcome_line(outcome: float) -> str:
    return f"Outcome for the agent: {outcome:+g} (+1 is a win, 0 a draw, -1 a loss)."


def return_line(outcome: float, index: int, steps: int, gamma: float) -> str:
    # the outcome as a target of step index: with a discount and later moves, also the return it is discounted to
    later = steps - 1 - index
    if gamma == 1 or later == 0:
        return outcome_line(outcome)
    return f"{outcome_line(outcome)} Discounted by gamma={gamma:g} over the agent's {later} later moves, the return of this move: {discounted(outcome, index, steps, gamma):+.2f}."


def next_discount(gamma: float) -> str:
    # after the critic's assessment of the next situation, a TD target in text: how much it counts against this one; empty without a discount
    return "" if gamma == 1 else f"\nIt counts gamma={gamma:g} times as much as this situation (the discount per move of the agent)."


def discount_line(gamma: float) -> str:
    # for a prompt that shows the whole game and the outcome once: how the outcome counts for each move; empty without a discount
    if gamma == 1:
        return ""
    return f"\nThe outcome is discounted by gamma={gamma:g} per move of the agent: for a move k moves before the agent's last, it counts gamma^k times."


def critic_values(assessments: list[Assessment], loss: str) -> list[float]:
    values = [assessment.value for assessment in assessments]
    if any(value is None for value in values):
        raise ValueError(f"{loss} inputs=delta needs a critic that gives numbers (oracle, tabular, or llm_v[value=true])")
    return [value for value in values if value is not None]


def deltas(assessments: list[Assessment], outcome: float, loss: str, gamma: float = 1.0) -> list[tuple[float, float, bool]]:
    # V(s_t) against gamma V(s_t+1), the next position the agent moves in; after its last move, the outcome (the one reward)
    values = critic_values(assessments, loss)
    return [(value, gamma * values[index + 1] if index + 1 < len(values) else outcome, index + 1 == len(values)) for index, value in enumerate(values)]


def delta_line(delta: tuple[float, float, bool], gamma: float = 1.0) -> str:
    before, after, last = delta
    source = " (the outcome)" if last else f" (gamma={gamma:g} times the next value)" if gamma != 1 else ""
    return f"Critic: value before the move {before:+.2f}, after {after:+.2f}{source}; change {after - before:+.2f}."


def action_values_line(step: TextRLStep, assessment: Assessment, loss: str) -> str:
    if assessment.action_values is None:
        raise ValueError(f"{loss} inputs=q needs a critic with action values (oracle, tabular or llm_q[value=true])")
    shown = [f"{action} {value:+.2f}" if value is not None else f"{action} unknown" for action, value in zip(step.legal, assessment.action_values)]
    return f"Critic's value of each move here: {', '.join(shown)}."


class _Numbered(Protocol):
    step: int

    def model_dump(self) -> dict[str, Any]: ...


def by_step[A: _Numbered](answers: Sequence[A], steps: int) -> tuple[dict[int, A], list[dict[str, Any]]]:
    # the answers by step number; an answer for a step the game does not have, or for a step already answered, is not used but kept for the record
    found: dict[int, A] = {}
    dropped: list[dict[str, Any]] = []
    for answer in answers:
        if not 1 <= answer.step <= steps or answer.step in found:
            dropped.append(answer.model_dump())
            continue
        found[answer.step] = answer
    return found, dropped

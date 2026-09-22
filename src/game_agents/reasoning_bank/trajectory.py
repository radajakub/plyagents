from __future__ import annotations

from dataclasses import dataclass

from plybench.common.enums import GameResults
from plybench.configs.player_config import PlayerConfig
from plybench.trackers.game_tracker import GameStep, GameTracker
from plybench.trackers.step_data import StepData

# The <observation>/<think>/<action> trajectory induce_memory.py builds from a rollout, read off a
# plybench transcript instead: `_think` mirrors its _extract_think_from_output, and the retrieval key
# of a state is the intent plus the state, as in the query it embeds.

RETRIEVAL_QUERY = "retrieval_query"
_STATE_SEPARATOR = "\n\nGame state:\n"


def state_query(objective: str, state: str) -> str:
    """The retrieval key of one decision: the static objective plus the position it was faced with."""
    return f"{objective}{_STATE_SEPARATOR}{state}"


def split_state_query(query: str) -> tuple[str, str]:
    objective, _, state = query.partition(_STATE_SEPARATOR)
    return objective, state


@dataclass(frozen=True)
class Trajectory:
    """One game as the agent lived it, in the shape the judge and the inducer read."""

    objective: str
    thinks: list[str]
    actions: list[str]
    observations: list[str]
    queries: list[str]
    final_state: str
    result: GameResults | None

    @staticmethod
    def _think(step: GameStep) -> str:
        # models that reason outside a thinking block leave the trace empty and the reasoning in the
        # completion, so "what it thought" needs both
        return StepData.from_dict(step.data).thinking

    @staticmethod
    def _recorded_query(step: GameStep) -> str:
        return str(StepData.from_dict(step.data).extra.get(RETRIEVAL_QUERY) or "")

    @classmethod
    def from_tracker(cls, tracker: GameTracker, player: PlayerConfig, objective: str = "") -> Trajectory:
        steps = tracker.steps_of(player)
        ending = tracker.ending
        recorded = [cls._recorded_query(step) for step in steps]
        # a replayed game is read back without a live prompt adapter, so the objective is recovered
        # from the keys the agent recorded rather than asked of the game again
        objective = objective or next((split_state_query(query)[0] for query in recorded if query), "")
        return cls(
            objective,
            [cls._think(step) for step in steps],
            [step.move for step in steps],
            [step.observation for step in steps],
            [query or state_query(objective, step.observation) for query, step in zip(recorded, steps)],
            ending.observation if ending is not None else "",
            tracker.get_result(player) if ending is not None else None,
        )

    @property
    def outcome(self) -> str:
        return self.result.to_prompt() if self.result is not None else "UNKNOWN"

    def steps_prompt(self) -> str:
        blocks = [
            f"<observation>\n{observation}\n</observation>\n<think>\n{think}\n</think>\n<action>\n{action}\n</action>"
            for observation, think, action in zip(self.observations, self.thinks, self.actions)
        ]
        return "\n\n".join(blocks)

    def to_prompt(self) -> str:
        return f"**Query:** {self.objective}\n\n**Trajectory:**\n{self.steps_prompt()}"

    def action_history(self) -> str:
        lines = []
        for index, (think, action) in enumerate(zip(self.thinks, self.actions)):
            body = f"<think>{think}</think>\n   <action>{action}</action>" if think else f"<action>{action}</action>"
            lines.append(f"{index + 1}: {body}")
        return "\n".join(lines)

    def board_states(self, last: int = 5) -> str:
        recent = self.observations[-last:] if len(self.observations) >= last else self.observations
        states = [f"[Game state {index + 1}/{len(recent)}]\n{state}" for index, state in enumerate(recent)]
        return "\n\n---\n\n".join(states + [f"[Final state]\n{self.final_state}"])

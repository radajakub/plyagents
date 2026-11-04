from __future__ import annotations

from plybench.core.game import TurnBasedGame
from plybench.core.minimax import AVQCache, load_or_solve
from plybench.core.prompt_adapter import PromptAdapter

from plyagents.tac.feedback.base import Feedback
from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.feedback import Grade, StepFeedback
from plyagents.tac.model.step import TACStep
from plyagents.tac.utils.calls import Calls

_RESULT = {1.0: "wins", 0.0: "draws", -1.0: "loses"}


def _result(value: float) -> str:
    return _RESULT.get(value, f"is worth {value:+g}")


class OracleFeedback(Feedback):
    # F from the solved game, no call: a move is good when Q*(s, a) = V*(s), dubious when it gives up some value but does not
    # lose, a mistake when it loses; the lesson names the best moves. Privileged: it reads the true state
    def __init__(self) -> None:
        self._cache: AVQCache | None = None

    def prepare(self, game: TurnBasedGame) -> None:
        self._cache = load_or_solve(game)

    def initialize(self, game: TurnBasedGame, adapter_template: PromptAdapter) -> None:
        # again per game: a sampled variant is solved per instance (load_or_solve keeps each in memory)
        self._cache = load_or_solve(game)

    def _judge(self, step: TACStep) -> StepFeedback:
        assert self._cache is not None, "initialize must be called before the feedback judges"
        avq = self._cache[step.player, step.os_state]
        if avq is None or step.action is None:
            return StepFeedback()
        number = step.legal_numbers[step.legal.index(step.action)]
        value, action_value = avq.V(), avq.Q(number)
        best = [action for action, other in zip(step.legal, step.legal_numbers) if avq.check_optimal(other)]
        reason = f"Q* {action_value:+g} against V* {value:+g}"
        note = f"with best play this position {_result(value)}"
        if action_value == value:
            return StepFeedback("good", reason, f"{step.action} is a best move here: with best play it {_result(value)}.", note)
        grade: Grade = "mistake" if action_value < 0 else "dubious"
        lesson = f"{step.action} was a {grade} here: with best play it {_result(action_value)}. Better: {', '.join(best)}."
        return StepFeedback(grade, reason, lesson, note)

    async def give(self, steps: list[TACStep], assessments: list[Assessment], outcome: float, calls: Calls) -> list[StepFeedback]:
        return [self._judge(step) for step in steps]

    def describe(self) -> str:
        return "oracle: grades each move by Q*(s, a) against V*(s) and names the best moves (privileged: reads the true state)"

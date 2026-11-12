from __future__ import annotations

from collections.abc import Mapping

import pyspiel as sp
from plybench.core.game import TurnBasedGame
from plybench.core.prompt_adapter import PromptAdapter

from plyagents.textrl.losses.base import ActorLoss, CriticLoss, JointLoss, Loss
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.counts import KeyCounts
from plyagents.textrl.model.feedback import Grade, StepFeedback
from plyagents.textrl.model.flow import Given
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.rewards.base import RewardShaper
from plyagents.textrl.rewards.result import ResultShaper
from plyagents.textrl.utils.calls import Calls
from plyagents.textrl.utils.solved import Solved


def _certain(game: TurnBasedGame) -> bool:
    # without chance nodes a solved value is an outcome that happens; with them it is an expectation
    return game.game.get_type().chance_mode == sp.GameType.ChanceMode.DETERMINISTIC


class OracleJudge:
    # the verdicts of the solved game, no call, shared by the oracle losses: a move is good when Q*(s, a) = V*(s), dubious when it gives up some
    # value but does not lose, a mistake when it loses; the lesson names the best moves, the critic note the value. Privileged: it reads the true state
    def __init__(self) -> None:
        self.solved = Solved()
        # the words for a value: the player's shaper from prepare; the result's scale until then
        self._shaper: RewardShaper = ResultShaper()
        self._certain = True

    def prepare(self, game: TurnBasedGame, shaper: RewardShaper) -> None:
        self.solved.load(game)
        self._shaper = shaper
        self._certain = _certain(game)

    def initialize(self, game: TurnBasedGame) -> None:
        self.solved.load(game)
        self._certain = _certain(game)

    def _result(self, value: float) -> str:
        return self._shaper.value_text(value, self._certain)

    def judge(self, step: TextRLStep, assessment: Assessment) -> StepFeedback:
        avq = self.solved.get(step.player, step.os_state)
        if avq is None or step.action is None:
            return StepFeedback()
        number = step.legal_numbers[step.legal.index(step.action)]
        value, action_value = avq.V(), avq.Q(number)
        best = [action for action, other in zip(step.legal, step.legal_numbers) if avq.check_optimal(other)]
        reason = f"Q* {action_value:+g} against V* {value:+g}"
        # the critic note states what the critic estimates: the position's value, or for a critic of the moves the value of the move played
        note = f"with best play this move {self._result(action_value)}" if assessment.moves else f"with best play this position {self._result(value)}"
        if action_value == value:
            return StepFeedback("good", reason, f"{step.action} is a best move here: with best play it {self._result(value)}.", note)
        grade: Grade = "mistake" if action_value < 0 else "dubious"
        lesson = f"{step.action} was a {grade} here: with best play it {self._result(action_value)}. Better: {', '.join(best)}."
        return StepFeedback(grade, reason, lesson, note)


class OracleLoss(Loss):
    # the oracle loss of every role: its part (Loss.part) of the verdict of the solved game (OracleJudge), per step, no call. Privileged: it reads
    # the true state
    def __init__(self) -> None:
        self.judge = OracleJudge()

    def prepare(self, game: TurnBasedGame, shaper: RewardShaper) -> None:
        self.judge.prepare(game, shaper)

    def initialize(self, game: TurnBasedGame, adapter_template: PromptAdapter) -> None:
        self.judge.initialize(game)

    async def give(self, steps: list[TextRLStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts], calls: Calls) -> Given:
        return Given([self.part(self.judge.judge(step, assessment)) for step, assessment in zip(steps, assessments)])

    async def judge_step(
        self, index: int, steps: list[TextRLStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts], corrections: list[str | None], calls: Calls
    ) -> StepFeedback:
        return self.part(self.judge.judge(steps[index], assessments[index]))


class OracleJointLoss(OracleLoss, JointLoss):
    # F from the solved game: the grade, the lesson and the critic note of every move
    def describe(self) -> str:
        return "grades each move by Q*(s, a) against V*(s) and names the best moves (privileged: reads the true state)"


class OracleActorLoss(OracleLoss, ActorLoss):
    # the actor's part: the grade, its reason and the lesson
    def describe(self) -> str:
        return "grades each move by Q*(s, a) against V*(s) and names the best moves (privileged: reads the true state)"


class OracleCriticLoss(OracleLoss, CriticLoss):
    # the critic's part: the value under best play, with its reason
    def describe(self) -> str:
        return "notes the value of each position (or move) under best play (privileged: reads the true state)"

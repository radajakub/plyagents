from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from plybench.common.enums import GameResults
from plybench.llm import LLM, LLMMessage, ModelConfig

from game_agents.reasoning_bank.trajectory import Trajectory

# Ported from WebArena/autoeval: prompts.build_text_eval_prompt (the branch evaluator.eval_text
# actually takes), the Thoughts:/Status: line parsing in evaluator.py, and the success test in
# evaluate_trajectory.process_sample. Every deviation is a re-domaining of the task
# (web navigation -> turn-based games); the evaluator framing, the outcome typology, the strictness
# rules, the two-line output format and the binary status are upstream's. induce_memory.py's
# --criteria switch is mirrored by JudgeCriteria: under "autoeval" the label and the rationale both
# come from this judge, under "gt" the label comes from the engine and no rationale is appended at
# all -- upstream has no mode that mixes the two.

JudgeCriteria = Literal["autoeval", "gt"]

_SYSTEM_MSG = """You are an expert in evaluating the performance of a turn-based game agent. The agent is designed to play a turn-based game against an opponent and win. Given the agent's intent, the agent's move history, the final state of the board, and the agent's final move, your goal is to decide whether the agent's execution is successful or not.

There are three types of outcomes:
1. Win: The agent completed the winning condition of the game before the opponent did, or the opponent could not continue legally. Carefully examine the final state of the board to confirm the completed alignment belongs to the agent.
2. Loss: The opponent completed the winning condition first, or the agent played a move the rules do not permit and forfeited. Carefully examine the move history and the final state of the board to determine which side the completed alignment belongs to.
3. Neither: The board reached a terminal state with no winning alignment for either side. This is not a success.

*Strictness rules*
Before calling an execution successful, verify all three:
- Completeness: the game actually reached a terminal state in the agent's favour, rather than merely looking promising where the record ends.
- Grounding: the winning alignment is traceable to specific occupied positions in the final board state; an alignment that was inferred or assumed without visible support counts as a failure.
- Right side: confirm the completed alignment belongs to the side the agent was playing and not to the opponent.
When uncertain on any of these, mark failure. A false success is more harmful than a false failure, because memory induction amplifies it into future behavior.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process>"
Status: "success" or "failure"
"""

_SUCCESS = (GameResults.WIN, GameResults.OPPONENT_FAIL)


def _extract_content(text: str, start_tag: str) -> str:
    # extract_content in WebArena/autoeval/prompts.py
    for line in text.split("\n"):
        if line.startswith(start_tag):
            return line[len(start_tag) :].strip()
    return ""


@dataclass(frozen=True)
class Verdict:
    success: bool
    thoughts: str
    result: GameResults | None
    predicted: bool


class OutcomeJudge:
    def __init__(self, llm: LLM, model: ModelConfig, criteria: JudgeCriteria = "autoeval", draw_is_success: bool = False) -> None:
        self._llm = llm
        self._model = model
        self._criteria = criteria
        self._draw_is_success = draw_is_success

    def _ground_truth(self, result: GameResults | None) -> bool:
        if result is None:
            # an unfinished game cannot be a success, matching evaluate_trajectory's rm=None path
            return False
        if result in _SUCCESS:
            return True
        return self._draw_is_success and result is GameResults.DRAW

    def _user_prompt(self, trajectory: Trajectory) -> str:
        final_move = trajectory.actions[-1] if trajectory.actions else "N/A"
        return (
            f"User Intent: {trajectory.objective}\n\n"
            f"Move History:\n{trajectory.action_history()}\n\n"
            f"The detailed final state of the board:\n\n```md\n{trajectory.board_states()}\n```\n\n"
            f"Agent's final move: {final_move}."
        )

    async def __call__(self, trajectory: Trajectory) -> Verdict:
        # pipeline_memory.py runs step 2 unconditionally, so the prediction is always recorded even
        # under --criteria gt, where it is kept only to measure the judge against the engine
        response = await self._llm.generate(self._model, LLMMessage.system(_SYSTEM_MSG), [LLMMessage.user(self._user_prompt(trajectory))])
        thoughts = _extract_content(response.output_text, "Thoughts:")
        status = _extract_content(response.output_text, "Status:").replace('"', "")
        predicted = status.lower() == "success"

        if self._criteria == "autoeval":
            return Verdict(predicted, thoughts, trajectory.result, predicted)
        return Verdict(self._ground_truth(trajectory.result), "", trajectory.result, predicted)

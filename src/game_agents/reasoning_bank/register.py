from __future__ import annotations

from typing import Any

from plybench.app import PlyBench
from plybench.configs.player_config import PlayerConfig
from plybench.core.game import TurnBasedGame
from plybench.llm import LLM
from plybench.player.player import Player, PlayerIdentifier, PlayerOutput
from plybench.player.spec import PlayerSpec
from plybench.registry import Registry
from plybench.trackers.player_tracker import PlayerTracker
from plybench.trackers.step_data import StepData

from game_agents.reasoning_bank.params import PLAYER_KEY, ReasoningBankParams
from game_agents.reasoning_bank.player import ReasoningBankPlayer


class ReasoningBankTracker(PlayerTracker):
    def record(self, player_output: PlayerOutput) -> dict[str, Any]:
        # the retrieval key and how many items it pulled: the one affordance the baseline lacks, so it
        # has to be visible in the transcript rather than assumed
        return StepData.from_output(player_output, **(player_output.data or {})).to_dict()


def register_reasoning_bank(registry: Registry, llm: LLM) -> None:
    def build(game: TurnBasedGame, player_config: PlayerConfig, identifier: PlayerIdentifier) -> Player:
        del game
        return ReasoningBankPlayer(player_config, llm, identifier)

    registry.register_player(PlayerSpec(PLAYER_KEY, ReasoningBankParams, build, tracker=ReasoningBankTracker()))


def register_agents(op: PlyBench) -> PlyBench:
    register_reasoning_bank(op.registry, op.llm)
    return op

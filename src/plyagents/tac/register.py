from __future__ import annotations

from plybench.player.spec import PlayerSpec
from plybench.registry import Registry
from plyllm import LLM

from plyagents.tac.build import build_tac_player
from plyagents.tac.configs.components import Services
from plyagents.tac.params import TACParams
from plyagents.tac.player import TACTracker


def register_players(registry: Registry, llm: LLM) -> None:
    # the plybench.players entry point (pyproject.toml): every PlyBench in plyagents' environment builds tac: players, whose LLM
    # components call through this LLM router
    services = Services(llm)
    registry.register_player(
        PlayerSpec("tac", TACParams, lambda game, player_config, identifier: build_tac_player(game, player_config, identifier, services), tracker=TACTracker())
    )

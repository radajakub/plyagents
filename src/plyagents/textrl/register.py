from __future__ import annotations

from plybench.player.spec import PlayerSpec
from plybench.registry import Registry
from plyllm import LLM

from plyagents.textrl.build import build_textrl_player
from plyagents.textrl.configs.components import Services
from plyagents.textrl.params import TextRLParams
from plyagents.textrl.player import TextRLTracker
from plyagents.textrl.trace import read_textrl


def register_players(registry: Registry, llm: LLM) -> None:
    # the plybench.players entry point (pyproject.toml): every PlyBench in plyagents' environment builds textrl: players, whose LLM
    # components call through this LLM router
    services = Services(llm, registry.solvable)
    registry.register_player(
        PlayerSpec(
            "textrl",
            TextRLParams,
            lambda game, player_config, identifier: build_textrl_player(game, player_config, identifier, services),
            tracker=TextRLTracker(),
            trace=read_textrl,
        )
    )

from __future__ import annotations

from plybench.configs.player_config import PlayerConfig
from plybench.player.player import PlayerIdentifier

from plyagents.tac.actors.build import ACTORS
from plyagents.tac.actors.tabular import TabularActorParams
from plyagents.tac.critics.build import CRITICS
from plyagents.tac.enums import CriticType, TargetType
from plyagents.tac.params import TACParams
from plyagents.tac.player import TACPlayer
from plyagents.tac.rewards.build import REWARDS
from plyagents.tac.selectors.build import SELECTORS


def build_tac_player(player_config: PlayerConfig, identifier: PlayerIdentifier) -> TACPlayer:
    # the TAC composed of the components its params name, each built once by its role's collection
    params = player_config.params

    if not isinstance(params, TACParams):
        raise TypeError(f"{player_config.to_string()} is not a TAC config")

    # a critic without values cannot give the actor its targets; caught here rather than after the first game
    actor = params.actor.params

    if isinstance(actor, TabularActorParams) and actor.target.kind is TargetType.CRITIC and params.critic.kind is CriticType.NONE:
        raise ValueError("target=critic needs a critic that gives values (critic=oracle or critic=tabular)")

    return TACPlayer(
        player_config,
        identifier,
        ACTORS.build(params.actor),
        CRITICS.build(params.critic),
        REWARDS.build(params.reward),
        SELECTORS.build(params.train_selector),
        SELECTORS.build(params.eval_selector),
        params.key,
    )

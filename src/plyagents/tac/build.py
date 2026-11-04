from __future__ import annotations

from dataclasses import replace

from plybench.common.enums import Mode
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

# the configs already described by this process (see build_tac_player)
_ANNOUNCED: set[str] = set()


def build_tac_player(player_config: PlayerConfig, identifier: PlayerIdentifier) -> TACPlayer:
    # the TAC composed of the components its params name, each built once by its role's collection
    params = player_config.params

    if not isinstance(params, TACParams):
        raise TypeError(f"{player_config.to_string()} is not a TAC config")

    # a critic without values cannot give the actor its targets; caught here rather than after the first game
    actor = params.actor.params

    if isinstance(actor, TabularActorParams) and actor.target.kind is TargetType.CRITIC and params.critic.kind is CriticType.NONE:
        raise ValueError("target=critic needs a critic that gives values (critic=oracle or critic=tabular)")

    player = TACPlayer(
        player_config,
        identifier,
        ACTORS.build(params.actor),
        CRITICS.build(params.critic),
        REWARDS.build(params.reward),
        SELECTORS.build(params.train_selector),
        SELECTORS.build(params.eval_selector),
        params.key,
    )

    # plybench builds a player for every evaluation game and every epoch, each epoch with its own checkpoint, so a config is
    # described once per process, whatever checkpoint it carries
    spec = f"{player_config.key}:{replace(params, checkpoint_path=None).to_string()}"
    if params.debug and spec not in _ANNOUNCED:
        _ANNOUNCED.add(spec)
        print(describe_tac_player(spec, player, params))

    return player


def describe_tac_player(spec: str, player: TACPlayer, params: TACParams) -> str:
    rows = [
        ("actor", params.actor.to_string(), player.actor.describe()),
        ("critic", params.critic.to_string(), player.critic.describe()),
        ("train selector", params.train_selector.to_string(), player.selectors[Mode.TRAIN].describe()),
        ("eval selector", params.eval_selector.to_string(), player.selectors[Mode.EVAL].describe()),
        ("reward", params.reward.to_string(), player.shaper.describe()),
        ("key", params.key.value, "the observation text in this formalism, for retrieval only"),
    ]
    width = max(len(role) for role, _, _ in rows)
    return "\n".join([f"TAC player {spec}", *(f"  {role:<{width}}  {config}: {summary}" for role, config, summary in rows)])

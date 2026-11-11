from __future__ import annotations

from dataclasses import replace

from plybench.common.enums import Mode
from plybench.configs.player_config import PlayerConfig
from plybench.core.game import TurnBasedGame
from plybench.player.player import PlayerIdentifier

from plyagents.tac.actors.build import ACTORS
from plyagents.tac.actors.tabular import TabularActorParams
from plyagents.tac.configs.components import Services
from plyagents.tac.critics.build import CRITICS
from plyagents.tac.critics.llm import LLMCriticParams
from plyagents.tac.enums import CriticType, ReferenceType, TargetType
from plyagents.tac.feedback.build import FEEDBACKS
from plyagents.tac.feedback.llm import LLMFeedbackParams
from plyagents.tac.params import TACParams
from plyagents.tac.player import TACPlayer
from plyagents.tac.rewards.build import REWARDS
from plyagents.tac.selectors.build import SELECTORS

# the configs already described by this process (see build_tac_player)
_ANNOUNCED: set[str] = set()


def build_tac_player(game: TurnBasedGame, player_config: PlayerConfig, identifier: PlayerIdentifier, services: Services) -> TACPlayer:
    # the TAC composed of the components its params name, each built once by its role's collection
    params = player_config.params

    if not isinstance(params, TACParams):
        raise TypeError(f"{player_config.to_string()} is not a TAC config")

    # a critic without values cannot give the actor its targets; caught here rather than after the first game. The LLM critics'
    # optional numbers are for diagnostics, not targets
    actor = params.actor.params

    if isinstance(actor, TabularActorParams) and actor.target.kind is TargetType.CRITIC and params.critic.kind in (CriticType.NONE, CriticType.LLM_V, CriticType.LLM_Q):
        raise ValueError("target=critic needs a critic that gives values (critic=oracle or critic=tabular)")

    # the feedback's inputs that need numbers from the critic, checked here for the same reason
    feedback, critic = params.feedback.params, params.critic.params
    if isinstance(feedback, LLMFeedbackParams):
        # the Q critic gives no value of the situation, so no delta
        numbers = params.critic.kind in (CriticType.ORACLE, CriticType.TABULAR) or (params.critic.kind is CriticType.LLM_V and isinstance(critic, LLMCriticParams) and critic.value)
        if "delta" in feedback.shown and not numbers:
            raise ValueError("feedback inputs=delta needs a critic that gives numbers (critic=oracle, critic=tabular or critic=llm_v[value=true])")
        if "q" in feedback.shown and params.critic.kind not in (CriticType.ORACLE, CriticType.TABULAR):
            raise ValueError("feedback inputs=q needs a critic with action values (critic=oracle or critic=tabular)")

    player = TACPlayer(
        player_config,
        identifier,
        services,
        ACTORS.build(params.actor, services),
        CRITICS.build(params.critic, services),
        FEEDBACKS.build(params.feedback, services),
        REWARDS.build(params.reward, services),
        SELECTORS.build(params.train_selector, services),
        SELECTORS.build(params.eval_selector, services),
        params.key,
        params.reference is ReferenceType.SOLVED and services.solvable(game.game_type),
    )
    player.prepare(game)

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
        ("feedback", params.feedback.to_string(), player.feedback.describe()),
        ("train selector", params.train_selector.to_string(), player.selectors[Mode.TRAIN].describe()),
        ("eval selector", params.eval_selector.to_string(), player.selectors[Mode.EVAL].describe()),
        ("reward", params.reward.to_string(), player.shaper.describe()),
        ("key", params.key.value, "the observation text in this formalism, for retrieval only"),
        ("reference", params.reference.value, "counts per key and move" + (", V* and Q*" if player.solved else "") + ", recorded per game, shown to no component"),
    ]
    width = max(len(role) for role, _, _ in rows)
    return "\n".join([f"TAC player {spec}", *(f"  {role:<{width}}  {config}: {summary}" for role, config, summary in rows)])

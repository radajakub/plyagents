from __future__ import annotations

from dataclasses import replace

from plybench.configs.player_config import PlayerConfig
from plybench.core.game import TurnBasedGame
from plybench.player.player import PlayerIdentifier

from plyagents.textrl.actors.build import ACTORS
from plyagents.textrl.actors.llm import LLMActorParams
from plyagents.textrl.apply.compress import CompressParams
from plyagents.textrl.configs.components import ComponentParams, Need, Services
from plyagents.textrl.configs.presets import preset_defines
from plyagents.textrl.critics.build import CRITICS
from plyagents.textrl.critics.llm import LLMCriticParams
from plyagents.textrl.enums import ActorLossType, ApplyType, BatchType, Capability, CriticLossType, JointLossType, ReduceType, ReferenceType
from plyagents.textrl.losses.base import Losses
from plyagents.textrl.losses.build import ACTOR_LOSSES, CRITIC_LOSSES, JOINT_LOSSES
from plyagents.textrl.params import ROLES, TextRLParams
from plyagents.textrl.player import TextRLPlayer
from plyagents.textrl.rewards.build import REWARDS
from plyagents.textrl.selectors.build import SELECTORS
from plyagents.textrl.sweeps.build import SWEEPS
from plyagents.textrl.utils.options import split_options

# the configs already described by this process (see build_textrl_player)
_ANNOUNCED: set[str] = set()

# how an error names what a role's component gives (ComponentSpec.gives) when another component needs it
_GIVES = {
    ("critic", Capability.ASSESSMENT): "that assesses the steps (any critic but critic=none)",
    ("critic", Capability.VALUE): "that gives numbers (critic=oracle, critic=tabular or critic=llm_v[value=true])",
    ("critic", Capability.ACTION_VALUES): "with action values (critic=oracle, critic=tabular or critic=llm_q[value=true])",
    ("critic", Capability.TARGET): "that gives values to learn toward (critic=oracle or critic=tabular; an LLM critic's numbers are for diagnostics)",
    ("sweep", Capability.WHOLE_GAME): "that shows the losses the whole game at once (sweep=episode)",
}


def check_textrl(params: TextRLParams) -> None:
    # what the components need from each other (ComponentSpec.needs against ComponentSpec.gives), caught here rather than after the first game; the
    # joint loss or the actor's and the critic's own, not both, since both would write the same memories; with batches, a write that reduces a key's
    # gradients rather than cutting all but the first to the budget; and from a preset, the roles that define its algorithm as it defines them. Checked
    # here, not when parsing, so a stored spec still reads back when a preset's definition changes later
    if params.preset is not None:
        for name, defined in preset_defines(params.preset).items():
            if (config := _departs(params, name, defined)) is not None:
                raise ValueError(f"{name}={config} is not the preset {params.preset} (it defines {name}={defined}); write the spec in full without preset=")
    own = params.actor_loss.kind is not ActorLossType.NONE or params.critic_loss.kind is not CriticLossType.NONE
    if own and params.feedback.kind is not JointLossType.NONE:
        raise ValueError("feedback (the joint loss) or actor_loss and critic_loss, not both")
    gives = {"critic": CRITICS.gives(params.critic), "sweep": SWEEPS.gives(params.sweep)}
    for role in ROLES:
        for need in role.collection.needs(params.config(role)):
            if need.capability not in gives[need.provider]:
                raise ValueError(_missing(role.name, need))
    if params.batch is not BatchType.GAME:
        for role, config in (("actor", params.actor), ("critic", params.critic)):
            if _joins_within_budget(config.params):
                raise ValueError(
                    f"batch={params.batch.value} writes all of a key's gradients of the batch at once, so the {role} with a budget needs an Apply that reduces"
                    " them to their mean (apply=compress[reduce=mean] or apply=rewrite), or no budget: joined, all but the first are cut to the budget"
                )


def _departs(params: TextRLParams, name: str, defined: str) -> str | None:
    # the role's config when it departs from what a preset defines (another type, or another value of a param the definition names), else None
    role = {role.name: role for role in ROLES}[name]
    config, definition = params.config(role), role.collection.config(defined)
    named = split_options(defined.partition("[")[2][:-1], f"the preset definition {name}={defined}")
    if config.kind != definition.kind or any(getattr(config.params, param) != getattr(definition.params, param) for param in named):
        return config.to_string()
    return None


def _joins_within_budget(params: ComponentParams) -> bool:
    # an LLM block that writes its memory within a budget with an Apply that joins a key's gradients (their sum): truncate, compress with reduce=join
    if not isinstance(params, (LLMActorParams, LLMCriticParams)) or params.budget is None or (isinstance(params, LLMActorParams) and not params.memory):
        return False
    apply = params.apply.params
    return params.apply.kind is ApplyType.TRUNCATE or (isinstance(apply, CompressParams) and apply.reduce is ReduceType.JOIN)


def _missing(role: str, need: Need) -> str:
    option = f" {need.option}" if need.option else ""
    return f"{role}{option} needs a {need.provider} {_GIVES[need.provider, need.capability]}"


def build_textrl_player(game: TurnBasedGame, player_config: PlayerConfig, identifier: PlayerIdentifier, services: Services) -> TextRLPlayer:
    # the TextRL composed of the components its params name, each built once by its role's collection
    params = player_config.params
    if not isinstance(params, TextRLParams):
        raise TypeError(f"{player_config.to_string()} is not a TextRL config")
    check_textrl(params)

    # the TextRL's discount reaches the components that build targets from the outcome or the critic's next value
    services = replace(services, gamma=params.gamma)
    player = TextRLPlayer(
        player_config,
        identifier,
        services,
        ACTORS.build(params.actor, services),
        CRITICS.build(params.critic, services),
        Losses(JOINT_LOSSES.build(params.feedback, services), ACTOR_LOSSES.build(params.actor_loss, services), CRITIC_LOSSES.build(params.critic_loss, services)),
        SWEEPS.build(params.sweep, services),
        REWARDS.build(params.reward, services),
        SELECTORS.build(params.train_selector, services),
        SELECTORS.build(params.eval_selector, services),
        params.key,
        params.reference is ReferenceType.SOLVED and services.solvable(game.game_type),
        params.batch,
    )
    player.prepare(game)

    # plybench builds a player for every evaluation game and every epoch, each epoch with its own checkpoint, so a config is
    # described once per process, whatever checkpoint it carries
    spec = f"{player_config.key}:{replace(params, checkpoint_path=None).to_string()}"
    if params.debug and spec not in _ANNOUNCED:
        _ANNOUNCED.add(spec)
        print(describe_textrl_player(spec, player, params))

    return player


def describe_textrl_player(spec: str, player: TextRLPlayer, params: TextRLParams) -> str:
    components = player.components()
    rows = [(role.label, params.config(role).to_string(), components[role.name].describe()) for role in ROLES]
    rows += [
        ("key", params.key.value, "the observation text in this formalism, for retrieval only"),
        (
            "batch",
            params.batch.value,
            {
                BatchType.GAME: "one update per game",
                BatchType.EPOCH: "one update of the memories per epoch, from all its training games; the counts per game",
                BatchType.EVERY: "one update of the memories per batch of update_every games; the counts per game",
            }[params.batch],
        ),
        ("gamma", str(params.gamma), "the discount per move of the agent in the losses' and the tabular targets; the oracle's values are not discounted"),
        ("reference", params.reference.value, "counts per key and move" + (", V* and Q*" if player.solved is not None else "") + ", recorded per game, shown to no component"),
    ]
    width = max(len(role) for role, _, _ in rows)
    return "\n".join([f"TextRL player {spec}", *(f"  {role:<{width}}  {config}: {summary}" for role, config, summary in rows)])

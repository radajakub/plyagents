from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from plybench.common.enums import ObservationType
from plybench.configs.player_params import CheckpointedParams
from plybench.utils.text import to_bool
from plyllm import ModelConfig

from plyagents.textrl.actors.build import ACTORS
from plyagents.textrl.actors.llm import LLMActorParams
from plyagents.textrl.configs.components import Collection, ComponentConfig, with_model
from plyagents.textrl.configs.model import ModelSpec
from plyagents.textrl.configs.presets import preset_spec
from plyagents.textrl.critics.build import CRITICS
from plyagents.textrl.critics.llm import LLMCriticParams
from plyagents.textrl.enums import ActorLossType, ActorType, BatchType, CriticLossType, CriticType, JointLossType, ReferenceType, RewardType, SelectorType, SweepType
from plyagents.textrl.losses.build import ACTOR_LOSSES, CRITIC_LOSSES, JOINT_LOSSES
from plyagents.textrl.losses.llm_actor import LLMActorLossParams
from plyagents.textrl.losses.llm_critic import LLMCriticLossParams
from plyagents.textrl.losses.llm_joint import LLMJointLossParams
from plyagents.textrl.rewards.build import REWARDS
from plyagents.textrl.selectors.build import SELECTORS
from plyagents.textrl.sweeps.build import SWEEPS
from plyagents.textrl.utils.options import read_options

# the longest path suffix kept whole, leaving room in plybench's 255-byte directory name for "textrl_" and "_vs_<trainer>"
MAX_SUFFIX = 160
HASH_LENGTH = 12


@dataclass(frozen=True)
class Role:
    # one role of the TextRL: its option in the spec (and its field of TextRLParams), the collection of its component types, its name in the debug summary
    name: str
    collection: Collection[Any, Any]
    label: str


# the roles, in the order of the spec, the path and the debug summary
ROLES = (
    Role("actor", ACTORS, "actor"),
    Role("critic", CRITICS, "critic"),
    Role("feedback", JOINT_LOSSES, "feedback"),
    Role("actor_loss", ACTOR_LOSSES, "actor loss"),
    Role("critic_loss", CRITIC_LOSSES, "critic loss"),
    Role("sweep", SWEEPS, "sweep"),
    Role("train_selector", SELECTORS, "train selector"),
    Role("eval_selector", SELECTORS, "eval selector"),
    Role("reward", REWARDS, "reward"),
)
# the options a spec names; model= and preset= are read when parsing and resolved into the roles
_OPTIONS = {role.name for role in ROLES} | {"key", "reference", "batch", "gamma", "checkpoint", "debug", "model", "preset"}


def _text(value: ComponentConfig | ObservationType | ReferenceType | BatchType | float) -> str:
    # an option's value in the spec: a component's config, an enum's value, a number as Python writes it, so it reads back exactly
    if isinstance(value, ComponentConfig):
        return value.to_string()
    return str(value) if isinstance(value, (int, float)) else value.value


@dataclass(frozen=True, eq=True)
class TextRLParams(CheckpointedParams):
    # the TextRL composed of one config per role; actor has no default, so every spec and path names it
    actor: ComponentConfig[ActorType]
    critic: ComponentConfig[CriticType] = CRITICS.default(CriticType.NONE)
    # the joint loss (the feedback model F)
    feedback: ComponentConfig[JointLossType] = JOINT_LOSSES.default(JointLossType.NONE)
    # the actor's and the critic's own losses, instead of the joint one
    actor_loss: ComponentConfig[ActorLossType] = ACTOR_LOSSES.default(ActorLossType.NONE)
    critic_loss: ComponentConfig[CriticLossType] = CRITIC_LOSSES.default(CriticLossType.NONE)
    sweep: ComponentConfig[SweepType] = SWEEPS.default(SweepType.EPISODE)
    train_selector: ComponentConfig[SelectorType] = SELECTORS.default(SelectorType.GREEDY)
    eval_selector: ComponentConfig[SelectorType] = SELECTORS.default(SelectorType.GREEDY)
    reward: ComponentConfig[RewardType] = REWARDS.default(RewardType.RESULT)
    # the observation type the key is computed in; only for retrieval, independent of what an LLM is shown
    key: ObservationType = ObservationType.ACTIONS
    # the numbers recorded beside the text of every training game (never shown to a component): counts always, V* and Q* too
    # when solved and the game is solvable
    reference: ReferenceType = ReferenceType.SOLVED
    # the games one update of the memories learns from: each game, or every training game of the epoch
    batch: BatchType = BatchType.GAME
    # the discount per move of the agent, in (0, 1]: the return of a move k moves before the agent's last is gamma^k G, and a TD target is gamma
    # times the next value; the oracle components keep the solved, undiscounted values
    gamma: float = 1.0
    # the named config the spec started from (textrl/presets/<name>.json), part of the identity: a preset and the same spec in full are different
    # players with different paths
    preset: str | None = None
    checkpoint_path: Path | None = None
    # prints the components when the player is built; not part of the player's identity, so it stays out of the string, the
    # path and equality, and a run with it lands on the same records as one without
    debug: bool = field(default=False, compare=False)

    @classmethod
    def from_string(cls, params_string: str) -> TextRLParams:
        # format: actor=tabular[target=critic],critic=tabular,train_selector=eps_greedy[eps=0.2],...; an LLM component names its own model, or takes
        # the TextRL's model= when it names none. preset=<name> starts from a named config; the other options replace its roles
        options = read_options(params_string, _OPTIONS, "TextRL params")
        preset = options.pop("preset", None)
        if preset is not None:
            options = {**read_options(preset_spec(preset), _OPTIONS - {"preset"}, f"TextRL preset {preset!r}"), **options}
        model = ModelSpec.from_string(options.pop("model")) if "model" in options else None
        if "actor" not in options:
            raise ValueError(f"TextRL params need an actor: {params_string!r}")
        key = ObservationType.from_value(options.get("key", ObservationType.ACTIONS.value))
        if key is None:
            raise ValueError(f"Invalid TextRL key {options['key']!r}; choose from {ObservationType.values()}")
        reference = ReferenceType.from_value(options.get("reference", ReferenceType.SOLVED.value))
        if reference is None:
            raise ValueError(f"Invalid TextRL reference {options['reference']!r}; choose from {ReferenceType.values()}")
        checkpoint = options.get("checkpoint")
        batch = BatchType.from_value(options.get("batch", BatchType.GAME.value))
        if batch is None:
            raise ValueError(f"Invalid TextRL batch {options['batch']!r}; choose from {BatchType.values()}")
        gamma = float(options.get("gamma", 1.0))
        if not 0 < gamma <= 1:
            raise ValueError(f"Invalid TextRL gamma {options['gamma']!r}; give a discount in (0, 1]")
        configs = {role.name: role.collection.config(options[role.name]) for role in ROLES if role.name in options}
        checkpoint_path = Path(checkpoint) if checkpoint else None
        debug = to_bool(options.get("debug", False))
        params = replace(cls(configs.pop("actor")), **configs, key=key, reference=reference, batch=batch, gamma=gamma, preset=preset, checkpoint_path=checkpoint_path, debug=debug)
        return params if model is None else params.with_model(model)

    def with_model(self, model: ModelSpec) -> TextRLParams:
        # the TextRL's model= given to every LLM component that names none: resolved here, so the spec equals the same spec written in full
        return replace(self, **{role.name: with_model(self.config(role), model) for role in ROLES})

    def config(self, role: Role) -> ComponentConfig[Any]:
        return getattr(self, role.name)

    def _options(self) -> list[tuple[str, ComponentConfig | ObservationType | ReferenceType | BatchType | float]]:
        # the actor and every role that differs from its default, so a new option leaves existing specs and paths unchanged. From a preset every role,
        # so the string pins the whole config whatever the preset's file says later
        defaults = TextRLParams(self.actor)
        roles: list[tuple[str, ComponentConfig | ObservationType | ReferenceType | BatchType | float]] = [(role.name, self.config(role)) for role in ROLES]
        roles += [("key", self.key), ("reference", self.reference), ("batch", self.batch), ("gamma", self.gamma)]
        return [(name, value) for name, value in roles if name == "actor" or self.preset is not None or value != getattr(defaults, name)]

    def to_string(self) -> str:
        options = [f"{name}={_text(value)}" for name, value in self._options()]
        if self.preset is not None:
            options.insert(0, f"preset={self.preset}")
        if self.checkpoint_path is not None:
            options.append(f"checkpoint={self.checkpoint_path}")
        return ",".join(options)

    def display_name(self) -> str | None:
        # for plots: TextRL and its actor, critic and feedback by type, e.g. "TextRL: LLM actor, oracle critic, oracle feedback (gemini-3.5-flash-lite)"; the
        # model once when every LLM component calls the same one, else after each. Selectors, reward, key and the components' params are left out
        # the actor's and the critic's own losses only when used, so the names of the runs before them do not change
        roles: list[tuple[str, ComponentConfig[Any]]] = [("actor", self.actor), ("critic", self.critic), ("feedback", self.feedback)]
        roles += [(role, config) for role, config in (("actor loss", self.actor_loss), ("critic loss", self.critic_loss)) if config.kind.value != "none"]
        llm_params = (LLMActorParams, LLMCriticParams, LLMJointLossParams, LLMActorLossParams, LLMCriticLossParams)
        models = {role: config.params.model if isinstance(config.params, llm_params) else None for role, config in roles}
        names = {model.config.model_name for model in models.values() if model is not None}
        parts = []
        for role, config in roles:
            kind = {"none": "no", "llm": "LLM", "llm_v": "LLM V", "llm_q": "LLM Q"}.get(str(config.kind.value), str(config.kind.value))
            model = models[role]
            parts.append(f"{kind} {role}" + (f" ({model.config.model_name})" if len(names) > 1 and model is not None else ""))
        return f"TextRL{f' {self.preset}' if self.preset is not None else ''}: {', '.join(parts)}" + (f" ({names.pop()})" if len(names) == 1 else "")

    def models(self) -> list[ModelConfig]:
        # every model the TextRL calls, once each in the order of the roles, for plybench's --check and total limits
        return list(dict.fromkeys(model for role in ROLES for model in self.config(role).params.models()))

    @property
    def path_suffix(self) -> str:
        # no "=", "," or brackets: the checkpoint path is built from this suffix and read back by from_string. plybench names a
        # directory <player>_vs_<trainer>, at most 255 bytes, so a longer suffix keeps its readable start and a hash of the whole
        # (run.json and metadata.json hold the full spec)
        suffix = "_".join(f"{name}-{value.path_suffix if isinstance(value, ComponentConfig) else _text(value)}" for name, value in self._options())
        # from a preset: its name and a hash of the whole config, which changes when the preset's file does
        if self.preset is not None:
            return f"preset-{self.preset}-{hashlib.sha256(suffix.encode()).hexdigest()[:HASH_LENGTH]}"
        if len(suffix) <= MAX_SUFFIX:
            return suffix
        return f"{suffix[: MAX_SUFFIX - HASH_LENGTH - 1]}-{hashlib.sha256(suffix.encode()).hexdigest()[:HASH_LENGTH]}"

    @property
    def checkpoint(self) -> Path | None:
        return self.checkpoint_path

    def with_checkpoint(self, checkpoint: Path) -> TextRLParams:
        return replace(self, checkpoint_path=checkpoint)

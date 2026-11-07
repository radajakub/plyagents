from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path

from plybench.common.enums import ObservationType
from plybench.configs.player_params import CheckpointedParams
from plybench.utils.text import to_bool
from plyllm import ModelConfig

from plyagents.tac.actors.build import ACTORS
from plyagents.tac.actors.llm import LLMActorParams
from plyagents.tac.configs.components import ComponentConfig
from plyagents.tac.critics.build import CRITICS
from plyagents.tac.critics.llm import LLMCriticParams
from plyagents.tac.enums import ActorType, CriticType, FeedbackType, RewardType, SelectorType
from plyagents.tac.feedback.build import FEEDBACKS
from plyagents.tac.feedback.llm import LLMFeedbackParams
from plyagents.tac.rewards.build import REWARDS
from plyagents.tac.selectors.build import SELECTORS
from plyagents.tac.utils.options import read_options

# the longest path suffix kept whole, leaving room in plybench's 255-byte directory name for "tac_" and "_vs_<trainer>"
MAX_SUFFIX = 160
HASH_LENGTH = 12

_OPTIONS = {"actor", "critic", "feedback", "train_selector", "eval_selector", "reward", "key", "checkpoint", "debug"}


@dataclass(frozen=True, eq=True)
class TACParams(CheckpointedParams):
    # the TAC composed of one config per role; actor has no default, so every spec and path names it
    actor: ComponentConfig[ActorType]
    critic: ComponentConfig[CriticType] = CRITICS.default(CriticType.NONE)
    feedback: ComponentConfig[FeedbackType] = FEEDBACKS.default(FeedbackType.NONE)
    train_selector: ComponentConfig[SelectorType] = SELECTORS.default(SelectorType.GREEDY)
    eval_selector: ComponentConfig[SelectorType] = SELECTORS.default(SelectorType.GREEDY)
    reward: ComponentConfig[RewardType] = REWARDS.default(RewardType.RESULT)
    # the observation type the key is computed in; only for retrieval, independent of what an LLM is shown
    key: ObservationType = ObservationType.ACTIONS
    checkpoint_path: Path | None = None
    # prints the components when the player is built; not part of the player's identity, so it stays out of the string, the
    # path and equality, and a run with it lands on the same records as one without
    debug: bool = field(default=False, compare=False)

    @classmethod
    def from_string(cls, params_string: str) -> TACParams:
        # format: actor=tabular[target=critic],critic=tabular,train_selector=eps_greedy[eps=0.2],...; an LLM component names its own model
        options = read_options(params_string, _OPTIONS, "TAC params")
        if "actor" not in options:
            raise ValueError(f"TAC params need an actor: {params_string!r}")
        key = ObservationType.from_value(options.get("key", ObservationType.ACTIONS.value))
        if key is None:
            raise ValueError(f"Invalid TAC key {options['key']!r}; choose from {ObservationType.values()}")
        checkpoint = options.get("checkpoint")
        return cls(
            ACTORS.config(options["actor"]),
            CRITICS.config(options["critic"]) if "critic" in options else cls.critic,
            FEEDBACKS.config(options["feedback"]) if "feedback" in options else cls.feedback,
            SELECTORS.config(options["train_selector"]) if "train_selector" in options else cls.train_selector,
            SELECTORS.config(options["eval_selector"]) if "eval_selector" in options else cls.eval_selector,
            REWARDS.config(options["reward"]) if "reward" in options else cls.reward,
            key,
            Path(checkpoint) if checkpoint else None,
            to_bool(options.get("debug", False)),
        )

    def _options(self) -> list[tuple[str, ComponentConfig | ObservationType]]:
        # the actor and every role that differs from its default, so a new option leaves existing specs and paths unchanged
        defaults = TACParams(self.actor)
        roles: list[tuple[str, ComponentConfig | ObservationType]] = [
            ("actor", self.actor),
            ("critic", self.critic),
            ("feedback", self.feedback),
            ("train_selector", self.train_selector),
            ("eval_selector", self.eval_selector),
            ("reward", self.reward),
            ("key", self.key),
        ]
        return [(name, value) for name, value in roles if name == "actor" or value != getattr(defaults, name)]

    def to_string(self) -> str:
        options = [f"{name}={value.to_string() if isinstance(value, ComponentConfig) else value.value}" for name, value in self._options()]
        if self.checkpoint_path is not None:
            options.append(f"checkpoint={self.checkpoint_path}")
        return ",".join(options)

    def display_name(self) -> str | None:
        # for plots: TAC and its actor, critic and feedback by type, e.g. "TAC: LLM actor, oracle critic, oracle feedback (gemini-3.5-flash-lite)"; the
        # model once when every LLM component calls the same one, else after each. Selectors, reward, key and the components' params are left out
        roles = [("actor", self.actor), ("critic", self.critic), ("feedback", self.feedback)]
        models = {role: config.params.model if isinstance(config.params, (LLMActorParams, LLMCriticParams, LLMFeedbackParams)) else None for role, config in roles}
        names = {model.config.model_name for model in models.values() if model is not None}
        parts = []
        for role, config in roles:
            kind = {"none": "no", "llm": "LLM"}.get(str(config.kind.value), str(config.kind.value))
            model = models[role]
            parts.append(f"{kind} {role}" + (f" ({model.config.model_name})" if len(names) > 1 and model is not None else ""))
        return f"TAC: {', '.join(parts)}" + (f" ({names.pop()})" if len(names) == 1 else "")

    def models(self) -> list[ModelConfig]:
        # every model the TAC calls, once each in the order of the roles, for plybench's --check and total limits
        roles = (self.actor, self.critic, self.feedback, self.train_selector, self.eval_selector, self.reward)
        return list(dict.fromkeys(model for role in roles for model in role.params.models()))

    @property
    def path_suffix(self) -> str:
        # no "=", "," or brackets: the checkpoint path is built from this suffix and read back by from_string. plybench names a
        # directory <player>_vs_<trainer>, at most 255 bytes, so a longer suffix keeps its readable start and a hash of the whole
        # (run.json and metadata.json hold the full spec)
        suffix = "_".join(f"{name}-{value.path_suffix if isinstance(value, ComponentConfig) else value.value}" for name, value in self._options())
        if len(suffix) <= MAX_SUFFIX:
            return suffix
        return f"{suffix[: MAX_SUFFIX - HASH_LENGTH - 1]}-{hashlib.sha256(suffix.encode()).hexdigest()[:HASH_LENGTH]}"

    @property
    def checkpoint(self) -> Path | None:
        return self.checkpoint_path

    def with_checkpoint(self, checkpoint: Path) -> TACParams:
        return replace(self, checkpoint_path=checkpoint)

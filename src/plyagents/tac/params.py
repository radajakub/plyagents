from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from plybench.common.enums import ObservationType
from plybench.configs.player_params import CheckpointedParams

from plyagents.tac.actors.build import ACTORS
from plyagents.tac.configs.components import ComponentConfig
from plyagents.tac.critics.build import CRITICS
from plyagents.tac.enums import ActorType, CriticType, RewardType, SelectorType
from plyagents.tac.rewards.build import REWARDS
from plyagents.tac.selectors.build import SELECTORS
from plyagents.tac.utils.options import read_options

_OPTIONS = {"actor", "critic", "train_selector", "eval_selector", "reward", "key", "checkpoint"}


@dataclass(frozen=True, eq=True)
class TACParams(CheckpointedParams):
    # the TAC composed of one config per role; actor has no default, so every spec and path names it
    actor: ComponentConfig[ActorType]
    critic: ComponentConfig[CriticType] = CRITICS.default(CriticType.NONE)
    train_selector: ComponentConfig[SelectorType] = SELECTORS.default(SelectorType.GREEDY)
    eval_selector: ComponentConfig[SelectorType] = SELECTORS.default(SelectorType.GREEDY)
    reward: ComponentConfig[RewardType] = REWARDS.default(RewardType.RESULT)
    # the observation type the key is computed in; only for retrieval, independent of what an LLM is shown
    key: ObservationType = ObservationType.ACTIONS
    checkpoint_path: Path | None = None

    @classmethod
    def from_string(cls, params_string: str) -> TACParams:
        # format: actor=tabular[target=critic],critic=tabular,train_selector=eps_greedy[eps=0.2],... (later an LLM part before a "|")
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
            SELECTORS.config(options["train_selector"]) if "train_selector" in options else cls.train_selector,
            SELECTORS.config(options["eval_selector"]) if "eval_selector" in options else cls.eval_selector,
            REWARDS.config(options["reward"]) if "reward" in options else cls.reward,
            key,
            Path(checkpoint) if checkpoint else None,
        )

    def _options(self) -> list[tuple[str, ComponentConfig | ObservationType]]:
        # the actor and every role that differs from its default, so a new option leaves existing specs and paths unchanged
        defaults = TACParams(self.actor)
        roles: list[tuple[str, ComponentConfig | ObservationType]] = [
            ("actor", self.actor),
            ("critic", self.critic),
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

    @property
    def path_suffix(self) -> str:
        # no "=", "," or brackets: the checkpoint path is built from this suffix and read back by from_string
        return "_".join(f"{name}-{value.path_suffix if isinstance(value, ComponentConfig) else value.value}" for name, value in self._options())

    @property
    def checkpoint(self) -> Path | None:
        return self.checkpoint_path

    def with_checkpoint(self, checkpoint: Path) -> TACParams:
        return replace(self, checkpoint_path=checkpoint)

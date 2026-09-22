from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Self, get_args

from plybench.common.enums import ObservationType, OutputStrategies
from plybench.configs.player_params import CheckpointedParams
from plybench.llm import LLMCallOptions, ModelConfig, Provider, ReasoningEffort
from plybench.llm.model_config import EmbeddingModelConfig
from plybench.player.llm_player import options_to_string
from plybench.utils.text import extract_params, to_bool

from game_agents.reasoning_bank.judge import JudgeCriteria

PLAYER_KEY = "reasoning_bank"
CHECKPOINT_OPTION = "checkpoint"
_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh", "max")


def _parse_provider(name: str) -> Provider:
    provider = Provider.from_value(name)
    if provider is None:
        raise ValueError(f"Unknown provider {name!r}; known: {Provider.values()}")
    return provider


def _parse_effort(value: str) -> ReasoningEffort:
    if value not in _EFFORTS:
        raise ValueError(f"Invalid reasoning_effort {value!r}; allowed: {_EFFORTS}")
    return value  # type: ignore[return-value]


def _parse_call_options(options: dict[str, str]) -> LLMCallOptions:
    effort = options.get("reasoning_effort")
    return LLMCallOptions(
        reasoning_effort=None if effort is None else _parse_effort(effort),
        thinking_enabled=to_bool(options.get("thinking_enabled", False)),
        max_tokens=int(options["max_tokens"]) if "max_tokens" in options else None,
        temperature=float(options["temperature"]) if "temperature" in options else None,
    )


def _split_spec(spec: str, what: str) -> tuple[Provider, str]:
    # nested specs use "/" because ":" already separates the fields of a player config string
    provider, _, model_name = spec.partition("/")
    if not model_name:
        raise ValueError(f"Invalid {what} {spec!r} (need provider/model)")
    return _parse_provider(provider), model_name


def parse_model(spec: str, options: LLMCallOptions | None = None) -> ModelConfig:
    provider, model_name = _split_spec(spec, "model")
    return ModelConfig(provider, model_name, options if options is not None else LLMCallOptions())


def parse_embedding_model(spec: str) -> EmbeddingModelConfig:
    provider, model_name = _split_spec(spec, "embedding model")
    return EmbeddingModelConfig(provider, model_name)


def model_to_string(model: ModelConfig | EmbeddingModelConfig) -> str:
    return f"{model.provider.value}/{model.model_name}"


DEFAULT_EMBED_MODEL = parse_embedding_model("gemini/gemini-embedding-2")


def _parse_criteria(value: str) -> JudgeCriteria:
    if value not in get_args(JudgeCriteria):
        raise ValueError(f"Invalid criteria {value!r}; allowed: {get_args(JudgeCriteria)}")
    return value  # type: ignore[return-value]


def _parse_switch(value: str) -> bool:
    return True if value == "on" else False if value == "off" else to_bool(value)


def _switch(value: bool) -> str:
    return "on" if value else "off"


@dataclass(frozen=True, eq=True)
class ReasoningBankParams(CheckpointedParams):
    """`reasoning_bank:<observation>:<strategy>:<provider>:<model>:<options>`.

    `<options>` is one comma-separated `key=value` bag holding the model's call options, the knobs
    below and the checkpoint the training harness injects.
    """

    observation_type: ObservationType
    output_strategy: OutputStrategies
    model: ModelConfig
    # the field behind `CheckpointedParams.checkpoint`, which stays the read-only property it declares
    checkpoint_path: Path | None = None
    memory: bool = True
    top_n: int = 1
    criteria: JudgeCriteria = "autoeval"
    draw_is_success: bool = False
    asymmetric: bool = False
    embed: EmbeddingModelConfig = DEFAULT_EMBED_MODEL
    # None means "the model the agent plays with", which is what the paper's pipeline uses
    judge: ModelConfig | None = None
    induce: ModelConfig | None = None

    @classmethod
    def from_string(cls, params_string: str) -> Self:
        parts = params_string.split(":")
        if len(parts) < 4:
            raise ValueError(f"Invalid {PLAYER_KEY} params {params_string!r} (need observation:strategy:provider:model[:options])")
        observation_type = ObservationType.from_value(parts[0])
        output_strategy = OutputStrategies.from_value(parts[1])
        if observation_type is None or output_strategy is None:
            raise ValueError(f"Invalid observation/strategy in {params_string!r}")
        options = extract_params(":".join(parts[4:]))
        checkpoint = options.get(CHECKPOINT_OPTION)
        model = ModelConfig(_parse_provider(parts[2]), parts[3], _parse_call_options(options))
        return cls(
            observation_type,
            output_strategy,
            model,
            Path(checkpoint) if checkpoint else None,
            memory=_parse_switch(options.get("memory", "on")),
            top_n=int(options.get("top_n", 1)),
            criteria=_parse_criteria(options.get("criteria", "autoeval")),
            draw_is_success=to_bool(options.get("draw_is_success", False)),
            asymmetric=to_bool(options.get("asymmetric", False)),
            embed=parse_embedding_model(options["embed"]) if "embed" in options else DEFAULT_EMBED_MODEL,
            judge=parse_model(options["judge"], model.options) if "judge" in options else None,
            induce=parse_model(options["induce"], model.options) if "induce" in options else None,
        )

    @property
    def checkpoint(self) -> Path | None:
        return self.checkpoint_path

    def with_checkpoint(self, checkpoint: Path) -> Self:
        return replace(self, checkpoint_path=checkpoint)

    @property
    def judge_model(self) -> ModelConfig:
        return self.judge if self.judge is not None else self.model

    @property
    def induce_model(self) -> ModelConfig:
        return self.induce if self.induce is not None else self.model

    def _options(self) -> dict[str, str]:
        options = {
            "memory": _switch(self.memory),
            "top_n": str(self.top_n),
            "criteria": self.criteria,
            "draw_is_success": str(self.draw_is_success),
            "asymmetric": str(self.asymmetric),
            "embed": model_to_string(self.embed),
        }
        if self.judge is not None:
            options["judge"] = model_to_string(self.judge)
        if self.induce is not None:
            options["induce"] = model_to_string(self.induce)
        if self.checkpoint is not None:
            options[CHECKPOINT_OPTION] = str(self.checkpoint)
        return options

    def to_string(self) -> str:
        head = f"{self.observation_type.value}:{self.output_strategy.value}:{self.model.provider.value}:{self.model.model_name}"
        tail = ",".join(part for part in [options_to_string(self.model.options), *(f"{key}={value}" for key, value in self._options().items())] if part)
        return f"{head}:{tail}"

    @property
    def path_suffix(self) -> str:
        head = f"{self.observation_type.value}_{self.output_strategy.value}_{self.model.provider.value}_{self.model.model_name}"
        call_options = options_to_string(self.model.options).replace("=", "_").replace(",", "_")
        # a directory name carries only the knobs that distinguish arms, and a checkpoint path is
        # abbreviated: it is long and repeats the run directory it already sits in
        options = {"memory": _switch(self.memory), "top_n": str(self.top_n), "criteria": self.criteria}
        if self.checkpoint is not None:
            options[CHECKPOINT_OPTION] = hashlib.sha256(str(self.checkpoint).encode()).hexdigest()[:8]
        tail = "_".join(f"{key}_{value}" for key, value in options.items())
        return "_".join(part for part in (head, call_options, tail) if part)

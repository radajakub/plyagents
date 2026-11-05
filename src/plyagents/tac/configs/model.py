from __future__ import annotations

from dataclasses import dataclass

from plyllm import ModelConfig, options_to_string


@dataclass(frozen=True)
class ModelSpec:
    # the model an LLM component calls, written provider:model[options] inside a component config, e.g.
    # metacentrum:gpt-oss-120b[thinking_enabled=True,reasoning_effort=low]: plyllm's provider:model:options with the options
    # bracketed, since their commas would otherwise split the component's own options
    config: ModelConfig

    @classmethod
    def from_string(cls, text: str) -> ModelSpec:
        head, _, rest = text.partition("[")
        if rest and not rest.endswith("]"):
            raise ValueError(f"Invalid model {text!r}; expected provider:model or provider:model[options]")
        return cls(ModelConfig.from_string(f"{head}:{rest[:-1]}" if rest else head))

    def to_string(self) -> str:
        options = options_to_string(self.config.options)
        head = f"{self.config.provider.value}:{self.config.model_name}"
        return f"{head}[{options}]" if options else head

    @property
    def path_suffix(self) -> str:
        return self.config.slug

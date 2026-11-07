from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import Field, dataclass, fields
from typing import Any, ClassVar

from plybench.utils.enums import ExtendedEnum
from plyllm import LLM, ModelConfig

from plyagents.tac.configs.model import ModelSpec
from plyagents.tac.utils.options import read_options

# A component is configured like a plybench player: a type (an enum value) and its params, written "type[params]", e.g.
# eps_greedy[eps=0.2] or tabular[target=critic,scheduler=constant[beta=0.05]]; a component without params is its type alone.
# Each role keeps a Collection of specs (type -> params class and builder), as plybench's Registry keeps PlayerSpecs.


class ComponentParams(ABC):
    # the params of one component type, a frozen dataclass whose defaults are the type's defaults. Written as the fields that
    # differ from the defaults, so a new field leaves existing configs and paths unchanged
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    @classmethod
    @abstractmethod
    def from_string(cls, params_string: str) -> ComponentParams:
        raise NotImplementedError

    def _changed(self) -> list[tuple[str, Any]]:
        defaults = type(self)()
        return [(field.name, getattr(self, field.name)) for field in fields(self) if getattr(self, field.name) != getattr(defaults, field.name)]

    def to_string(self) -> str:
        return ",".join(f"{name}={value.to_string() if isinstance(value, (ComponentConfig, ModelSpec)) else value}" for name, value in self._changed())

    def models(self) -> list[ModelConfig]:
        # every model this component calls, its nested components' included (e.g. apply=compress[model=...] of an LLM actor)
        found: list[ModelConfig] = []
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, ModelSpec):
                found.append(value.config)
            elif isinstance(value, ComponentConfig):
                found.extend(value.params.models())
        return found

    @property
    def path_suffix(self) -> str:
        return "-".join(f"{name}-{value.path_suffix if isinstance(value, (ComponentConfig, ModelSpec)) else value}" for name, value in self._changed())


@dataclass(frozen=True, eq=True)
class NoParams(ComponentParams):
    @classmethod
    def from_string(cls, params_string: str) -> NoParams:
        read_options(params_string, set(), "a component without params")
        return cls()


@dataclass(frozen=True, eq=True)
class ComponentConfig[E: ExtendedEnum]:
    kind: E
    params: ComponentParams

    def to_string(self) -> str:
        params = self.params.to_string()
        return f"{self.kind.value}[{params}]" if params else str(self.kind.value)

    @property
    def path_suffix(self) -> str:
        params = self.params.path_suffix
        return f"{self.kind.value}-{params}" if params else str(self.kind.value)


@dataclass(frozen=True)
class ComponentSpec[T, P: ComponentParams]:
    params_cls: type[P]
    build: Callable[[P, Services], T]


@dataclass(frozen=True)
class Services:
    # what every builder hands a component besides its params: the LLM router (each LLM component names its own model)
    llm: LLM


def require_model(model: ModelSpec | None, component: str, services: Services) -> ModelSpec:
    # an LLM component's model, checked when the player is built rather than at its first call: it must be named, and its
    # options must suit it (plyllm's validation, e.g. a thinking-only model without thinking_enabled). A provider that is not
    # configured here is not checked; its first call says so
    if model is None:
        raise ValueError(f"{component} calls an LLM, so it needs a model: model=provider:model[options] in its brackets")
    config = model.config
    if config.provider in services.llm.available_providers:
        try:
            services.llm.resolve_model(config.provider, config.model_name).validate(config.options)
        except ValueError as error:
            raise ValueError(f"{component}: model {model.to_string()}: {error}") from error
    return model


class Collection[E: ExtendedEnum, T]:
    # the components of one role by type: parses their configs and builds them
    def __init__(self, role: str, kinds: type[E], specs: Mapping[E, ComponentSpec[T, Any]]) -> None:
        self.role = role
        self.kinds = kinds
        self.specs = specs

    def config(self, text: str) -> ComponentConfig[E]:
        name, _, rest = text.partition("[")
        if rest and not rest.endswith("]"):
            raise ValueError(f"Invalid {self.role} config {text!r}; expected type or type[params]")
        kind = self.kinds.from_value(name)
        if kind is None or kind not in self.specs:
            raise ValueError(f"Invalid {self.role} {name!r}; choose from {[kind.value for kind in self.specs]}")
        return ComponentConfig(kind, self.specs[kind].params_cls.from_string(rest[:-1]))

    def default(self, kind: E) -> ComponentConfig[E]:
        return ComponentConfig(kind, self.specs[kind].params_cls())

    def build(self, config: ComponentConfig[E], services: Services) -> T:
        return self.specs[config.kind].build(config.params, services)

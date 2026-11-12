"""The component configs, on a made-up role: a collection parses "type[params]", writes only what differs from the defaults,
gives a path without separators, and builds the component."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from plybench.utils.enums import ExtendedEnum
from plyllm import LLM, LLMConfig, MetacentrumProviderConfig, ModelConfig

from plyagents.textrl.configs.components import Collection, ComponentConfig, ComponentParams, ComponentSpec, NoParams, Services, require_model
from plyagents.textrl.configs.model import ModelSpec
from plyagents.textrl.utils.options import read_options


class _Kind(ExtendedEnum):
    PLAIN = "plain"
    TUNED = "tuned"


@dataclass(frozen=True, eq=True)
class _TunedParams(ComponentParams):
    rate: float = 0.5
    inner: ComponentConfig[_Kind] = ComponentConfig(_Kind.PLAIN, NoParams())

    @classmethod
    def from_string(cls, params_string: str) -> _TunedParams:
        options = read_options(params_string, {"rate", "inner"}, "tuned")
        return cls(float(options.get("rate", 0.5)), _COLLECTION.config(options["inner"]) if "inner" in options else cls.inner)


_COLLECTION: Collection[_Kind, str] = Collection(
    "thing",
    _Kind,
    {_Kind.PLAIN: ComponentSpec(NoParams, lambda params, services: "plain"), _Kind.TUNED: ComponentSpec(_TunedParams, lambda params, services: f"tuned {params.rate}")},
)


def test_a_config_round_trips_and_writes_only_what_differs_from_the_defaults():
    config = _COLLECTION.config("tuned[rate=0.2,inner=tuned[rate=0.1]]")
    assert config == ComponentConfig(_Kind.TUNED, _TunedParams(0.2, ComponentConfig(_Kind.TUNED, _TunedParams(0.1))))
    assert config.to_string() == "tuned[rate=0.2,inner=tuned[rate=0.1]]"
    assert config.path_suffix == "tuned-rate-0.2-inner-tuned-rate-0.1"
    assert _COLLECTION.config(config.to_string()) == config
    assert _COLLECTION.config("tuned") == _COLLECTION.default(_Kind.TUNED)
    assert _COLLECTION.default(_Kind.TUNED).to_string() == "tuned"


def test_a_collection_builds_the_component_of_a_config(services):
    assert _COLLECTION.build(_COLLECTION.config("plain"), services) == "plain"
    assert _COLLECTION.build(_COLLECTION.config("tuned[rate=0.3]"), services) == "tuned 0.3"


def test_a_collection_refuses_unknown_types_and_params():
    with pytest.raises(ValueError, match="Invalid thing"):
        _COLLECTION.config("other")
    with pytest.raises(ValueError, match="Unknown options"):
        _COLLECTION.config("plain[rate=0.2]")
    with pytest.raises(ValueError, match="type\\[params\\]"):
        _COLLECTION.config("tuned[rate=0.2")


def test_a_model_is_written_with_its_options_in_brackets():
    model = ModelSpec.from_string("metacentrum:gpt-oss-120b[thinking_enabled=True,reasoning_effort=low]")
    assert model.config == ModelConfig.from_string("metacentrum:gpt-oss-120b:thinking_enabled=True,reasoning_effort=low")
    assert model.to_string() == "metacentrum:gpt-oss-120b[thinking_enabled=True,reasoning_effort=low]"
    assert ModelSpec.from_string(model.to_string()) == model
    assert ModelSpec.from_string("metacentrum:gpt-oss-120b").to_string() == "metacentrum:gpt-oss-120b"
    # no separators of the config strings in the path
    assert not any(char in model.path_suffix for char in "=,[]:")
    with pytest.raises(ValueError):
        ModelSpec.from_string("nowhere:model")


def test_a_models_options_are_validated_when_built_against_a_configured_provider():
    # a configured Metacentrum client (nothing is called): gpt-oss-120b reasons always, so it needs thinking_enabled
    services = Services(LLM(LLMConfig(metacentrum=MetacentrumProviderConfig(api_key="unused", base_url="http://localhost"))), lambda key: False)
    with pytest.raises(ValueError, match="requires thinking to be enabled"):
        require_model(ModelSpec.from_string("metacentrum:gpt-oss-120b[reasoning_effort=low]"), "llm actor", services)
    good = ModelSpec.from_string("metacentrum:gpt-oss-120b[thinking_enabled=True,reasoning_effort=low]")
    assert require_model(good, "llm actor", services) == good


def test_a_missing_model_is_named_and_an_unconfigured_provider_is_not_checked(services):
    with pytest.raises(ValueError, match="needs a model"):
        require_model(None, "compress", services)
    # no provider configured: the first call will say so
    model = ModelSpec.from_string("metacentrum:gpt-oss-120b[reasoning_effort=low]")
    assert require_model(model, "compress", services) == model

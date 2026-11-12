"""Named configs: the TAC's model= goes to every LLM component that names none and leaves the spec equal to the same spec in full; a preset starts
from a named config, the other options replace its roles, its string pins every role, and its path is its name and a hash of the whole config."""

from __future__ import annotations

import pytest
from plybench.app import PlyBench
from plyllm import LLMConfig

from plyagents.tac import params as params_module
from plyagents.tac.actors.llm import LLMActorParams
from plyagents.tac.apply.compress import CompressParams
from plyagents.tac.configs.model import ModelSpec
from plyagents.tac.configs.presets import preset_names, preset_spec
from plyagents.tac.critics.llm import LLMCriticParams
from plyagents.tac.enums import CriticType, SweepType
from plyagents.tac.params import TACParams
from plyagents.tac.player import TACPlayer

M = "metacentrum:gpt-oss-120b"
OTHER = "metacentrum:qwen3-32b"


def test_the_tac_model_goes_to_every_llm_component_that_names_none_and_the_spec_equals_it_in_full():
    short = TACParams.from_string(f"model={M},actor=llm[budget=80,apply=compress],critic=llm_v[model={OTHER}],feedback=llm,train_selector=greedy")
    full = TACParams.from_string(f"actor=llm[model={M},budget=80,apply=compress[model={M}]],critic=llm_v[model={OTHER}],feedback=llm[model={M}]")
    assert short == full and short.to_string() == full.to_string() and short.path_suffix == full.path_suffix
    actor, critic = short.actor.params, short.critic.params
    # the nested Apply takes it too; a component that names its own model keeps it
    assert isinstance(actor, LLMActorParams) and isinstance(actor.apply.params, CompressParams) and actor.apply.params.model == ModelSpec.from_string(M)
    assert isinstance(critic, LLMCriticParams) and critic.model == ModelSpec.from_string(OTHER)
    assert TACParams.from_string(f"model={M},actor=tabular") == TACParams.from_string("actor=tabular")


@pytest.mark.parametrize("name", preset_names())
def test_every_preset_parses_builds_and_round_trips(name):
    params = TACParams.from_string(f"preset={name},model={M}")
    assert params.preset == name and TACParams.from_string(params.to_string()) == params
    assert params.path_suffix.startswith(f"preset-{name}-") and len(params.path_suffix) == len(f"preset-{name}-") + params_module.HASH_LENGTH
    # model= reaches every LLM component; the bandit baselines have none
    assert all(model.model_name == "gpt-oss-120b" for model in params.models())
    registry = PlyBench(LLMConfig()).registry
    game = registry.build_engine(registry.game_config("tic_tac_toe:")).game
    assert isinstance(registry.build_player(game, registry.player_config(f"tac:{params.to_string()}"), "i"), TACPlayer)


def test_the_presets_are_the_named_designs():
    assert set(preset_names()) == {"joint", "sac_td", "a2c_td", "mc", "hindsight", "greedy", "eps_greedy", "ucb"}
    assert all(TACParams.from_string(f"preset={name},model={M}").models() for name in ("joint", "sac_td", "a2c_td", "mc", "hindsight"))
    assert not any(TACParams.from_string(f"preset={name},model={M}").models() for name in ("greedy", "eps_greedy", "ucb"))
    sac = TACParams.from_string(f"preset=sac_td,model={M}")
    assert sac.critic.kind is CriticType.LLM_Q and sac.sweep.kind is SweepType.PARALLEL


def test_options_replace_the_roles_of_the_preset_and_a_preset_differs_from_its_spec_in_full():
    preset = TACParams.from_string(f"preset=sac_td,model={M}")
    overridden = TACParams.from_string(f"preset=sac_td,model={M},critic=llm_v[budget=600,apply=compress]")
    assert overridden.critic.kind is CriticType.LLM_V and overridden.actor == preset.actor and overridden.path_suffix != preset.path_suffix
    # the string names every role, so it reads back the same whatever the preset's file says later
    assert all(f"{role}=" in preset.to_string() for role in ("actor", "critic", "feedback", "actor_loss", "critic_loss", "sweep", "batch", "reference"))
    full = TACParams.from_string(f"model={M},{preset_spec('sac_td')}")
    assert full != preset and full.preset is None and not full.path_suffix.startswith("preset-")
    assert preset.display_name() is not None and (preset.display_name() or "").startswith("TAC sac_td: ")


def test_a_changed_preset_file_gives_a_new_path(monkeypatch):
    before = TACParams.from_string(f"preset=sac_td,model={M}")
    monkeypatch.setattr(params_module, "preset_spec", lambda name: preset_spec(name).replace("lesson_chars=200", "lesson_chars=300"))
    after = TACParams.from_string(f"preset=sac_td,model={M}")
    assert after != before and after.path_suffix != before.path_suffix
    # a stored string reads back as it was written, whatever the preset's file says now
    assert TACParams.from_string(before.to_string()) == before


def test_an_unknown_preset_is_refused_with_the_known_ones():
    with pytest.raises(ValueError, match="Unknown TAC preset 'sac'.*sac_td"):
        TACParams.from_string("preset=sac")

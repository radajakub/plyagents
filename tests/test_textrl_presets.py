"""Named configs: the TextRL's model= goes to every LLM component that names none and leaves the spec equal to the same spec in full; a preset starts
from a named config, the other options replace its roles, its string pins every role, and its path is its name and a hash of the whole config. The
roles that define a preset's algorithm keep what it defines when the player is built; the others are open."""

from __future__ import annotations

import pytest
from plybench.app import PlyBench
from plyllm import LLMConfig

from plyagents.textrl import params as params_module
from plyagents.textrl.actors.llm import LLMActorParams
from plyagents.textrl.apply.compress import CompressParams
from plyagents.textrl.build import check_textrl
from plyagents.textrl.configs.model import ModelSpec
from plyagents.textrl.configs.presets import preset_defines, preset_names, preset_spec
from plyagents.textrl.critics.llm import LLMCriticParams
from plyagents.textrl.enums import CriticType, SweepType
from plyagents.textrl.params import ROLES, TextRLParams
from plyagents.textrl.player import TextRLPlayer

M = "metacentrum:gpt-oss-120b"
OTHER = "metacentrum:qwen3-32b"


def test_the_textrl_model_goes_to_every_llm_component_that_names_none_and_the_spec_equals_it_in_full():
    short = TextRLParams.from_string(f"model={M},actor=llm[budget=80,apply=compress],critic=llm_v[model={OTHER}],feedback=llm,train_selector=greedy")
    full = TextRLParams.from_string(f"actor=llm[model={M},budget=80,apply=compress[model={M}]],critic=llm_v[model={OTHER}],feedback=llm[model={M}]")
    assert short == full and short.to_string() == full.to_string() and short.path_suffix == full.path_suffix
    actor, critic = short.actor.params, short.critic.params
    # the nested Apply takes it too; a component that names its own model keeps it
    assert isinstance(actor, LLMActorParams) and isinstance(actor.apply.params, CompressParams) and actor.apply.params.model == ModelSpec.from_string(M)
    assert isinstance(critic, LLMCriticParams) and critic.model == ModelSpec.from_string(OTHER)
    assert TextRLParams.from_string(f"model={M},actor=tabular") == TextRLParams.from_string("actor=tabular")


@pytest.mark.parametrize("name", preset_names())
def test_every_preset_parses_builds_and_round_trips(name):
    params = TextRLParams.from_string(f"preset={name},model={M}")
    assert params.preset == name and TextRLParams.from_string(params.to_string()) == params
    assert params.path_suffix.startswith(f"preset-{name}-") and len(params.path_suffix) == len(f"preset-{name}-") + params_module.HASH_LENGTH
    # model= reaches every LLM component; the bandit baselines have none
    assert all(model.model_name == "gpt-oss-120b" for model in params.models())
    registry = PlyBench(LLMConfig()).registry
    game = registry.build_engine(registry.game_config("tic_tac_toe:")).game
    assert isinstance(registry.build_player(game, registry.player_config(f"textrl:{params.to_string()}"), "i"), TextRLPlayer)


def test_the_presets_are_the_named_designs():
    assert set(preset_names()) == {"joint", "tac", "a2c", "mc", "hindsight", "reinforce", "greedy", "eps_greedy", "ucb"}
    assert all(TextRLParams.from_string(f"preset={name},model={M}").models() for name in ("joint", "tac", "a2c", "mc", "hindsight", "reinforce"))
    assert not any(TextRLParams.from_string(f"preset={name},model={M}").models() for name in ("greedy", "eps_greedy", "ucb"))
    sac = TextRLParams.from_string(f"preset=tac,model={M}")
    assert sac.critic.kind is CriticType.LLM_Q and sac.sweep.kind is SweepType.PARALLEL


def test_options_replace_the_roles_of_the_preset_and_a_preset_differs_from_its_spec_in_full():
    preset = TextRLParams.from_string(f"preset=tac,model={M}")
    overridden = TextRLParams.from_string(f"preset=tac,model={M},critic=llm_v[budget=600,apply=compress]")
    assert overridden.critic.kind is CriticType.LLM_V and overridden.actor == preset.actor and overridden.path_suffix != preset.path_suffix
    # the string names every role, so it reads back the same whatever the preset's file says later
    assert all(f"{role}=" in preset.to_string() for role in ("actor", "critic", "feedback", "actor_loss", "critic_loss", "sweep", "batch", "reference"))
    full = TextRLParams.from_string(f"model={M},{preset_spec('tac')}")
    assert full != preset and full.preset is None and not full.path_suffix.startswith("preset-")
    assert preset.display_name() is not None and (preset.display_name() or "").startswith("TextRL tac: ")


def test_a_changed_preset_file_gives_a_new_path(monkeypatch):
    before = TextRLParams.from_string(f"preset=tac,model={M}")
    monkeypatch.setattr(params_module, "preset_spec", lambda name: preset_spec(name).replace("lesson_chars=200", "lesson_chars=300"))
    after = TextRLParams.from_string(f"preset=tac,model={M}")
    assert after != before and after.path_suffix != before.path_suffix
    # a stored string reads back as it was written, whatever the preset's file says now
    assert TextRLParams.from_string(before.to_string()) == before


def test_an_unknown_preset_is_refused_with_the_known_ones():
    with pytest.raises(ValueError, match="Unknown TextRL preset 'sac'.*tac"):
        TextRLParams.from_string("preset=sac")


@pytest.mark.parametrize("name", preset_names())
def test_every_preset_defines_real_roles_and_params_and_keeps_its_own_definition(name):
    roles = {role.name: role for role in ROLES}
    # a role or param the definition misnames fails here, not at a run's build
    for role, defined in preset_defines(name).items():
        roles[role].collection.config(defined)
    check_textrl(TextRLParams.from_string(f"preset={name},model={M}"))


@pytest.mark.parametrize(
    ("spec", "refused"),
    [
        ("preset=tac,critic=llm_v", "critic=llm_v.* is not the preset tac \\(it defines critic=llm_q\\)"),
        ("preset=tac,actor_loss=llm[inputs=outcome]", "actor_loss=.* is not the preset tac"),
        ("preset=tac,critic_loss=llm[target=mc]", "critic_loss=.* is not the preset tac"),
        ("preset=tac,actor=llm[memory=false]", "actor=.* is not the preset tac"),
        ("preset=reinforce,critic=llm_q,critic_loss=llm[target=critic]", "critic=llm_q.* is not the preset reinforce"),
        ("preset=ucb,train_selector=greedy", "train_selector=greedy is not the preset ucb"),
    ],
)
def test_an_override_that_changes_what_a_preset_defines_is_refused_when_built(spec, refused):
    params = TextRLParams.from_string(f"{spec},model={M}")
    # it still parses, so a stored spec reads back whatever the preset's definition says later
    assert params.preset is not None
    with pytest.raises(ValueError, match=refused):
        check_textrl(params)


@pytest.mark.parametrize(
    "spec",
    [
        "preset=tac,critic=llm_q[budget=8000,apply=compress[reduce=mean],backward=llm],gamma=0.9,batch=epoch",
        "preset=tac,actor=llm[budget=8000,apply=rewrite,backward=llm[reasoning=false]],actor_loss=llm[lesson_chars=300]",
        "preset=reinforce,gamma=0.95,reward=length_penalty",
        "preset=ucb,train_selector=ucb[c=1.0]",
    ],
)
def test_an_override_of_an_open_role_or_param_keeps_the_preset(spec):
    check_textrl(TextRLParams.from_string(f"{spec},model={M}"))

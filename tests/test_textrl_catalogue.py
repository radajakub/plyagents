"""The catalogue of plyweb's algorithms page: every role, type, param and value of the code has a text and no text names an option the code lacks; the
defaults and the presets are the code's; the prompts are the components' own, rendered on the example, one option changed at a time."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from plyagents.textrl import catalogue as catalogue_module
from plyagents.textrl.catalogue import TEXTS, catalogue, missing_texts
from plyagents.textrl.configs.presets import preset_names
from plyagents.textrl.losses import llm_critic
from plyagents.textrl.params import TextRLParams


@pytest.fixture(scope="module")
def data() -> dict[str, Any]:
    return catalogue()


def _type(data: dict[str, Any], role: str, kind: str) -> dict[str, Any]:
    return next(item for collection in data["collections"] if collection["role"] == role for item in collection["types"] if item["type"] == kind)


def _keys(data: dict[str, Any]) -> set[str]:
    # every key a text may have: the roles, types, params and values of the catalogue, and the TextRL's options
    keys = {"textrl"}
    for collection in data["collections"]:
        keys.add(collection["role"])
        for kind in collection["types"]:
            keys.add(f"{collection['role']}/{kind['type']}")
            for param in kind["params"]:
                key = f"{collection['role']}/{kind['type']}/{param['name']}"
                keys |= {key, *(f"{key}={value}" for value in param.get("values", []))}
    for option in data["textrl"]["options"]:
        keys |= {f"textrl/{option['name']}", *(f"textrl/{option['name']}={value}" for value in option.get("values", []))}
    return keys


def test_every_role_type_param_and_value_has_a_text_and_every_text_names_one(data):
    assert missing_texts() == []
    assert set(TEXTS) - _keys(data) == set()
    assert set(catalogue_module.SENTENCES) - _keys(data) == set()


def test_the_catalogue_is_json(data):
    assert json.loads(json.dumps(data)) == data


def test_the_defaults_are_the_params_classes(data):
    params = {param["name"]: param for param in _type(data, "critic", "llm_q")["params"]}
    # the critic's step is its own default, not the scheduler type's
    assert params["scheduler"]["default"] == {"type": "constant", "params": {"beta": "0.4"}} and params["scheduler"]["component"] == "scheduler"
    assert params["value"] == {**params["value"], "kind": "bool", "default": "false", "values": ["true", "false"]}
    assert params["budget"]["kind"] == "int" and params["budget"]["optional"] and params["budget"]["default"] is None
    target = {param["name"]: param for param in _type(data, "critic_loss", "llm")["params"]}["target"]
    assert target["values"] == ["mc", "critic", "record"] and target["default"] == "mc"
    inputs = {param["name"]: param for param in _type(data, "actor_loss", "llm")["params"]}["inputs"]
    assert inputs["kind"] == "inputs" and "reasoning" in inputs["values"] and inputs["default"] == "none"


@pytest.mark.parametrize("name", preset_names())
def test_a_preset_written_in_full_reads_back_as_the_preset(data, name):
    preset = next(item for item in data["presets"] if item["name"] == name)
    assert TextRLParams.from_string(preset["full"]) == replace(TextRLParams.from_string(f"preset={name}"), preset=None)
    assert set(preset["config"]) == set(data["roles"])


def test_the_prompts_are_the_components_own_one_option_changed_at_a_time(data):
    examples = {(item["critic"], item["option"]): item for item in _type(data, "critic_loss", "llm")["examples"]}
    default, record = examples[("llm_q", None)], examples[("llm_q", "target=record")]
    assert llm_critic.SAMPLE in default["user"] and llm_critic.MEAN not in default["user"]
    assert llm_critic.MEAN in record["user"] and "The record of the move played here, this game included: 8 games (5 win, 3 loss), mean outcome +0.25." in record["user"]
    assert default["system"] == llm_critic.SYSTEM
    # an input the example's critic cannot give is named as such, not rendered
    actor_loss = {(item["critic"], item["option"]): item for item in _type(data, "actor_loss", "llm")["examples"]}
    assert "error" in actor_loss[("none", "inputs=q")] and "The notes the agent read here:" in actor_loss[("llm_q", "inputs=memory")]["user"]
    critic = {item["option"]: item for item in _type(data, "critic", "llm_q")["examples"]}
    assert "Notes from past assessments of these moves:" in critic[None]["user"] and critic["value=true"]["user"].endswith("as a number from -1 to +1.")

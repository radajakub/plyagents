"""The structure of the TextRL: every role of the params has one component in the player; the learning blocks share their machinery by family (a textual
memory, tables); and what a component needs from another is declared with its type and checked when the player is built, naming what gives it."""

from __future__ import annotations

import pytest
from plybench.app import PlyBench
from plyllm import LLMConfig

from plyagents.textrl.block import MemoryBlock, TableBlock
from plyagents.textrl.build import check_textrl, describe_textrl_player
from plyagents.textrl.configs.components import Need
from plyagents.textrl.critics.build import CRITICS
from plyagents.textrl.enums import Capability
from plyagents.textrl.losses.base import ActorLoss, CriticLoss, JointLoss
from plyagents.textrl.losses.build import ACTOR_LOSSES, CRITIC_LOSSES, JOINT_LOSSES
from plyagents.textrl.losses.llm import LLMLoss
from plyagents.textrl.params import ROLES, TextRLParams
from plyagents.textrl.player import TextRLPlayer
from plyagents.textrl.sweeps.build import SWEEPS

M = "metacentrum:gpt-oss-120b"


def _player(spec: str) -> TextRLPlayer:
    registry = PlyBench(LLMConfig()).registry
    game = registry.build_engine(registry.game_config("tic_tac_toe:")).game
    player = registry.build_player(game, registry.player_config(f"textrl:{spec}"), "i")
    assert isinstance(player, TextRLPlayer)
    return player


def test_every_role_has_one_component_and_one_line_in_the_summary():
    params = TextRLParams.from_string(f"preset=tac,model={M}")
    player = _player(params.to_string())
    assert list(player.components()) == [role.name for role in ROLES]
    summary = describe_textrl_player("textrl", player, params).splitlines()
    assert [line.split()[0] for line in summary[1 : len(ROLES) + 1]] == [role.label.split()[0] for role in ROLES]


def test_the_blocks_and_the_losses_share_their_machinery_by_family():
    sac = _player(f"preset=tac,model={M}")
    assert isinstance(sac.actor, MemoryBlock) and isinstance(sac.critic, MemoryBlock) and (sac.actor.kind, sac.critic.kind) == ("actor", "critic")
    assert isinstance(sac.losses.actor, (LLMLoss, ActorLoss)) and isinstance(sac.losses.critic, (LLMLoss, CriticLoss))
    tabular = _player("actor=tabular[target=critic],critic=tabular")
    assert isinstance(tabular.actor, TableBlock) and isinstance(tabular.critic, TableBlock) and isinstance(tabular.losses.joint, JointLoss)


def test_the_critics_and_the_sweeps_declare_what_they_give():
    numeric = {Capability.ASSESSMENT, Capability.VALUE, Capability.ACTION_VALUES, Capability.TARGET}
    assert CRITICS.gives(CRITICS.config("none")) == frozenset() and CRITICS.gives(CRITICS.config("oracle")) == numeric == CRITICS.gives(CRITICS.config("tabular"))
    # an LLM critic's numbers are for diagnostics: a value of the situation (V) or of each move (Q), never a target
    assert CRITICS.gives(CRITICS.config(f"llm_v[model={M}]")) == {Capability.ASSESSMENT}
    assert CRITICS.gives(CRITICS.config(f"llm_v[model={M},value=true]")) == {Capability.ASSESSMENT, Capability.VALUE}
    assert CRITICS.gives(CRITICS.config(f"llm_q[model={M},value=true]")) == {Capability.ASSESSMENT, Capability.ACTION_VALUES}
    assert SWEEPS.gives(SWEEPS.config("episode")) == {Capability.WHOLE_GAME} and SWEEPS.gives(SWEEPS.config("parallel")) == frozenset()


def test_the_losses_declare_what_they_need_by_their_options():
    assert ACTOR_LOSSES.needs(ACTOR_LOSSES.config(f"llm[model={M},inputs=outcome+delta+q]")) == (
        Need("critic", Capability.VALUE, "inputs=delta"),
        Need("critic", Capability.ACTION_VALUES, "inputs=q"),
    )
    assert ACTOR_LOSSES.needs(ACTOR_LOSSES.config(f"llm[model={M}]")) == (Need("critic", Capability.ASSESSMENT, "without inputs=outcome or trajectory"),)
    assert CRITIC_LOSSES.needs(CRITIC_LOSSES.config(f"llm[model={M},inputs=trajectory]")) == (
        Need("critic", Capability.ASSESSMENT),
        Need("sweep", Capability.WHOLE_GAME, "inputs=trajectory"),
    )
    assert JOINT_LOSSES.needs(JOINT_LOSSES.config("none")) == () and JOINT_LOSSES.needs(JOINT_LOSSES.config("log")) == (Need("sweep", Capability.WHOLE_GAME),)


@pytest.mark.parametrize(
    ("spec", "refused"),
    [
        ("actor=tabular[target=critic],critic=llm_v[model={M},value=true]", "actor target=critic needs a critic that gives values to learn toward"),
        ("actor=tabular,critic=llm_q[model={M}],feedback=llm[model={M},inputs=q]", r"feedback inputs=q needs a critic with action values \(.*llm_q\[value=true\]\)"),
        ("actor=tabular,critic=oracle,feedback=oracle,sweep=backward", "feedback needs a sweep that shows the losses the whole game at once"),
    ],
)
def test_a_missing_capability_is_refused_with_what_gives_it(spec, refused):
    with pytest.raises(ValueError, match=refused):
        check_textrl(TextRLParams.from_string(spec.format(M=M)))

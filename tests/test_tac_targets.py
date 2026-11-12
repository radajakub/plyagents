"""The targets of the tabular actor: the outcome, or the critic's value of the next state."""

from __future__ import annotations

import pytest

from plyagents.tac.model.assessment import Assessment
from plyagents.tac.targets.build import TARGETS
from plyagents.tac.targets.critic import CriticTarget
from plyagents.tac.targets.outcome import OutcomeTarget


def test_the_outcome_target_gives_g_at_every_step():
    assert OutcomeTarget().targets([Assessment(0.1, ""), Assessment(None, "")], 1.0) == [1.0, 1.0]


def test_the_critic_target_gives_the_next_value_and_g_at_the_end():
    assessments = [Assessment(0.1, ""), Assessment(0.2, ""), Assessment(0.3, "")]
    assert CriticTarget().targets(assessments, 1.0) == [0.2, 0.3, 1.0]
    assert CriticTarget().targets([Assessment(None, "")], -1.0) == [-1.0]
    assert CriticTarget().targets([], 1.0) == []


def test_the_critic_target_needs_values():
    with pytest.raises(ValueError, match="values"):
        CriticTarget().targets([Assessment(None, ""), Assessment(None, "")], 1.0)


def test_the_collection_builds_both_targets(services):
    assert isinstance(TARGETS.build(TARGETS.config("mc"), services), OutcomeTarget)
    assert isinstance(TARGETS.build(TARGETS.config("critic"), services), CriticTarget)


def test_each_target_names_its_source():
    assert OutcomeTarget().sources(2) == ["outcome", "outcome"]
    assert CriticTarget().sources(3) == ["critic", "critic", "outcome"] and CriticTarget().sources(0) == []

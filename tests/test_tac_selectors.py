"""The selectors: greedy, epsilon-greedy and UCB pick among the legal actions, seeded, and their collection builds them."""

from __future__ import annotations

import numpy as np
import pytest

from plyagents.tac.configs.components import ComponentConfig
from plyagents.tac.enums import SelectorType
from plyagents.tac.selectors.build import SELECTORS
from plyagents.tac.selectors.eps_greedy import EpsGreedy, EpsGreedyParams
from plyagents.tac.selectors.greedy import Greedy
from plyagents.tac.selectors.ucb import UCB


def test_greedy_takes_the_best_and_breaks_ties_at_random():
    rng = np.random.default_rng(0)
    assert {Greedy().select([0.0, 1.0, 0.5], [1, 1, 1], rng) for _ in range(20)} == {1}
    assert {Greedy().select([1.0, 0.0, 1.0], [1, 1, 1], rng) for _ in range(50)} == {0, 2}


def test_eps_greedy_explores_at_rate_eps():
    rng = np.random.default_rng(0)
    picks = [EpsGreedy(0.3).select([1.0, 0.0, 0.0, 0.0], [1, 1, 1, 1], rng) for _ in range(4000)]
    # the best action is taken by the greedy part and by a quarter of the random part
    off = sum(pick != 0 for pick in picks) / len(picks)
    assert off == pytest.approx(0.3 * 3 / 4, abs=0.03)
    assert EpsGreedy(0.0).select([0.0, 1.0], [1, 1], rng) == 1


def test_ucb_tries_untried_actions_first_then_balances_value_and_count():
    rng = np.random.default_rng(0)
    assert UCB(1.0).select([5.0, 0.0, 0.0], [10, 0, 3], rng) == 1
    # with every action tried, the rarely tried one wins for a large c and the best one for c = 0
    assert UCB(10.0).select([1.0, 0.9], [100, 1], rng) == 1
    assert UCB(0.0).select([1.0, 0.9], [100, 1], rng) == 0


def test_the_same_seed_gives_the_same_choices():
    picks = [[EpsGreedy(0.5).select([0.0, 0.0, 0.0], [1, 1, 1], rng) for _ in range(20)] for rng in (np.random.default_rng(7), np.random.default_rng(7))]
    assert picks[0] == picks[1]


def test_the_collection_builds_each_selector_from_its_config(services):
    config = SELECTORS.config("eps_greedy[eps=0.2]")
    assert config == ComponentConfig(SelectorType.EPS_GREEDY, EpsGreedyParams(eps=0.2))
    selector = SELECTORS.build(config, services)
    assert isinstance(selector, EpsGreedy) and selector.eps == 0.2
    assert isinstance(SELECTORS.build(SELECTORS.config("greedy"), services), Greedy)
    ucb = SELECTORS.build(SELECTORS.config("ucb[c=2.0]"), services)
    assert isinstance(ucb, UCB) and ucb.c == 2.0
    with pytest.raises(ValueError, match="Invalid selector"):
        SELECTORS.config("softmax")


def test_each_pick_records_how_it_was_made_with_the_same_draws_as_select():
    # the record of a pick: the rule, the scores compared, the ties broken at random, the untried actions, whether it explored
    greedy = Greedy().choose([1.0, 0.0, 1.0], [1, 1, 1], np.random.default_rng(3))
    assert (greedy.rule, greedy.scores, greedy.tied) == ("greedy", (1.0, 0.0, 1.0), (0, 2)) and greedy.index in (0, 2)

    untried = UCB(1.0).choose([5.0, 0.0, 0.0], [10, 0, 3], np.random.default_rng(3))
    assert (untried.index, untried.rule, untried.untried) == (1, "untried", (1,))
    scored = UCB(0.0).choose([1.0, 0.9], [100, 1], np.random.default_rng(3))
    assert (scored.index, scored.rule, scored.scores) == (0, "ucb", (1.0, 0.9))

    explored = [EpsGreedy(0.5).choose([1.0, 0.0], [1, 1], rng).to_dict() for rng in [np.random.default_rng(seed) for seed in range(20)]]
    assert {pick["explored"] for pick in explored} == {True, False}
    assert all(pick["rule"] == ("random" if pick["explored"] else "greedy") for pick in explored)
    # choose draws as select does, so recording changes no game
    for selector in (Greedy(), EpsGreedy(0.3), UCB(1.0)):
        picks = [selector.select([0.5, 0.5, 0.1], [2, 2, 1], rng) for rng in [np.random.default_rng(seed) for seed in range(30)]]
        assert picks == [selector.choose([0.5, 0.5, 0.1], [2, 2, 1], rng).index for rng in [np.random.default_rng(seed) for seed in range(30)]]

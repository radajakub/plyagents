"""The schedulers: the step size of an update from the number of updates so far."""

from __future__ import annotations

import pytest

from plyagents.textrl.enums import ShareRule
from plyagents.textrl.model.table import Table
from plyagents.textrl.schedulers.build import SCHEDULERS
from plyagents.textrl.schedulers.constant import ConstantScheduler
from plyagents.textrl.schedulers.harmonic import HarmonicScheduler


def test_the_constant_scheduler_gives_beta_at_every_count():
    assert [ConstantScheduler(0.2).step_size(count) for count in (0, 1, 100)] == [0.2, 0.2, 0.2]


def test_the_collection_builds_the_constant_scheduler_with_its_beta(services):
    scheduler = SCHEDULERS.build(SCHEDULERS.config("constant[beta=0.05]"), services)
    assert isinstance(scheduler, ConstantScheduler) and scheduler.beta == 0.05
    assert SCHEDULERS.config("constant").to_string() == "constant"


def test_the_harmonic_scheduler_makes_the_estimate_the_sample_mean(services):
    scheduler = SCHEDULERS.build(SCHEDULERS.config("harmonic"), services)
    assert isinstance(scheduler, HarmonicScheduler) and [scheduler.step_size(count) for count in (0, 1, 3)] == [1.0, 0.5, 0.25]
    table = Table()
    targets = [1.0, -1.0, 1.0, 1.0, -1.0]
    for target in targets:
        table.update("z", target, scheduler.step_size)
    assert abs(table.get("z").value - sum(targets) / len(targets)) < 1e-12 and table.get("z").count == len(targets)


def test_the_share_of_a_write_is_one_step_per_write_or_the_steps_of_its_samples_capped():
    # per write: the step of the entry's next write, whatever the number of gradients (the mean gradient)
    assert ConstantScheduler(0.25).share(writes=4, samples=9, n=3) == 0.25 and HarmonicScheduler().share(writes=3, samples=9, n=3) == 0.25
    # per sample: as if the n gradients were written one after another; harmonic is then the running mean over all samples
    assert abs(ConstantScheduler(0.25, ShareRule.SAMPLE).share(writes=4, samples=9, n=3) - (1 - 0.75**3)) < 1e-12
    assert abs(HarmonicScheduler(ShareRule.SAMPLE).share(writes=1, samples=6, n=2) - 2 / 8) < 1e-12
    assert ConstantScheduler(0.25, ShareRule.SAMPLE).share(0, 0, 1) == ConstantScheduler(0.25).share(0, 0, 1)
    assert ConstantScheduler(0.5, ShareRule.SAMPLE, cap=0.6).share(0, 0, 5) == 0.6


def test_the_share_options_are_named_only_when_they_differ_from_the_defaults(services):
    config = SCHEDULERS.config("constant[beta=0.25,per=sample,cap=0.8]")
    scheduler = SCHEDULERS.build(config, services)
    assert isinstance(scheduler, ConstantScheduler) and (scheduler.per, scheduler.cap) == (ShareRule.SAMPLE, 0.8)
    assert SCHEDULERS.config(config.to_string()) == config and SCHEDULERS.config("constant[beta=0.25,per=write]").to_string() == "constant[beta=0.25]"
    harmonic = SCHEDULERS.build(SCHEDULERS.config("harmonic[per=sample]"), services)
    assert isinstance(harmonic, HarmonicScheduler) and harmonic.per is ShareRule.SAMPLE and SCHEDULERS.config("harmonic").to_string() == "harmonic"
    with pytest.raises(ValueError, match="Invalid scheduler cap"):
        SCHEDULERS.config("constant[cap=1.5]")
    with pytest.raises(ValueError, match="Invalid scheduler per"):
        SCHEDULERS.config("constant[per=game]")

"""The schedulers: the step size of an update from the number of updates so far."""

from __future__ import annotations

from plyagents.tac.model.table import Table
from plyagents.tac.schedulers.build import SCHEDULERS
from plyagents.tac.schedulers.constant import ConstantScheduler
from plyagents.tac.schedulers.harmonic import HarmonicScheduler


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

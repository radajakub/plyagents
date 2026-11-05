"""The schedulers: the step size of an update from the number of updates so far."""

from __future__ import annotations

from plyagents.tac.schedulers.build import SCHEDULERS
from plyagents.tac.schedulers.constant import ConstantScheduler


def test_the_constant_scheduler_gives_beta_at_every_count():
    assert [ConstantScheduler(0.2).step_size(count) for count in (0, 1, 100)] == [0.2, 0.2, 0.2]


def test_the_collection_builds_the_constant_scheduler_with_its_beta(services):
    scheduler = SCHEDULERS.build(SCHEDULERS.config("constant[beta=0.05]"), services)
    assert isinstance(scheduler, ConstantScheduler) and scheduler.beta == 0.05
    assert SCHEDULERS.config("constant").to_string() == "constant"

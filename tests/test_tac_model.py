"""The TAC data model: tables of running estimates, their updates, and the update of one game, all surviving JSON unchanged."""

from __future__ import annotations

from plyagents.tac.model.step import TACStep
from plyagents.tac.model.table import Sample, Table, TableUpdate
from plyagents.tac.model.update import TACUpdate


def test_a_table_starts_at_zero_and_moves_toward_its_targets():
    table = Table()
    assert (table.get("z").value, table.get("z").count) == (0.0, 0)
    table.update("z", 1.0, lambda count: 0.5)
    table.update("z", 1.0, lambda count: 0.5)
    assert (table.get("z").value, table.get("z").count) == (0.75, 2)


def test_a_table_gets_the_step_size_from_the_number_of_updates_so_far():
    counts: list[int] = []
    table = Table()
    for _ in range(3):
        table.update("z", 1.0, lambda count: counts.append(count) or 1.0)
    assert counts == [0, 1, 2]


def test_a_table_survives_json():
    table = Table()
    table.update("z", -1.0, lambda count: 0.3)
    assert Table.from_dict(table.to_dict()).to_dict() == table.to_dict()


def test_an_update_reads_back_equal():
    update = TACUpdate(TableUpdate((Sample("z", 1.0, "a"), Sample("y", -0.5, "b"))), TableUpdate((Sample("z", 1.0),)))
    assert TACUpdate.from_dict(update.to_dict()) == update
    assert TACUpdate.from_dict(TACUpdate(None, None).to_dict()) == TACUpdate(None, None)


def test_a_step_reads_back_equal():
    step = TACStep("You play as first player.", "<C1R1>", 0, "...")
    assert TACStep.from_dict(step.to_dict()) == step

"""The TextRL data model: tables of running estimates, their updates, and the update of one game, all surviving JSON unchanged."""

from __future__ import annotations

import json
from dataclasses import asdict

from plyagents.textrl.model.counts import MoveCounts
from plyagents.textrl.model.feedback import StepFeedback
from plyagents.textrl.model.memory import MemoryWrite
from plyagents.textrl.model.reference import Reference, ReferenceStep, ReferenceUpdate, Values
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.model.table import Sample, Table, TableUpdate
from plyagents.textrl.model.update import TextRLUpdate


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
    update = TextRLUpdate(TableUpdate((Sample("z", 1.0, "a"), Sample("y", -0.5, "b"))), TableUpdate((Sample("z", 1.0),)))
    # each part is parsed by its component; here both are tables
    assert TextRLUpdate.from_dict(update.to_dict(), TableUpdate.from_dict, TableUpdate.from_dict) == update
    assert TextRLUpdate.from_dict(TextRLUpdate(None, None).to_dict(), TableUpdate.from_dict, TableUpdate.from_dict) == TextRLUpdate(None, None)


def test_a_step_reads_back_equal():
    step = TextRLStep("You play as first player.", "<C1R1>", 0, "...")
    assert TextRLStep.from_dict(step.to_dict()) == step


def test_a_step_with_its_legal_actions_and_context_reads_back_equal():
    step = TextRLStep("key", "<C1R1>", 0, "...", ("<C1R1>", "<C2R1>"), (0, 1), "Notes from past experience in this situation:\nplay the centre")
    assert TextRLStep.from_dict(step.to_dict()) == step


def test_a_step_reads_back_without_the_empty_values_the_record_drops():
    # StepData drops None values: a failed move's action, no context, no rules
    assert TextRLStep.from_dict({"key": "k", "player": 1, "os_state": "s", "legal": ["a"], "legal_numbers": [0]}) == TextRLStep("k", None, 1, "s", ("a",), (0,))


def test_feedback_reads_back_equal():
    feedback = StepFeedback("mistake", "left the row open", "block the row", "the position was lost")
    assert StepFeedback.from_dict(feedback.to_dict()) == feedback


def _record() -> ReferenceUpdate:
    counts = {"a": MoveCounts({"loss": 2, "win": 1}, -1.0, 3.0)}
    steps = (
        ReferenceStep(1, "z", "a", ("a", "b"), counts, Values(0.25, (0.5, None)), Values(0.0, (0.0, -1.0)), "good"),
        ReferenceStep(2, "y", None, ("c",), {}, Values(None, None), None, None),
    )
    return ReferenceUpdate(steps, -1.0)


def test_an_update_with_a_reference_record_reads_back_equal_and_an_older_one_has_none():
    update = TextRLUpdate(None, TableUpdate((Sample("z", 1.0),)), _record())
    data = json.loads(json.dumps(update.to_dict()))
    assert data["reference"]["result"] == "loss"
    assert TextRLUpdate.from_dict(data, TableUpdate.from_dict, TableUpdate.from_dict) == update
    assert TextRLUpdate.from_dict({"actor": None, "critic": None}, TableUpdate.from_dict, TableUpdate.from_dict) == TextRLUpdate(None, None)


def test_the_reference_counts_every_played_step_and_a_snapshot_stays_as_it_was():
    reference = Reference()
    before = reference.before("z")
    _record().apply_to(reference)
    _record().apply_to(reference)
    # the step without a move is not counted; the sums are of the shaped outcome
    assert before == {} and reference.before("z") == {"a": MoveCounts({"loss": 2}, -2.0, 2.0)} and "y" not in reference.counts
    snapshot = reference.before("z")
    reference.add("z", "a", 1.0)
    assert snapshot["a"].visits == 2 and reference.before("z")["a"] == MoveCounts({"loss": 2, "win": 1}, -1.0, 3.0)
    assert Reference.from_dict(json.loads(json.dumps(reference.to_dict()))).counts == reference.counts


def test_a_step_reads_the_actors_reasoning_from_the_step_data():
    base = {"key": "k", "player": 0, "os_state": "s", "legal": [], "legal_numbers": []}
    assert TextRLStep.from_dict(base | {"reasoning_trace": "door 1 is as good as any", "full_output": "Action: <door:1>"}).reasoning == "door 1 is as good as any"
    # without a trace, the full output, as plybench's StepData reads "what it thought"; a bot records neither
    assert TextRLStep.from_dict(base | {"full_output": "I pick door 1. Action: <door:1>"}).reasoning == "I pick door 1. Action: <door:1>"
    assert TextRLStep.from_dict(base).reasoning is None
    step = TextRLStep("k", "a", 0, "s", reasoning="because")
    assert TextRLStep.from_dict(step.to_dict()) == step


def test_an_update_with_its_judging_record_reads_back_equal():
    record = {"assessments": [{"value": None, "text": "", "read": [{"memory": "critic", "key": "z"}]}], "feedback": {"verdicts": [], "dropped": [], "inputs": {"shown": []}}}
    update = TextRLUpdate(None, TableUpdate((Sample("z", 1.0, "a", "outcome"),)), _record(), record)
    assert TextRLUpdate.from_dict(json.loads(json.dumps(update.to_dict())), TableUpdate.from_dict, TableUpdate.from_dict) == update


def test_a_memory_write_reads_back_equal():
    write = MemoryWrite("z", "old", "lesson", 0.5, "ol", "ol\nlesson", True, (1, 3), 2, 40, 20, 4, True, False, ("lesson",), 0.4)
    assert MemoryWrite.from_dict(json.loads(json.dumps(asdict(write)))) == write

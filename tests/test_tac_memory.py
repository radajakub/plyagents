"""The textual memory: entries with their counts survive JSON, and a game's update counts the visits and replaces the written
entries, deterministically."""

from __future__ import annotations

from plyagents.tac.model.feedback import StepFeedback
from plyagents.tac.model.memory import Entry, KeyedMemory, MemoryUpdate, MemoryWrite


def _update() -> MemoryUpdate:
    return MemoryUpdate(
        (("z", "<C1R1>"), ("y", None), ("z", "<C1R1>")),
        (StepFeedback("good", "took the centre", "take the centre"), StepFeedback(), StepFeedback("mistake", "left a row open", None, "lost")),
        (MemoryWrite("z", "", "take the centre", 1.0, "", "take the centre", False),),
    )


def test_an_update_reads_back_equal():
    assert MemoryUpdate.from_dict(_update().to_dict()) == _update()


def test_commit_counts_visits_and_actions_and_replaces_the_written_entries():
    memory = KeyedMemory({"z": Entry("old", 2, {"<C1R1>": 1}, 1)})
    _update().apply_to(memory)
    assert memory.get("z") == Entry("take the centre", 4, {"<C1R1>": 3}, 2)
    # a failed move is a visit without an action; a key visited but not written keeps an empty entry
    assert memory.get("y") == Entry("", 1, {}, 0)


def test_a_memory_survives_json():
    memory = KeyedMemory()
    _update().apply_to(memory)
    assert KeyedMemory.from_dict(memory.to_dict()).to_dict() == memory.to_dict()
    assert memory.get("never seen") == Entry()

"""The Apply operator: a lesson is appended to the entry after the old text is shortened to (1 - alpha) B; truncate keeps the
most recent characters without a call, compress asks an LLM once and cuts an over-long answer; an empty entry or a short one
needs no call, and without a budget the lesson is only appended."""

from __future__ import annotations

import asyncio

import pytest

from plyagents.tac.apply.build import APPLIES
from plyagents.tac.apply.compress import CompressApply
from plyagents.tac.apply.truncate import TruncateApply
from plyagents.tac.configs.model import ModelSpec
from plyagents.tac.model.memory import Entry, KeyedMemory
from plyagents.tac.schedulers.constant import ConstantScheduler

MODEL = ModelSpec.from_string("metacentrum:gpt-oss-120b[reasoning_effort=low]")


def _write(apply, memory, lessons, calls, budget: int | None = 40, alpha: float = 0.5):
    return asyncio.run(apply.write(memory, lessons, ConstantScheduler(alpha), budget, calls, "apply"))


def test_an_empty_entry_becomes_the_lesson_without_a_call(stub_calls):
    calls, stub, recorded = stub_calls([])
    [write] = _write(CompressApply(MODEL), KeyedMemory(), [("z", "take the centre")], calls)
    assert (write.old, write.new, write.cut) == ("", "take the centre", False)
    assert stub.seeds == [] and recorded == []


def test_without_a_budget_the_lesson_is_only_appended(stub_calls):
    calls, _, recorded = stub_calls([])
    memory = KeyedMemory({"z": Entry("x" * 100)})
    [write] = _write(CompressApply(MODEL), memory, [("z", "block the row")], calls, budget=None)
    assert write.new == "x" * 100 + "\nblock the row" and recorded == []


def test_truncate_keeps_the_most_recent_share_and_appends(calls):
    memory = KeyedMemory({"z": Entry("0123456789" * 4)})
    [write] = _write(TruncateApply(), memory, [("z", "new")], calls)
    # alpha 0.5 of 40: the last 20 characters are kept, then the lesson
    assert write.compressed == ("0123456789" * 4)[-20:] and write.new == write.compressed + "\nnew" and not write.cut


def test_the_lessons_of_a_key_are_joined_in_step_order_and_alpha_follows_its_writes(calls):
    memory = KeyedMemory({"z": Entry("old", writes=3)})
    writes = asyncio.run(TruncateApply().write(memory, [("z", "first"), ("y", "other"), ("z", "second")], _AverageScheduler(), None, calls, "apply"))
    assert [(write.key, write.lesson, write.alpha) for write in writes] == [("z", "first\nsecond", 0.25), ("y", "other", 1.0)]


def test_compress_asks_once_with_the_share_and_cuts_an_over_long_answer(stub_calls):
    calls, _, recorded = stub_calls(["s" * 30])
    memory = KeyedMemory({"z": Entry("x" * 40)})
    [write] = _write(CompressApply(MODEL), memory, [("z", "lesson")], calls)
    assert len(recorded) == 1 and recorded[0].role == "apply"
    assert "at most 20 characters" in recorded[0].prompt_message and "x" * 40 in recorded[0].prompt_message
    # the answer overshot its 20 characters: cut and flagged
    assert write.compressed == "s" * 20 and write.cut


def test_compress_needs_no_call_when_the_entry_fits_its_share(stub_calls):
    calls, _, recorded = stub_calls([])
    [write] = _write(CompressApply(MODEL), KeyedMemory({"z": Entry("short")}), [("z", "lesson")], calls)
    assert write.new == "short\nlesson" and recorded == []


def test_a_lesson_over_the_budget_is_cut_and_flagged(calls):
    [write] = _write(TruncateApply(), KeyedMemory({"z": Entry("abc")}), [("z", "y" * 60)], calls)
    assert len(write.new) == 40 and write.new.startswith("abc\n") and write.cut


def test_the_collection_builds_both_and_compress_needs_a_model(services):
    assert isinstance(APPLIES.build(APPLIES.config("truncate"), services), TruncateApply)
    compress = APPLIES.build(APPLIES.config("compress[model=metacentrum:gpt-oss-120b[reasoning_effort=low]]"), services)
    assert isinstance(compress, CompressApply) and compress.model == MODEL
    with pytest.raises(ValueError, match="needs a model"):
        APPLIES.build(APPLIES.config("compress"), services)


class _AverageScheduler(ConstantScheduler):
    # 1 / (k + 1), to show that alpha follows the key's writes
    def __init__(self) -> None:
        super().__init__(0.0)

    def step_size(self, count: int) -> float:
        return 1 / (count + 1)


def test_a_write_records_its_steps_its_share_and_the_call_that_shortened_it(stub_calls):
    calls, _, recorded = stub_calls(["s" * 30])
    memory = KeyedMemory({"z": Entry("x" * 40, writes=2)})
    [write] = asyncio.run(CompressApply(MODEL).write(memory, [("z", "first"), ("z", "second")], ConstantScheduler(0.5), 40, calls, "apply", [1, 3]))

    assert (write.steps, write.writes_before, write.budget, write.keep, write.call) == ((1, 3), 2, 40, 20, 0)
    # the shortening overshot its share; the joined lesson fits its own
    assert (write.cut_old, write.cut_lesson, write.cut) == (True, False, True)
    assert recorded[0].about == {"key": "z", "steps": [1, 3]}


def test_a_lesson_cut_to_the_budget_is_told_apart_from_a_shortening_that_overshot(calls):
    [write] = _write(TruncateApply(), KeyedMemory({"z": Entry("abc")}), [("z", "y" * 60)], calls)
    assert (write.cut_old, write.cut_lesson, write.call) == (False, True, None)

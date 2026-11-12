"""The Apply operator: a lesson is appended to the entry after the old text is shortened to (1 - alpha) B; truncate keeps the
most recent characters without a call, compress asks an LLM once and cuts an over-long answer; an empty entry or a short one
needs no call, and without a budget the lesson is only appended."""

from __future__ import annotations

import asyncio
import json

import pytest

from plyagents.textrl.apply.base import changed
from plyagents.textrl.apply.build import APPLIES
from plyagents.textrl.apply.compress import CompressApply
from plyagents.textrl.apply.rewrite import RewriteApply
from plyagents.textrl.apply.truncate import TruncateApply
from plyagents.textrl.configs.model import ModelSpec
from plyagents.textrl.enums import ReduceType, ShareRule
from plyagents.textrl.model.feedback import StepFeedback
from plyagents.textrl.model.memory import Entry, KeyedMemory
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.schedulers.constant import ConstantScheduler

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


def test_several_gradients_of_a_key_make_one_write_with_the_share_for_all_of_them_and_count_as_samples(calls):
    memory = KeyedMemory({"z": Entry("old", writes=2, samples=2)})
    scheduler = ConstantScheduler(0.25, ShareRule.SAMPLE)
    update = asyncio.run(
        TruncateApply().update(memory, [TextRLStep("z", "a", 0, "s"), TextRLStep("z", "b", 0, "s")], [StepFeedback()] * 2, ["one", "two"], scheduler, 100, calls, "apply")
    )
    [write] = update.writes
    # joined, as before batches; the share of two samples, the parts recorded and the realised step measured
    assert write.lesson == "one\ntwo" and write.parts == ("one", "two") and abs(write.alpha - (1 - 0.75**2)) < 1e-12
    assert write.changed == changed("old", write.new) and 0 < (write.changed or 0) <= 1
    update.apply_to(memory)
    assert (memory.get("z").writes, memory.get("z").samples) == (3, 4)


def test_compress_with_the_mean_combines_the_gradients_and_shortens_the_entry_in_one_call(stub_calls):
    calls, _, recorded = stub_calls([json.dumps({"kept": "kept notes", "combined": "prefer the centre"})])
    memory = KeyedMemory({"z": Entry("x" * 40)})
    [write] = asyncio.run(CompressApply(MODEL, ReduceType.MEAN).write(memory, [("z", "take the centre"), ("z", "the centre won")], ConstantScheduler(0.5), 40, calls, "apply"))
    prompt = recorded[0].prompt_message
    assert len(recorded) == 1 and "at most 20 characters" in prompt and "1. take the centre\n2. the centre won" in prompt and "x" * 40 in prompt
    assert (write.compressed, write.lesson, write.new, write.call) == ("kept notes", "prefer the centre", "kept notes\nprefer the centre", 0)


def test_compress_with_the_mean_only_combines_when_the_entry_fits_and_writes_one_gradient_as_before(stub_calls):
    calls, _, recorded = stub_calls(["prefer the centre"])
    [write] = asyncio.run(CompressApply(MODEL, ReduceType.MEAN).write(KeyedMemory({"z": Entry("short")}), [("z", "a"), ("z", "b")], ConstantScheduler(0.5), 40, calls, "apply"))
    assert len(recorded) == 1 and recorded[0].prompt_message.startswith("Combine these 2 lessons") and write.new == "short\nprefer the centre"
    calls, _, recorded = stub_calls([])
    [write] = asyncio.run(CompressApply(MODEL, ReduceType.MEAN).write(KeyedMemory({"z": Entry("short")}), [("z", "a")], ConstantScheduler(0.5), 40, calls, "apply"))
    assert recorded == [] and write.new == "short\na"


def test_rewrite_asks_once_with_the_step_and_the_budget_and_records_the_realised_step(stub_calls):
    calls, _, recorded = stub_calls(["tried: b2 3 times, lost 2; prefer the centre" + "!" * 60])
    memory = KeyedMemory({"z": Entry("take b2")})
    [write] = asyncio.run(RewriteApply(MODEL).write(memory, [("z", "b2 lost"), ("z", "b2 lost again")], ConstantScheduler(0.25), 60, calls, "apply"))
    prompt = recorded[0].prompt_message
    assert "Change about 25% of the notes" in prompt and "at most 60 characters" in prompt and "Notes:\ntake b2" in prompt and "2. b2 lost again" in prompt
    assert len(write.new) == 60 and write.cut and write.call == 0 and write.parts == ("b2 lost", "b2 lost again") and write.changed == changed("take b2", write.new)
    # an empty entry with one gradient takes it as it is, without a call
    calls, _, recorded = stub_calls([])
    [write] = asyncio.run(RewriteApply(MODEL).write(KeyedMemory(), [("z", "take the centre")], ConstantScheduler(0.25), 60, calls, "apply"))
    assert recorded == [] and write.new == "take the centre"


def test_the_collection_builds_rewrite_and_compress_with_the_mean(services):
    rewrite = APPLIES.build(APPLIES.config("rewrite[model=metacentrum:gpt-oss-120b[reasoning_effort=low]]"), services)
    compress = APPLIES.build(APPLIES.config("compress[model=metacentrum:gpt-oss-120b[reasoning_effort=low],reduce=mean]"), services)
    assert isinstance(rewrite, RewriteApply) and isinstance(compress, CompressApply) and compress.reduce_type is ReduceType.MEAN
    assert APPLIES.config("compress[model=metacentrum:gpt-oss-120b,reduce=join]").to_string() == "compress[model=metacentrum:gpt-oss-120b]"
    with pytest.raises(ValueError, match="needs a model"):
        APPLIES.build(APPLIES.config("rewrite"), services)


def test_the_mean_of_several_gradients_states_how_many_support_each_point_and_a_shortening_keeps_the_counts(stub_calls):
    # 3 against 1 must not read as 2 against 2: the combine, the combine with shortening and the rewrite of several gradients ask for the support of each
    # point; a shortening keeps a count as written
    lessons = [("z", "switch"), ("z", "switch"), ("z", "switch"), ("z", "stay")]
    calls, _, recorded = stub_calls(["switch (3 of 4 lessons); stay (1 of 4 lessons)"])
    _write(CompressApply(MODEL, ReduceType.MEAN), KeyedMemory({"z": Entry("short")}), lessons, calls)
    calls, _, combined = stub_calls([json.dumps({"kept": "kept", "combined": "c"})])
    _write(CompressApply(MODEL, ReduceType.MEAN), KeyedMemory({"z": Entry("x" * 40)}), lessons, calls)
    calls, _, rewritten = stub_calls(["new notes"])
    _write(RewriteApply(MODEL), KeyedMemory({"z": Entry("old")}), lessons, calls)
    for prompt in (recorded[0].prompt_message, combined[0].prompt_message, rewritten[0].prompt_message):
        assert "how many of the 4" in prompt and '"k of 4 lessons: ..."' in prompt and "each side with its count" in prompt
    assert "uncertain" not in recorded[0].prompt_message + combined[0].prompt_message
    assert "keep any count of the lessons that support a point as it is written" in combined[0].prompt_message
    calls, _, shortened = stub_calls(["kept"])
    _write(CompressApply(MODEL), KeyedMemory({"z": Entry("x" * 40)}), [("z", "one")], calls)
    assert "keep any count of the lessons that support a point as it is written" in shortened[0].prompt_message
    # one gradient has no support to state
    calls, _, single = stub_calls(["new notes"])
    _write(RewriteApply(MODEL), KeyedMemory({"z": Entry("old")}), [("z", "switch")], calls)
    assert "how many of the" not in single[0].prompt_message

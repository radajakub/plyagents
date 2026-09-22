from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

from plybench.app import PlyBench
from plybench.harness.training.training import TrainingHarness

from game_agents.reasoning_bank.memory.store import RECORDS_FILE
from game_agents.reasoning_bank.register import register_reasoning_bank
from tests.stub_llm import StubLLM

TRAINEE = "reasoning_bank:actions:text:gemini:stub:memory=on,top_n=1"
TRAINER = "random:distribution=uniform"
SCHEDULE = "num_epochs=1,num_training_games=2,num_test_games=2"


def _harness(experiment: str) -> tuple[TrainingHarness, StubLLM]:
    op = PlyBench()
    llm = StubLLM()
    register_reasoning_bank(op.registry, llm)
    return TrainingHarness(experiment, op, ["tic_tac_toe:"], [TRAINEE], [TRAINER], [TRAINER], [SCHEDULE], 1), llm


def _run(harness: TrainingHarness) -> None:
    asyncio.run(harness.run(sync=True))


def _checkpoint(harness: TrainingHarness, epoch: int) -> Path:
    run = harness._matrix()[0]
    return harness.path_builder.checkpoint(harness.experiment, run, epoch)


def test_training_fills_and_replays_the_bank(tmp_path: Path) -> None:
    os.chdir(tmp_path)
    harness, llm = _harness("test")

    _run(harness)

    # epoch 0 is the untrained baseline, so its checkpoint exists but holds nothing
    assert _checkpoint(harness, 0).exists()
    assert not (_checkpoint(harness, 0) / RECORDS_FILE).read_text().strip()

    trained = (_checkpoint(harness, 1) / RECORDS_FILE).read_text()
    assert trained.strip(), "the training phase induced nothing"

    # a rerun of a finished experiment plays nothing and derives nothing
    calls = llm.generate_calls
    _run(harness)
    assert llm.generate_calls == calls

    # an epoch whose checkpoint is gone replays its recorded games through the learner; the cached
    # derivations must reproduce the bank exactly, even though recomputing them would not (the stub
    # numbers every induced item by its call index)
    shutil.rmtree(_checkpoint(harness, 1))
    replayed, replayed_llm = _harness("test")
    _run(replayed)
    assert (_checkpoint(replayed, 1) / RECORDS_FILE).read_text() == trained
    assert replayed_llm.generate_calls == 0, "the replay re-sampled instead of reading its cached derivations"


def test_eval_phase_is_frozen(tmp_path: Path) -> None:
    os.chdir(tmp_path)
    harness, _ = _harness("frozen")
    _run(harness)

    run = harness._matrix()[0]
    results = harness.get_results()
    for epoch in range(2):
        before = (harness.path_builder.checkpoint(harness.experiment, run, epoch) / RECORDS_FILE).read_text()
        tests = results.find(run, epoch).tests
        assert tests and all(tracker.is_complete() for tracker in tests)
        assert (harness.path_builder.checkpoint(harness.experiment, run, epoch) / RECORDS_FILE).read_text() == before

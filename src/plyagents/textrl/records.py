from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from plybench.player.learnable import LEARNER_CALLS

from plyagents.textrl.player import UPDATE_FILE

# the update of a batch (batch=every) or of the epoch (batch=epoch), next to the games' folders in a learner folder
_BATCH = re.compile(r"^batch_(\d+)$")


@dataclass(frozen=True)
class GameLearning:
    # where and how one training game was learned from, read from the learner folder of its epoch:
    #   record     the critic's assessments, the feedback and the own losses of the game (TextRLUpdate.record, or its entry in the batch's record)
    #   reference  the game's reference record with the critic's numbers and the grades
    #   calls      the LLM calls of the file that holds the game's learning, and game: the round its calls are about there (None: every call is)
    #   writes     the update whose actor and critic parts hold the memory writes the game took part in: the game's own; for a batch, the batch's,
    #              given with the batch's last game only (None for the others)
    #   batch      the batch's folder name (batch_<i>, epoch), None for a game learned from alone
    record: dict[str, Any]
    reference: dict[str, Any] | None
    calls: list[dict[str, Any]] = field(default_factory=list)
    game: int | None = None
    writes: dict[str, Any] | None = None
    batch: str | None = None

    def game_calls(self) -> list[dict[str, Any]]:
        # the calls about this game: all of its own file, or those of the batch's about it (the blocks' backward and writes are the batch's, not a game's)
        return [call for call in self.calls if self.game is None or (call.get("about") or {}).get("game") == self.game]

    def verdicts(self) -> list[dict[str, Any]]:
        # the verdict of every step for the blocks
        return (self.record.get("feedback") or {}).get("verdicts", [])

    def calls_of(self, role: str, step: int | None = None) -> list[int]:
        # the indices (in calls) of the usable calls of a role about this game, and about a step of it when given
        return [
            index
            for index, call in enumerate(self.calls)
            if call["role"] == role
            and call.get("failure") is None
            and (self.game is None or (call.get("about") or {}).get("game") == self.game)
            and (step is None or step in (call.get("about") or {}).get("steps", []))
        ]


def read_json(path: Path) -> Any:
    # a record file, None when it was not written
    return json.loads(path.read_text()) if path.exists() else None


def _batches(learner: Path) -> list[Path]:
    # the batch folders in the order they were learned: batch_1, batch_2, ..., then the epoch's
    numbered = sorted((path for path in learner.iterdir() if _BATCH.match(path.name)), key=lambda path: int(path.name.removeprefix("batch_")))
    return [*numbered, learner / "epoch"]


def game_learning(learner: Path, round_: int) -> GameLearning | None:
    # how training game round_ of the epoch whose learner folder this is was learned from; None when nothing was recorded for it
    game = learner / f"game_{round_}"
    update = read_json(game / UPDATE_FILE)
    if not isinstance(update, dict):
        return None
    if update.get("record") is not None:
        return GameLearning(update["record"], update.get("reference"), read_json(game / LEARNER_CALLS) or [], None, update)
    # a game of a batch: its own update holds its counts only; its judgement and the writes are in the batch's update
    for folder in _batches(learner) if learner.is_dir() else []:
        batch = read_json(folder / UPDATE_FILE)
        games = ((batch or {}).get("record") or {}).get("games") or []
        entry = next((item for item in games if item.get("round") == round_), None)
        if entry is None:
            continue
        record = {name: value for name, value in entry.items() if name not in ("round", "reference")}
        last = games[-1].get("round") == round_
        return GameLearning(record, entry.get("reference"), read_json(folder / LEARNER_CALLS) or [], round_, batch if last else None, folder.name)
    return GameLearning({}, update.get("reference"))

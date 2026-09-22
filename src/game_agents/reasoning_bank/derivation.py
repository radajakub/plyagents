from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class DerivationCache:
    """Per-epoch cache of what a finished game taught, keyed by the round it was derived from.

    The training harness resumes an interrupted epoch by replaying its already-recorded games through
    `_observe`, and judging plus induction are model samples: recomputing them would land the epoch on
    a different checkpoint than an uninterrupted run. The epoch's `workdir` survives the interruption
    (the checkpoint it is still producing by definition does not), so caching there both restores that
    equality and spares the rerun from paying for the derivation twice.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._dir: Path | None = None

    def bind(self, workdir: Path) -> None:
        self._dir = workdir / self._name
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, game_round: int) -> Path | None:
        return None if self._dir is None else self._dir / f"game_{game_round}.json"

    def load(self, game_round: int) -> dict[str, Any] | None:
        path = self._path(game_round)
        if path is None or not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def store(self, game_round: int, payload: dict[str, Any]) -> None:
        path = self._path(game_round)
        if path is None:
            return
        # stage and rename: a crash mid-write must not leave a half-written derivation to be replayed
        staged = path.with_suffix(".tmp")
        with open(staged, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(staged, path)

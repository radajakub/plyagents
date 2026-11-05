from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Assessment:
    # the critic's view of one visited state: a number when it has one, and the text the feedback model reads
    value: float | None
    text: str

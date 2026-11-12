from __future__ import annotations

from typing import Any

from plyagents.tac.apply.base import Apply, Shortened
from plyagents.tac.utils.calls import Calls


class TruncateApply(Apply):
    # no LLM: keeps the last `keep` characters of the old entry, the most recent lessons, cut wherever that falls
    async def shorten(self, old: str, keep: int, calls: Calls, role: str, about: dict[str, Any] | None = None) -> Shortened:
        # cutting the old entry is the operation itself, not a failure, so it is not flagged
        return Shortened(old[len(old) - keep :], False)

    def describe(self) -> str:
        return "truncate: keeps the most recent (1 - alpha) B characters of the entry, then appends the lesson (no LLM)"

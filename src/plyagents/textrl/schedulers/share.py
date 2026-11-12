from __future__ import annotations

from plyagents.textrl.enums import ShareRule
from plyagents.textrl.utils.options import read_options


def read_share(options: dict[str, str]) -> tuple[ShareRule, float]:
    # the scheduler options of a memory write: per (write, sample) and cap (at most 1)
    per = ShareRule.from_value(options.get("per", ShareRule.WRITE.value))
    if per is None:
        raise ValueError(f"Invalid scheduler per {options['per']!r}; choose from {ShareRule.values()}")
    cap = float(options.get("cap", 1.0))
    if not 0 < cap <= 1:
        raise ValueError(f"Invalid scheduler cap {cap}; it must be in (0, 1]")
    return per, cap


def share_options(params_string: str, extra: set[str], what: str) -> dict[str, str]:
    return read_options(params_string, extra | {"per", "cap"}, what)

from __future__ import annotations

import json
import re
from importlib.resources import files
from typing import Any

# the named TextRL configs: one JSON file per name in plyagents/textrl/presets, versioned with the code, holding a spec in the TextRL's own format (its
# roles and optionally model=), a one-line description, and what defines the algorithm: per role the type and the params it must keep (`defines`), so
# an override that changes them is refused when the player is built (build.py)
_NAME = re.compile(r"^[a-z0-9_]+$")


def preset_names() -> list[str]:
    return sorted(path.name.removesuffix(".json") for path in files("plyagents.textrl.presets").iterdir() if path.name.endswith(".json"))


def _preset(name: str) -> dict[str, Any]:
    if not _NAME.match(name) or name not in preset_names():
        raise ValueError(f"Unknown TextRL preset {name!r}; choose from {preset_names()}")
    return json.loads(files("plyagents.textrl.presets").joinpath(f"{name}.json").read_text())


def preset_spec(name: str) -> str:
    return _preset(name)["spec"]


def preset_defines(name: str) -> dict[str, str]:
    # by role, the component config that defines the algorithm: its type, and the params it names (the others are open)
    return _preset(name)["defines"]

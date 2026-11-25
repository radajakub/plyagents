"""Write the TextRL catalogue as JSON: every role, component type and option with what it does, the sentences of the prompts, the prompts of the LLM
critics and losses rendered on an example situation (per option value), and the presets (src/plyagents/textrl/catalogue.py). plyweb's algorithms
page reads it from its src/components.json; run this after a change of a component or an option, and commit the file in plyweb. No LLM is called.

Example:
    uv run python scripts/export_catalogue.py --out ../plyweb/src/components.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from plyagents.textrl.catalogue import catalogue, missing_texts

ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="the JSON file to write")
    args = parser.parse_args()
    missing = missing_texts()
    if missing:
        raise SystemExit(f"options without a text in catalogue.py: {', '.join(missing)}")
    # the commit the catalogue was made from; dirty: the working tree had changes, so the commit alone does not reproduce it
    source = {"repository": "plyagents", "commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain"))}
    args.out.write_text(json.dumps({"source": source, **catalogue()}, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {args.out} (plyagents {source['commit'][:7]}{', dirty' if source['dirty'] else ''})")


if __name__ == "__main__":
    main()

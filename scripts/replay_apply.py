"""Replay the recorded writes of TextRL training runs through another Apply: the same gradients (the parts each write took), in the same order, into
an empty memory, with another Apply, budget or step. Checks offline how a write rule keeps the memory over a whole run (what form the entries take,
whether a fact changes), without training again.

A run is a replicate folder of a training result (replicate_<n>, with epoch_<k>/learner/game_<g>/update.json). Every write of the chosen block
(actor or critic) is taken in the order of the epochs and games, and written with the recorded step alpha, unless --alpha sets one. Each run's
writes are saved in <out>/<run>.json after every write (the gradients, the entry before and after, the realised step, the call, and the
reference beside it: the counts of the game's key before it, the move, the outcome and Q*), and a run continues from the writes saved, so a stopped
run continues. A plyservice scripts job: --check --json prints the plan; at the end, per run, the entry at a few points is printed with the counts.

Examples:
    uv run python scripts/replay_apply.py --results results/training/tac_bandit_two_arm --players preset-tac --apply "rewrite[model=gemini:...]" \
        --budget 500 --out analysis/tac_bandit_two_arm/replay_apply/rewrite_500 --check
    uv run python scripts/replay_apply.py --results results/training/tac_bandit_two_arm --players preset-tac --apply "rewrite[model=gemini:...]" \
        --budget 500 --out analysis/tac_bandit_two_arm/replay_apply/rewrite_500
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from plybench.cli.shared import build_op
from plybench.trackers.llm_call import LLMCall
from plybench.utils.seed import derive_seed

from plyagents.textrl.apply.base import Apply
from plyagents.textrl.apply.build import APPLIES
from plyagents.textrl.configs.components import Services
from plyagents.textrl.utils.calls import Calls

# the writes after which the summary prints the entry of each key
SHOWN = (10, 25, 50, 100)


@dataclass(frozen=True)
class Write:
    # one recorded write of the block and the game it came from
    epoch: int
    game: int
    key: str
    parts: tuple[str, ...]
    alpha: float
    steps: tuple[int, ...]
    reference: dict[str, Any]


def _number(path: Path) -> int:
    return int(re.sub(r"\D", "", path.name))


def find_runs(results: Path, players: str | None) -> list[Path]:
    # every replicate folder of a training result whose player folder contains `players`, in a fixed order
    runs = sorted(path for path in results.glob("*/*/replicate_*") if path.is_dir() and (players is None or players in path.parent.name))
    return [run for run in runs if any(run.glob("epoch_*/learner/game_*/update.json"))]


def run_name(results: Path, run: Path) -> str:
    return "__".join(run.relative_to(results).parts)


def recorded_writes(run: Path, block: str) -> list[Write]:
    # the block's writes in the order they were made: by epoch, then game, then as recorded within the game
    writes = []
    for epoch in sorted(run.glob("epoch_*"), key=_number):
        for game in sorted(epoch.glob("learner/game_*"), key=_number):
            update = json.loads((game / "update.json").read_text())
            steps = {step["key"]: step for step in update["reference"]["steps"]}
            for write in (update.get(block) or {}).get("writes", []):
                step = steps.get(write["key"].split("|")[0], {})
                reference = {"counts": step.get("counts"), "move": step.get("move"), "solved": step.get("solved"), "outcome": update["reference"]["outcome"]}
                writes.append(Write(_number(epoch), _number(game), write["key"], tuple(write["parts"]), write["alpha"], tuple(write["steps"]), reference))
    return writes


async def replay(apply: Apply, run: str, writes: list[Write], budget: int | None, alpha: float | None, out: Path, seed: int, llm: Any) -> None:
    # the writes one after another into an empty memory; saved after each, and continued from what is saved
    path = out / f"{run}.json"
    done: list[dict[str, Any]] = json.loads(path.read_text())["writes"] if path.exists() else []
    entries: dict[str, str] = {}
    for saved in done:
        entries[saved["key"]] = saved["new"]
    for index in range(len(done), len(writes)):
        write = writes[index]
        recorded: list[LLMCall] = []
        calls = Calls(llm, derive_seed(seed, run, index), recorded.append)
        step = alpha if alpha is not None else write.alpha
        old = entries.get(write.key, "")
        written = await apply.merge(write.key, old, write.parts, step, budget, calls, "apply", write.steps, sum(1 for saved in done if saved["key"] == write.key))
        entries[write.key] = written.new
        done.append(
            {**asdict(write), "old": old, "new": written.new, "alpha": step, "changed": written.changed, "cut": written.cut, "calls": [call.to_dict() for call in recorded]}
        )
        path.write_text(json.dumps({"run": run, "budget": budget, "writes": done}, indent=1))
        print(f"progress: {run} {index + 1}/{len(writes)}", flush=True)


def _counts_text(reference: dict[str, Any]) -> str:
    counts = reference.get("counts") or {}
    return "; ".join(f"{move} {value['results']}" for move, value in counts.items()) or "none"


def summary(out: Path, run: str) -> None:
    # per key, the entry after a few writes, beside the counts before that game and Q*; the step and the cuts over the run
    saved = json.loads((out / f"{run}.json").read_text())["writes"]
    print(f"\n== {run}: {len(saved)} writes, mean realised step {sum(write['changed'] for write in saved) / max(len(saved), 1):.2f}, cut {sum(write['cut'] for write in saved)}")
    by_key: dict[str, list[dict[str, Any]]] = {}
    for write in saved:
        by_key.setdefault(write["key"], []).append(write)
    for key, writes in by_key.items():
        print(f"-- {key}")
        for shown in sorted({number for number in SHOWN if number <= len(writes)} | {len(writes)}):
            write = writes[shown - 1]
            solved = (write["reference"].get("solved") or {}).get("action_values")
            print(f"after write {shown} (epoch {write['epoch']}, game {write['game']}); counts before it: {_counts_text(write['reference'])}; Q* {solved}")
            print(f"   {write['new']!r}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="a training result folder (relative to the repository: results/training/<name>)")
    parser.add_argument("--players", help="only the runs whose player folder contains this text (e.g. preset-tac)")
    parser.add_argument("--block", choices=("actor", "critic"), default="actor", help="whose writes to replay (default actor)")
    parser.add_argument("--apply", required=True, help="the Apply to write with, as in a config: rewrite[model=...], compress[model=...], truncate")
    parser.add_argument("--budget", type=int, help="the entry budget B in characters (default: none, the lessons are only appended)")
    parser.add_argument("--alpha", type=float, help="the step of every write (default: the recorded one)")
    parser.add_argument("--seed", type=int, default=0, help="the base seed of the calls (default 0)")
    parser.add_argument("--out", type=Path, required=True, help="folder of the replayed runs, one file each (relative to the repository: analysis/...)")
    parser.add_argument("--concurrency", type=int, default=10, help="requests in flight per provider (default 10)")
    parser.add_argument("--check", action="store_true", help="print the plan (runs, writes, the model, the output), then exit; no calls")
    parser.add_argument("--json", action="store_true", help="with --check: print the plan as JSON")
    args = parser.parse_args()

    op = build_op(concurrency=args.concurrency)
    apply_config = APPLIES.config(args.apply)
    apply = APPLIES.build(apply_config, Services(op.llm, lambda game: False))
    runs = {run_name(args.results, run): recorded_writes(run, args.block) for run in find_runs(args.results, args.players)}
    planned = sum(len(writes) for writes in runs.values())
    left = sum(
        len(writes) - len(json.loads((args.out / f"{name}.json").read_text())["writes"]) if (args.out / f"{name}.json").exists() else len(writes) for name, writes in runs.items()
    )

    if args.check:
        problems = [] if runs else [f"no runs with learner updates under {args.results}" + (f" for players {args.players!r}" if args.players else "")]
        model = getattr(apply_config.params, "model", None)
        plan = {"planned": planned, "remaining": left, "unit": "writes", "models": [model.to_string()] if model is not None else [], "outputs": [str(args.out)], "runs": list(runs)}
        print(
            json.dumps(plan | {"problems": problems})
            if args.json
            else f"{left} of {planned} writes to replay, {len(runs)} runs, {apply.describe()}, into {args.out}; problems: {problems or 'none'}"
        )
        raise SystemExit(1 if problems else 0)

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(runs)} runs, {left} of {planned} writes to replay with {apply.describe()}, budget {args.budget}", flush=True)
    await asyncio.gather(*(replay(apply, name, writes, args.budget, args.alpha, args.out, args.seed, op.llm) for name, writes in runs.items()))
    for name in runs:
        summary(args.out, name)


if __name__ == "__main__":
    asyncio.run(main())

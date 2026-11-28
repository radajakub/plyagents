"""Part A of the exploration experiment: sample the TextRL actor many times in one fixed state, with a fixed memory and an exploration level, and compare
the distributions of its actions.

A cell is a state (a game and the OpenSpiel actions from its start, chance included), a memory (a checkpoint of the TextRL player, or none) and a level
(the actor's options, e.g. `,explore=memory` or `,counts=request`, put into the config's actor spec; or {"options": "...", "instruction": "..."}, which
also shows a free instruction line after the notes, in training; with "info": {"line": "...", "first": "..."} an information line follows it). Every cell is sampled with the seeds 0..N-1,
the same seeds in every cell, in training mode (the mode the instruction and the counts are shown in); a sample is one move of the player, as in a
game seeded with that seed. The config (JSON):

    {"samples": 100, "baseline": "I0", "actor": "llm[model=...{options}]", "levels": {"I0": "", "I1": ",explore=generic"},
     "states": {"<state>": {"game": "bandit:...", "actions": [], "memories": {"M0": {"checkpoint": null, "levels": ["I0", "I1"]}}}}}

With "replicates": R in the config (default 1), every cell is sampled R times over, each replicate with its own seeds and in its own folder. Sample k
of a replicate is game k + 1 of N. The information line is a template with {game}, {games} (N), {counts} (counts_text of the earlier presses of the
replicate, no outcomes) and {history} (the earlier presses in order); "first" replaces it in game 1. A level whose line uses {counts} or {history} asks
each replicate in order (the replicates in parallel), reading the earlier presses from the saved samples; a failed sample stops its replicate until the
next run.

Each sample is saved as its own file, <out>/<state>_<memory>_<level>/sample_<k>.json (replicate_<r>/sample_<k>.json with replicates) (the action, the context and the guidance shown, the prompt,
the reasoning and the output), and a saved sample is not asked again, so a stopped run continues. A plyservice scripts job: --check --json prints
the plan and its problems (a memory without an entry for the state's key, counts shown from a checkpoint without counts); --job-report <path> keeps a
report (plybench.report) of the cells with samples left when it starts while it works, and the tables printed at the end are the same: per cell
the share of each action with a 95% Wilson interval and the entropy, and against the cell of the same state and memory at the baseline level the total variation distance (95% bootstrap interval) and
a permutation p-value. The bootstrap interval of a distance is biased away from 0, so whether a level changes the distribution is read from the
permutation test, and the distance is its size.

Examples:
    uv run python scripts/sample_actor.py --config experiments/exploration/part_a.json --out analysis/tac_exploration/part_a --check
    uv run python scripts/sample_actor.py --config experiments/exploration/part_a.json --out analysis/tac_exploration/part_a --dry-run
    uv run python scripts/sample_actor.py --config experiments/exploration/part_a.json --out analysis/tac_exploration/part_a
    uv run python scripts/sample_actor.py --config experiments/exploration/part_a.json --out analysis/tac_exploration/part_a --job-report <path>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from plybench.app import PlyBench
from plybench.cli.shared import build_op
from plybench.common.enums import Mode
from plybench.core.engine import TurnBasedEngine
from plybench.core.interface import InterfaceAction, InterfaceObservation
from plybench.report import Item, Report, ReportFile, estimate
from plybench.trackers.step_data import StepData
from plybench.utils.seed import derive_seed
from plyllm import LLM, LLMResponse, LLMTokens
from plyllm.response import OutputText

from plyagents.textrl.actors.llm import LLMActor, counts_text
from plyagents.textrl.model.counts import MoveCounts
from plyagents.textrl.player import TextRLPlayer
from plyagents.textrl.register import register_players

# the name of a failed move in the tables
FAIL = "fail"
# resamples of the bootstrap interval and of the permutation test; their generator is seeded, so the tables are the same on every run
RESAMPLES = 2000
# seconds between two writes of the report while sampling (each costs the bootstrap of every cell)
REPORT_INTERVAL = 30.0


@dataclass(frozen=True)
class Cell:
    state: str
    memory: str
    level: str
    game: str
    actions: tuple[int, ...]
    spec: str
    checkpoint: Path | None
    instruction: str | None = None
    # the information line after the instruction and its text in game 1 (None: the line)
    info: tuple[str, str | None] | None = None

    @property
    def name(self) -> str:
        return f"{self.state}_{self.memory}_{self.level}"

    @property
    def sequential(self) -> bool:
        # the line shows the earlier presses, so a sample waits for the one before it
        return self.info is not None and any(f"{{{name}}}" in text for text in self.info if text is not None for name in ("counts", "history"))


def read_cells(config: dict[str, Any]) -> list[Cell]:
    # every state with each of its memories at each of that memory's levels, in the order of the config
    cells = []
    for state, spec in config["states"].items():
        for memory, setup in spec["memories"].items():
            checkpoint = Path(setup["checkpoint"]) if setup.get("checkpoint") else None
            for level in setup["levels"]:
                value = config["levels"][level]
                options, instruction = (value, None) if isinstance(value, str) else (value.get("options", ""), value["instruction"])
                info = value.get("info") if isinstance(value, dict) else None
                actor = config["actor"].replace("{options}", options)
                player = f"textrl:actor={actor}" + (f",checkpoint={checkpoint}" if checkpoint is not None else "")
                line = (info["line"], info.get("first")) if info is not None else None
                cells.append(Cell(state, memory, level, spec["game"], tuple(spec["actions"]), player, checkpoint, instruction, line))
    return cells


@dataclass(frozen=True)
class Draw:
    # one sample to ask: sample k (game k + 1) of a replicate, out of `samples` per replicate; `replicates` > 1 gives each replicate a folder
    replicate: int
    sample: int
    samples: int
    replicates: int = 1

    @property
    def game(self) -> int:
        return self.sample + 1

    @property
    def seed_index(self) -> int:
        # replicate 0 has the seeds 0..N-1 of a run without replicates, every later replicate the next N
        return self.replicate * self.samples + self.sample

    def at(self, sample: int) -> Draw:
        return Draw(self.replicate, sample, self.samples, self.replicates)


def sample_path(out: Path, cell: Cell, draw: Draw) -> Path:
    folder = out / cell.name / (f"replicate_{draw.replicate}" if draw.replicates > 1 else "")
    return folder / f"sample_{draw.sample}.json"


def earlier_presses(out: Path, cell: Cell, draw: Draw) -> list[str]:
    # the actions of the saved samples before this one in its replicate, in order; a failed move (no action) is left out, as no button was pressed
    if not cell.sequential:
        return []
    actions = [json.loads(sample_path(out, cell, draw.at(k)).read_text())["action"] for k in range(draw.sample)]
    return [action for action in actions if action is not None]


def info_line(cell: Cell, draw: Draw, earlier: list[str], legal: list[str]) -> str | None:
    # the cell's information line for this sample: its game and the earlier presses of the replicate (as counts, without outcomes: stats off)
    if cell.info is None:
        return None
    line, first = cell.info
    counts = {action: MoveCounts({"draw": count}) for action, count in Counter(earlier).items()}
    template = first if draw.game == 1 and first is not None else line
    return template.format(game=draw.game, games=draw.samples, counts=counts_text(legal, counts, False), history=", ".join(earlier))


def situation(
    op: PlyBench, cell: Cell, draw: Draw, earlier: list[str] | None = None
) -> tuple[TurnBasedEngine, TextRLPlayer, InterfaceObservation, list[InterfaceAction], str | None]:
    # the cell's state and a fresh player, as a game seeded with the draw's seed index would have them at this move: seeded, in training mode, its memory restored;
    # the instruction followed by the information line, built from the earlier presses of the replicate
    engine = op.registry.build_engine(op.registry.game_config(cell.game))
    engine.reset()
    for action in cell.actions:
        engine.game.state.apply_action(action)
    player = op.registry.build_player(engine.game, op.registry.player_config(cell.spec), "i")
    if not isinstance(player, TextRLPlayer):
        raise SystemExit(f"{cell.name}: {cell.spec} is not a TextRL player")
    if cell.instruction is not None and not isinstance(player.actor, LLMActor):
        raise SystemExit(f"{cell.name}: an instruction needs the LLM actor")
    player.seed(derive_seed(draw.seed_index, player.identifier))
    player.set_mode(Mode.TRAIN)
    player.initialize_policy(engine.game, engine.prompt_adapter)
    pid = engine.game.get_player()
    observation = engine.observation_class.from_openspiel(engine.game.get_observation(pid), engine.interface_transformer)
    legal = [engine.action_class.from_openspiel(move, engine.interface_transformer) for move in engine.game.get_legal_moves(pid)]
    info = info_line(cell, draw, earlier or [], [move.to_llm().string for move in legal])
    if cell.instruction is not None and isinstance(player.actor, LLMActor):
        player.actor.instruction = cell.instruction + (f"\n{info}" if info is not None else "")
    return engine, player, observation, legal, info


async def sample_once(op: PlyBench, cell: Cell, draw: Draw, earlier: list[str] | None = None) -> dict[str, Any]:
    engine, player, observation, legal, info = situation(op, cell, draw, earlier)
    output = await player(engine.game, observation, legal)
    record = StepData.from_output(output, **(output.data or {})).to_dict()
    action = output.action.to_llm().string if output.action is not None else None
    return {
        "cell": cell.name,
        "sample": draw.sample,
        "replicate": draw.replicate,
        "spec": cell.spec,
        "instruction": cell.instruction,
        "game": draw.game,
        "info": info,
        "legal": [move.to_llm().string for move in legal],
        "action": action,
        "step": record,
    }


def problems_of(op: PlyBench, cell: Cell) -> list[str]:
    # what would make the cell measure something else than it says: a memory that does not reach the state, or counts that are not there
    if cell.checkpoint is not None and not (cell.checkpoint / "state.json").exists():
        return [f"{cell.name}: no checkpoint at {cell.checkpoint}"]
    try:
        _, player, observation, _, _ = situation(op, cell, Draw(0, 0, 1))
    except (ValueError, TypeError) as error:
        return [f"{cell.name}: {type(error).__name__}: {error}"]
    key, problems = player.key(observation), []
    actor = player.actor
    if not isinstance(actor, LLMActor):
        return [f"{cell.name}: the actor is not the LLM actor"]
    if cell.checkpoint is not None and not actor.memory.get(key).text:
        problems.append(f"{cell.name}: the checkpoint has no entry for the state's key {key!r}")
    if cell.checkpoint is not None and actor.counts.value != "off" and not player.reference.before(key):
        problems.append(f"{cell.name}: counts are shown, but the checkpoint holds no counts for the state's key (made before the reference record)")
    return problems


class _EchoLLM:
    # --dry-run: keeps the prompt and plays the first legal move, so the prompts can be read without a call; no configured provider, so building the
    # player does not validate the model against it
    available_providers: list = []

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, model_config, system, messages, output_schema=None, seed=None):
        prompt = messages[-1].content
        self.prompts.append(prompt)
        text = "Action: " + re.findall(r"<[^>]+>", prompt.split("legal moves are:")[1])[0]
        return LLMResponse(provider=model_config.provider, model_string="echo", tokens=LLMTokens(input_tokens=0, output_tokens=0), items=[OutputText([text])], output_text=text)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    share = successes / n
    centre = (share + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(share * (1 - share) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def entropy(counts: Counter[str]) -> float:
    # in bits
    n = sum(counts.values())
    return -sum(count / n * math.log2(count / n) for count in counts.values() if count) if n else 0.0


def distance(a: np.ndarray, b: np.ndarray, labels: int) -> float:
    # total variation distance of two samples of label indices
    return 0.5 * float(np.abs(np.bincount(a, minlength=labels) / len(a) - np.bincount(b, minlength=labels) / len(b)).sum())


def compare(actions: list[str], baseline: list[str]) -> dict[str, float]:
    # the distance from the baseline cell, a percentile bootstrap interval (each cell resampled on its own) and a permutation p-value (the labels of
    # the two cells shuffled together: how often a shuffle is at least as far apart)
    labels = sorted(set(actions) | set(baseline))
    a = np.array([labels.index(action) for action in actions])
    b = np.array([labels.index(action) for action in baseline])
    rng = np.random.default_rng(0)
    observed = distance(a, b, len(labels))
    boots = [distance(rng.choice(a, len(a)), rng.choice(b, len(b)), len(labels)) for _ in range(RESAMPLES)]
    pooled = np.concatenate([a, b])
    shuffled = (rng.permutation(pooled) for _ in range(RESAMPLES))
    extreme = sum(distance(perm[: len(a)], perm[len(a) :], len(labels)) >= observed - 1e-12 for perm in shuffled)
    low, high = np.percentile(boots, [2.5, 97.5])
    return {"tv": observed, "tv_low": float(low), "tv_high": float(high), "p_permutation": (extreme + 1) / (RESAMPLES + 1)}


def status(cells: list[Cell], out: Path, baseline: str) -> dict[str, Any]:
    saved: dict[str, list[str]] = {}
    for cell in cells:
        files = sorted((out / cell.name).glob("**/sample_*.json"))
        saved[cell.name] = [json.loads(path.read_text())["action"] or FAIL for path in files]
    tables = []
    for cell in cells:
        actions = saved[cell.name]
        counts = Counter(actions)
        row: dict[str, Any] = {"cell": cell.name, "state": cell.state, "memory": cell.memory, "level": cell.level, "samples": len(actions)}
        row["shares"] = {
            action: {"share": count / len(actions), "low": wilson(count, len(actions))[0], "high": wilson(count, len(actions))[1]} for action, count in counts.most_common()
        }
        row["entropy_bits"] = entropy(counts)
        base = next((other for other in cells if (other.state, other.memory, other.level) == (cell.state, cell.memory, baseline)), None)
        if base is not None and base is not cell and actions and saved[base.name]:
            row["against"] = base.name
            row |= compare(actions, saved[base.name])
        tables.append(row)
    return {"cells": tables}


def report(cells: list[Cell], out: Path, baseline: str, planned: int) -> Report:
    # the report of the cells this command samples: one item per cell, a table of the saved samples per cell (the status of every cell, as a cell's
    # baseline may be outside them)
    rows = {row["cell"]: row for row in status(cells, out, baseline)["cells"]}
    result = Report(unit="samples", items=[Item(cell.name, {"cell": cell.name}, planned, rows[cell.name]["samples"]) for cell in cells])
    table = []
    for cell in cells:
        row = rows[cell.name]
        shares = ", ".join(f"{action} {value['share']:.2f} [{value['low']:.2f}, {value['high']:.2f}]" for action, value in row["shares"].items())
        line: dict[str, Any] = {"item": cell.name, "state": cell.state, "memory": cell.memory, "level": cell.level, "samples": row["samples"]}
        line |= {"entropy": row["entropy_bits"], "shares": shares, "against": row.get("against")}
        line |= {"tv": estimate(row["tv"], row["tv_low"], row["tv_high"]) if "tv" in row else None, "p": row.get("p_permutation")}
        table.append(line)
    columns = [{"key": "item", "label": "cell", "name": True}] + [{"key": key, "label": key} for key in ("state", "memory", "level")]
    columns += [{"key": "samples", "label": "samples", "digits": 0}, {"key": "entropy", "label": "entropy (bits)", "digits": 2}, {"key": "shares", "label": "actions"}]
    columns += [{"key": "against", "label": "baseline"}, {"key": "tv", "label": "TV distance", "rate": True}, {"key": "p", "label": "permutation p", "digits": 4}]
    result.add_table(
        "actions per cell",
        columns,
        table,
        note="shares with 95% Wilson intervals; TV distance from the cell of the same state and memory at the baseline level, with a 95% bootstrap interval "
        "(biased away from 0: read whether a level changes the distribution from the permutation p)",
    )
    return result


def print_tables(tables: dict[str, Any]) -> None:
    for row in tables["cells"]:
        shares = ", ".join(f"{action} {value['share']:.2f} [{value['low']:.2f}, {value['high']:.2f}]" for action, value in row["shares"].items())
        against = f"; TV vs {row['against']} {row['tv']:.2f} [{row['tv_low']:.2f}, {row['tv_high']:.2f}], permutation p {row['p_permutation']:.4f}" if "against" in row else ""
        print(f"{row['cell']}: n {row['samples']}, entropy {row['entropy_bits']:.2f} bits{against}\n   {shares}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="the cells (JSON, see above)")
    parser.add_argument("--out", type=Path, required=True, help="folder of the samples, one file each (relative to the repository: analysis/...)")
    parser.add_argument("--cells", nargs="+", help="only these cells, by name (default: all)")
    parser.add_argument("--concurrency", type=int, default=10, help="requests in flight per provider (default 10)")
    parser.add_argument("--check", action="store_true", help="print the plan (samples planned and left, the models, the output) and its problems, then exit; no calls")
    parser.add_argument("--json", action="store_true", help="with --check: print the plan as JSON")
    parser.add_argument("--job-report", type=Path, help="keep a report (plybench.report) of the cells with samples left while it works, in this JSON file")
    parser.add_argument("--dry-run", action="store_true", help="print the prompt of each cell's first sample, no calls")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    cells = [cell for cell in read_cells(config) if args.cells is None or cell.name in args.cells]

    op = build_op(concurrency=args.concurrency)
    samples, replicates = int(config["samples"]), int(config.get("replicates", 1))
    planned = [(cell, Draw(r, k, samples, replicates)) for cell in cells for r in range(replicates) for k in range(samples)]
    left = [(cell, draw) for cell, draw in planned if not sample_path(args.out, cell, draw).exists()]

    if args.check:
        problems = [problem for cell in cells for problem in problems_of(op, cell)]
        models = sorted({model.to_string() for cell in cells for model in op.registry.player_config(cell.spec).params.models()})
        plan_json = {"planned": len(planned), "remaining": len(left), "unit": "samples", "models": models, "outputs": [str(args.out)]}
        print(
            json.dumps(plan_json | {"problems": problems})
            if args.json
            else f"{len(left)} of {len(planned)} samples to ask, {len(cells)} cells, models {models}, into {args.out}; problems: {problems or 'none'}"
        )
        raise SystemExit(1 if problems else 0)

    if args.dry_run:
        echo = _EchoLLM()
        register_players(op.registry, cast(LLM, echo))
        for cell in cells:
            # a sequential cell: games 1 to 3 of the first replicate, the earlier presses those of the echo, and the prompt of game 3
            earlier: list[str] = []
            for sample in range(3 if cell.sequential else 1):
                action = (await sample_once(op, cell, Draw(0, sample, samples, replicates), earlier))["action"]
                earlier.append(action)
            print(f"\n== {cell.name}\n{echo.prompts[-1]}")
        return

    print(f"{len(cells)} cells, {len(left)} of {len(planned)} samples to ask", flush=True)
    done, failed = len(planned) - len(left), 0
    # the report's items: the cells with samples left now; its tables cost a bootstrap per cell, so it is written at most every REPORT_INTERVAL seconds
    open_cells = [cell for cell in cells if any(other is cell for other, _ in left)]
    report_file = (
        ReportFile(args.job_report, lambda: report(open_cells, args.out, config["baseline"], samples * replicates), REPORT_INTERVAL) if args.job_report is not None else None
    )
    if report_file is not None:
        report_file.write()

    async def one(cell: Cell, draw: Draw) -> bool:
        nonlocal done, failed
        try:
            saved = await sample_once(op, cell, draw, earlier_presses(args.out, cell, draw))
        except Exception as error:
            # left unsaved, so a run again asks it again; the run ends with an error once the others are done
            failed += 1
            print(f"warning: {cell.name}, replicate {draw.replicate}, sample {draw.sample}: {type(error).__name__}: {error}", flush=True)
            return False
        path = sample_path(args.out, cell, draw)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(saved, indent=1))
        done += 1
        print(f"progress: {done}/{len(planned)}", flush=True)
        if report_file is not None:
            report_file.changed()
        return True

    async def in_order(cell: Cell, draws: list[Draw]) -> None:
        # the samples of one replicate in order; a failure stops it, as the later games would see a history with a gap
        for draw in draws:
            if not await one(cell, draw):
                return

    chains: dict[tuple[Cell, int], list[Draw]] = {}
    for cell, draw in left:
        if cell.sequential:
            chains.setdefault((cell, draw.replicate), []).append(draw)
    tasks = [one(cell, draw) for cell, draw in left if not cell.sequential]
    tasks += [in_order(cell, draws) for (cell, _), draws in chains.items()]

    completed = False
    try:
        await asyncio.gather(*tasks)
        completed = True
    finally:
        if report_file is not None:
            report_file.close(finished=completed and failed == 0)
    print_tables(status(cells, args.out, config["baseline"]))
    if failed:
        raise SystemExit(f"{failed} samples failed; run the same command again to ask them")


if __name__ == "__main__":
    asyncio.run(main())

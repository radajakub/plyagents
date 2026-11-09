"""Ask the LLM feedback model again about the training games of a finished TAC run, with more information than it had, and grade its answers.

Offline: no game is played. Each training game's recorded feedback call is sent again, changed by the variant, and the new grades are compared
with the solver's verdict on each move (as the feedback section of show_tac.py does for the recorded grades). Calls the model: costs money.

Variants:
    as_run   the recorded call unchanged: the noise of the grades, the baseline for the others
    caution  the system message also says the rules may leave parts of the game unstated, so a known puzzle's answer may not hold
    history  the prompt also shows, for each situation of the game, the moves the agent played there in the run's earlier training games
             and their outcomes: the evidence across games the feedback model never sees in a run

Each answer is saved as its own file, <out>/<variant>/epoch_<e>_game_<r>.json (the grades, the keys and the solver's verdicts, the call), and an
answer already saved is not asked again, so a stopped run continues. A plyservice scripts job (see its README): --check --json prints the plan,
each answer prints `progress: <done>/<total>`, --status prints the tables of the saved answers.

Examples:
    uv run python scripts/reask_feedback.py --experiment tac_monty_hall_ladder --trainees "<R3 config>" --out analysis/tac_monty_hall_ladder/reask_r3
    uv run python scripts/reask_feedback.py --experiment tac_monty_hall_ladder --trainees "<R3 config>" --out <dir> --variants history --epochs 1 2
    uv run python scripts/reask_feedback.py --experiment tac_monty_hall_ladder --trainees "<R3 config>" --out <dir> --variants caution history --dry-run
    uv run python scripts/reask_feedback.py --out analysis/tac_monty_hall_ladder/reask_r3 --status
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from plybench.analysis.replay import build_replayer
from plybench.cli.shared import add_training_source_args, build_op, training_from_args
from plybench.configs.training_run import TrainingRun
from plybench.harness.training.training import TrainingHarness
from plybench.player.learnable import LEARNER_CALLS
from plybench.trackers.llm_call import LLMCall
from plyllm import ModelConfig
from show_tac import grade_table

from plyagents.tac.feedback.llm import LLMFeedbackParams, _Verdicts
from plyagents.tac.model.feedback import GRADES
from plyagents.tac.params import TACParams
from plyagents.tac.utils.calls import Calls

VARIANTS = ("as_run", "caution", "history")
CAUTION = "\n\nThe rules may leave parts of the game unstated. Do not assume they work as in a puzzle or game you know; judge each move by what the games showed."
HISTORY = "What the agent's earlier games showed in these situations (the moves it played there and how those games ended):"
# the prompt's line after the moves; the history goes before it
OUTCOME = "\n\nOutcome for the agent:"


@dataclass
class Game:
    # one recorded training game: where it is, the agent's key and move and the solver's verdict per step, the outcome, the recorded call
    epoch: int
    round: int
    seed: int | None
    keys: list[str]
    moves: list[str]
    optimal: list[bool]
    result: str
    system: str
    prompt: str


def load_games(harness: TrainingHarness, run: TrainingRun, epochs: list[int] | None) -> list[Game]:
    # the run's training games in the order they were played, each with its last usable feedback call; a game without one is left out
    op, games = harness.op, []
    replayer = build_replayer(op.registry, run.game)
    for epoch in harness.get_results().for_run(run):
        tracker = epoch.train
        if tracker is None:
            continue
        for round_, record in enumerate(tracker.games, start=1):
            if record is None:
                continue
            path = harness.path_builder.learner_dir(harness.experiment, run, epoch.epoch.index) / f"game_{round_}" / LEARNER_CALLS
            calls = [call for call in json.loads(path.read_text()) if call["role"] == "feedback" and not call.get("failure")] if path.exists() else []
            if not calls:
                continue
            steps = replayer.replay_steps(record, tracker.i)
            call = calls[-1]
            games.append(
                Game(
                    epoch.epoch.index,
                    round_,
                    record.seed,
                    [(step.step.data or {}).get("key", step.step.observation) for step in steps],
                    [step.step.move for step in steps],
                    [step.is_optimal for step in steps],
                    record.get_result(tracker.i).name.lower(),
                    call["system_message"],
                    call["prompt_message"],
                )
            )
    return [game for game in games if epochs is None or game.epoch in epochs]


def history(game: Game, earlier: list[Game]) -> str:
    # per step of the game: the moves the agent played in the same situation in the earlier games, with their outcomes
    lines = []
    for number, key in enumerate(game.keys, start=1):
        seen: dict[str, Counter[str]] = defaultdict(Counter)
        for other in earlier:
            for other_key, move in zip(other.keys, other.moves):
                if other_key == key:
                    seen[move][other.result] += 1
        if not seen:
            lines.append(f"Step {number}: no earlier games in this situation.")
            continue
        parts = [f"{move} in {sum(counter.values())} games ({', '.join(f'{count} {result}' for result, count in counter.most_common())})" for move, counter in seen.items()]
        lines.append(f"Step {number}: " + "; ".join(parts) + ".")
    return f"\n\n{HISTORY}\n" + "\n".join(lines)


def changed(game: Game, earlier: list[Game], variant: str) -> tuple[str, str]:
    # the system message and the prompt of the call for the variant
    if variant == "caution":
        return game.system + CAUTION, game.prompt
    if variant == "history":
        if OUTCOME not in game.prompt:
            raise ValueError(f"epoch {game.epoch}, game {game.round}: the recorded prompt has no outcome line to put the history before")
        head, _, tail = game.prompt.partition(OUTCOME)
        return game.system, head + history(game, earlier) + OUTCOME + tail
    return game.system, game.prompt


def feedback_model(run: TrainingRun) -> ModelConfig:
    # the model of the run's LLM feedback, options included (the recorded calls name the model without them)
    params = TACParams.from_string(run.trainee.to_string().partition(":")[2]).feedback.params
    if not isinstance(params, LLMFeedbackParams) or params.model is None:
        raise SystemExit(f"the run's feedback names no LLM model: {run.trainee.to_string()}")
    return params.model.config


async def ask(calls: Calls, model: ModelConfig, game: Game, earlier: list[Game], variant: str) -> list[str]:
    # the new grade of every step; "none" for a step the answer leaves out
    system, prompt = changed(game, earlier, variant)
    answer = await calls.structured("feedback", model, system, prompt, _Verdicts)
    by_step: dict[int, str] = {}
    for verdict in answer.verdicts:
        by_step.setdefault(verdict.step, verdict.grade)
    return [by_step.get(number, "none") for number in range(1, len(game.keys) + 1)]


def answer_path(out: Path, variant: str, game: Game) -> Path:
    return out / variant / f"epoch_{game.epoch}_game_{game.round}.json"


async def answer(calls: Calls, model: ModelConfig, game: Game, earlier: list[Game], variant: str, path: Path, records: list[LLMCall]) -> list[str]:
    # one answer, saved as soon as it arrives (a stopped run keeps it), with the keys and the solver's verdicts the status needs
    graded = await ask(calls, model, game, earlier, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = {"epoch": game.epoch, "round": game.round, "variant": variant, "keys": game.keys, "optimal": game.optimal, "grades": graded}
    path.write_text(json.dumps(saved | {"calls": [record.to_dict() for record in records]}, indent=1))
    return graded


def agree(optimal: bool, grade: str) -> bool:
    # as show_tac.py's feedback section: good or strong on an optimal move, mistake or dubious on another
    return grade in (("good", "strong") if optimal else ("mistake", "dubious"))


def status(out: Path) -> dict[str, object]:
    # the saved answers as plyservice's tables: per variant the agreement with the solver, then per variant, situation and kind of move the grades
    rows: dict[tuple[str, str, bool], Counter[str]] = defaultdict(Counter)
    answers: Counter[str] = Counter()
    for path in sorted(out.glob("*/epoch_*_game_*.json")):
        saved = json.loads(path.read_text())
        answers[saved["variant"]] += 1
        for key, optimal, grade in zip(saved["keys"], saved["optimal"], saved["grades"]):
            rows[saved["variant"], key, optimal][grade] += 1

    def share(counters: list[tuple[bool, Counter[str]]]) -> float | None:
        total = sum(sum(counter.values()) for _, counter in counters)
        return sum(count for optimal, counter in counters for grade, count in counter.items() if agree(optimal, grade)) / total if total else None

    variants = [variant for variant in VARIANTS if variant in answers]
    summary = [
        {
            "variant": variant,
            "answers": answers[variant],
            "optimal": share([(True, counter) for (name, _, optimal), counter in rows.items() if name == variant and optimal]),
            "other": share([(False, counter) for (name, _, optimal), counter in rows.items() if name == variant and not optimal]),
            "all": share([(optimal, counter) for (name, _, optimal), counter in rows.items() if name == variant]),
        }
        for variant in variants
    ]
    detail = [
        {"variant": variant, "situation": key, "move": "optimal" if optimal else "not optimal", "n": sum(counter.values())}
        | {grade: counter.get(grade, 0) for grade in (*GRADES, "none")}
        | {"agree": share([(optimal, counter)])}
        for variant in variants
        for (name, key, optimal), counter in rows.items()
        if name == variant
    ]
    rate = {"rate": True}
    return {
        "kind": "tables",
        "tables": [
            {
                "title": "agreement with the solver, per variant",
                "note": "share of graded moves where the feedback agrees: good or strong on an optimal move, mistake or dubious on another",
                "columns": [{"key": "variant", "label": "variant"}, {"key": "answers", "label": "answers", "digits": 0}]
                + [{"key": key, "label": label} | rate for key, label in (("optimal", "optimal moves"), ("other", "other moves"), ("all", "all moves"))],
                "rows": summary,
            },
            {
                "title": "grades per situation",
                "columns": [{"key": "variant", "label": "variant"}, {"key": "situation", "label": "situation"}, {"key": "move", "label": "move"}]
                + [{"key": key, "label": key, "digits": 0} for key in ("n", *GRADES, "none")]
                + [{"key": "agree", "label": "agree"} | rate],
                "rows": detail,
            },
        ],
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_training_source_args(parser)
    parser.add_argument("--out", type=Path, required=True, help="folder of the answers, one file each (relative to the repository: analysis/...)")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["as_run", "history"], help="variants to ask, each over the same games (default as_run history)")
    parser.add_argument("--epochs", nargs="+", type=int, help="epochs whose training games are asked again (default: all); the history still counts every earlier game")
    parser.add_argument("--concurrency", type=int, default=10, help="requests in flight per provider (default 10)")
    parser.add_argument("--check", action="store_true", help="print what would be asked (answers planned and left, the model, the outputs), then exit; no calls")
    parser.add_argument("--json", action="store_true", help="with --check: print the plan as JSON")
    parser.add_argument("--status", action="store_true", help="print the tables of the saved answers as JSON, then exit; no calls")
    parser.add_argument("--dry-run", action="store_true", help="print the changed call of the last selected game per variant, no calls")
    parser.add_argument("--max-chars", type=int, default=70, help="cut each key to this length (default 70)")
    args = parser.parse_args()

    if args.status:
        print(json.dumps(status(args.out)))
        return
    op = build_op(concurrency=args.concurrency)
    harness = training_from_args(op, args)
    runs = harness.get_results().runs
    if len(runs) != 1:
        raise SystemExit(f"select exactly one run with --experiment and --trainees (selected {len(runs)})")
    run = runs[0]
    model = feedback_model(run)
    every = load_games(harness, run, None)
    games = [game for game in every if args.epochs is None or game.epoch in args.epochs]
    planned = [(variant, game) for variant in args.variants for game in games]
    left = [(variant, game) for variant, game in planned if not answer_path(args.out, variant, game).exists()]

    if args.check:
        problems = [] if games else [f"no training game of {run.trainee.to_string()} has a recorded feedback call"]
        plan = {"planned": len(planned), "remaining": len(left), "unit": "answers", "models": [f"{model.provider.value}:{model.model_name}"], "outputs": [str(args.out)]}
        print(json.dumps(plan | {"problems": problems}) if args.json else f"{len(left)} of {len(planned)} answers to ask, model {model.to_string()}, into {args.out}")
        raise SystemExit(1 if problems else 0)
    print(f"{run.trainee.to_string()}\n{len(games)} training games with a recorded feedback call; {len(left)} of {len(planned)} answers to ask", flush=True)
    if args.dry_run:
        game = games[-1]
        for variant in args.variants:
            system, prompt = changed(game, every[: every.index(game)], variant)
            print(f"\n== {variant}: epoch {game.epoch}, game {game.round}\n-- system\n{system}\n-- prompt\n{prompt}")
        return

    done, failed = len(planned) - len(left), 0

    async def one(variant: str, game: Game) -> None:
        nonlocal done, failed
        records: list[LLMCall] = []
        try:
            await answer(Calls(op.llm, game.seed, records.append), model, game, every[: every.index(game)], variant, answer_path(args.out, variant, game), records)
        except Exception as error:
            # left unsaved, so a run again asks it again; the run ends with an error once the others are done
            failed += 1
            print(f"warning: {variant}, epoch {game.epoch}, game {game.round}: {type(error).__name__}: {error}", flush=True)
            return
        done += 1
        print(f"progress: {done}/{len(planned)}", flush=True)

    await asyncio.gather(*(one(variant, game) for variant, game in left))
    for variant in args.variants:
        print(f"\n== {variant}: grades against the solver (agree: good or strong on an optimal move, mistake or dubious on another)")
        counts: dict[str, dict[bool, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
        for game in games:
            path = answer_path(args.out, variant, game)
            if path.exists():
                for key, optimal, grade in zip(game.keys, game.optimal, json.loads(path.read_text())["grades"]):
                    counts[key][optimal][grade] += 1
        grade_table(counts, args.max_chars)
        graded = [(optimal, grade) for by_optimal in counts.values() for optimal, counter in by_optimal.items() for grade in counter.elements()]
        print(f"   agree on all moves: {sum(agree(*pair) for pair in graded)} of {len(graded)}")
    if failed:
        raise SystemExit(f"{failed} answers failed; run the same command again to ask them")


if __name__ == "__main__":
    asyncio.run(main())

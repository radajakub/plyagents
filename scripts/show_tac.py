"""Show what a TAC training run did, to find where and why its test results drop.

Sections (--show, default summary moves):
    summary  per epoch and tester: win rate, optimal decisions and fails with their intervals; for a single-player game also the exact
             value of the optimal policy, and whether the epoch's interval excludes it
    moves    per situation (the TAC key): the moves played in each epoch, the solver's optimal moves marked with *
    memory   per checkpoint: the entries of each component's memory that changed since the previous checkpoint
    games    per game: the actor's key, the memory it read, its reasoning and move against the solver's, the feedback, the memory
             writes, the critic's assessments and a summary of the learner's LLM calls

Examples:
    uv run python scripts/show_tac.py --experiment tac_monty_hall
    uv run python scripts/show_tac.py --experiment tac_monty_hall --show memory --epochs 3 4
    uv run python scripts/show_tac.py --experiment tac_monty_hall --show games --epochs 4 --rounds 1 2 --full
    uv run python scripts/show_tac.py --experiment tac_monty_hall --show games --phase test --epochs 5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

from plybench.analysis.replay import ReplayedStep, TurnBasedReplayer, build_replayer
from plybench.analysis.statistics.bundle import CIBundle
from plybench.analysis.training import TrainingAnalysis
from plybench.analysis.visual import game_title, player_label
from plybench.app import PlyBench
from plybench.cli.shared import add_training_source_args, build_op, training_from_args
from plybench.common.enums import GameResults, MetricName
from plybench.configs.player_config import PlayerConfig
from plybench.configs.training_run import TrainingRun
from plybench.core.minimax import optimal_outcomes
from plybench.harness.training.results import TrainingResults
from plybench.harness.training.training import TrainingHarness
from plybench.player.learnable import LEARNER_CALLS
from plybench.trackers.result_tracker import ResultTracker

from plyagents.tac.player import STATE_FILE, UPDATE_FILE

SECTIONS = ("summary", "moves", "memory", "games")
SUMMARY_METRICS = (MetricName.WIN_RATE, MetricName.OPTIMALITY_RATE_NON_TRIVIAL, MetricName.FAIL_RATE)


def cut(text: str | None, limit: int | None) -> str:
    # one line, at most limit characters; None keeps the text whole, its line breaks included
    if text is None:
        return "-"
    if limit is None:
        return text.strip()
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def block(label: str, text: str | None, limit: int | None, indent: int = 4) -> str:
    # a labelled value; a whole text with several lines goes under its label, indented
    body = cut(text, limit)
    pad = " " * indent
    if "\n" not in body:
        return f"{pad}{label}: {body}"
    return f"{pad}{label}:\n" + "\n".join(f"{pad}  {line}" for line in body.splitlines())


def interval(bundle: CIBundle) -> str:
    ci = bundle.wilson or bundle.t
    bounds = f" [{ci.lower:.2f}, {ci.upper:.2f}]" if ci is not None else ""
    return f"{bundle.value:.2f}{bounds}"


def exact_optimum(op: PlyBench, run: TrainingRun) -> dict[GameResults, float] | None:
    # the outcome probabilities of the solved policy; None for a game with an opponent, whose policy decides them
    game = op.registry.build_engine(run.game).game
    return optimal_outcomes(game) if game.single_player() else None


def phase_trackers(results: TrainingResults, run: TrainingRun, phase: str) -> dict[int, list[ResultTracker]]:
    # the trackers of each epoch for the phase: the training games (epochs 1..N) or one tracker per tester (epochs 0..N)
    trackers: dict[int, list[ResultTracker]] = {}
    for epoch in results.for_run(run):
        if phase == "test":
            trackers[epoch.epoch.index] = epoch.tests
        elif epoch.train is not None:
            trackers[epoch.epoch.index] = [epoch.train]
    return trackers


def replay(replayer: TurnBasedReplayer, tracker: ResultTracker) -> list[tuple[int, list[ReplayedStep]]]:
    # the trainee's steps of each recorded game, judged by the solver
    games = [(round_, game) for round_, game in enumerate(tracker.games, start=1) if game is not None]
    return [(round_, replayer.replay_steps(game, tracker.i)) for round_, game in games]


def show_summary(analysis: TrainingAnalysis, run: TrainingRun, testers: list[PlayerConfig], optimum: dict[GameResults, float] | None) -> None:
    print("== Summary (test phases; 95% intervals: Wilson for rates)")
    if optimum is not None:
        exact = ", ".join(f"{result.name.lower()} {probability:.3f}" for result, probability in optimum.items() if probability > 0)
        print(f"   optimal policy (exact): {exact}")
    for tester in testers:
        print(f"-- vs {player_label(tester)}")
        curve = analysis.curve(run, tester)
        print(f"   {'epoch':>5}  {'n':>4}  {'win rate':<20}{'optimal decisions':<20}{'fail rate':<20}{'win Δ':>6}")
        previous: float | None = None
        for point in curve.points:
            win, optimal, fail = (point.bundle(metric) for metric in SUMMARY_METRICS)
            delta = f"{win.value - previous:+.2f}" if previous is not None else ""
            flags = []
            if optimum is not None and win.wilson is not None and not win.wilson.lower <= optimum.get(GameResults.WIN, 0.0) <= win.wilson.upper:
                flags.append("interval excludes the optimum")
            if optimal.n and optimal.value < 1.0:
                flags.append(f"{round((1 - optimal.value) * optimal.n)} of {optimal.n} decisions not optimal")
            print(f"   {point.epoch:>5}  {win.n:>4}  {interval(win):<20}{interval(optimal):<20}{interval(fail):<20}{delta:>6}  {'; '.join(flags)}")
            previous = win.value


def show_moves(results: TrainingResults, run: TrainingRun, replayer: TurnBasedReplayer, phase: str, top: int | None, limit: int | None) -> None:
    print(f"== Moves per situation ({phase} phases; * the solver's optimal moves)")
    # tester (or the trainer) -> key -> epoch -> move counts; the optimal moves and the order of first visit, per key
    counts: dict[str, dict[str, dict[int, Counter[str]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    optimal: dict[str, set[str]] = defaultdict(set)
    differing: set[str] = set()
    for epoch, trackers in phase_trackers(results, run, phase).items():
        for tracker in trackers:
            for _, steps in replay(replayer, tracker):
                for replayed in steps:
                    key = (replayed.step.data or {}).get("key", replayed.step.observation)
                    counts[tracker.o.to_string()][key][epoch][replayed.step.move] += 1
                    # in an imperfect-information game one key covers several true states; their optimal moves should agree
                    if optimal[key] and optimal[key] != set(replayed.optimal_moves):
                        differing.add(key)
                    optimal[key] |= set(replayed.optimal_moves)
    for opponent, keys in counts.items():
        print(f"-- vs {opponent}")
        ranked = sorted(keys.items(), key=lambda item: -sum(sum(counter.values()) for counter in item[1].values()))
        for number, (key, epochs) in enumerate(ranked[:top], start=1):
            moves = sorted({move for counter in epochs.values() for move in counter}, key=lambda move: (move not in optimal[key], move))
            visits = sum(sum(counter.values()) for counter in epochs.values())
            print(f"   situation {number} ({visits} moves): {cut(key, limit)}")
            if key in differing:
                print("   warning: the optimal moves differ between the true states of this key; * marks their union")
            names = [f"{move}{'*' if move in optimal[key] else ''}" for move in moves]
            width = max(8, *(len(name) + 1 for name in names))
            print(f"     {'epoch':>5}" + "".join(f"{name:>{width}}" for name in names) + f"{'optimal':>{width}}")
            for epoch in sorted(epochs):
                counter = epochs[epoch]
                share = sum(counter[move] for move in moves if move in optimal[key]) / sum(counter.values())
                print(f"     {epoch:>5}" + "".join(f"{counter.get(move, 0) or '':>{width}}" for move in moves) + f"{share:>{width}.2f}")
        if top is not None and len(ranked) > top:
            print(f"   ({len(ranked) - top} more situations; raise --top to see them)")


def entries(state: Any) -> dict[str, dict[str, Any]]:
    # a component's checkpoint state by key: a memory's entries, or a table's estimates; None for a component without one
    return state if isinstance(state, dict) else {}


def estimates(entry: dict[str, Any]) -> str:
    # a table entry: the tabular critic's V(z), one estimate, or the tabular actor's Q(z, .), one estimate per action
    if "value" in entry:
        return f"value {entry['value']:+.3f} ({entry['count']} updates)"
    return ", ".join(f"{action} {estimate['value']:+.3f} ({estimate['count']})" for action, estimate in entry.items())


def show_memory(harness: TrainingHarness, run: TrainingRun, epochs: list[int], all_entries: bool, limit: int | None) -> None:
    print("== Memory per checkpoint (entries that changed since the previous checkpoint)")
    for epoch in epochs:
        path = harness.path_builder.checkpoint(harness.experiment, run, epoch) / STATE_FILE
        if not path.exists():
            print(f"-- epoch {epoch}: no checkpoint at {path}")
            continue
        state = json.loads(path.read_text())
        previous_path = harness.path_builder.checkpoint(harness.experiment, run, epoch - 1) / STATE_FILE
        previous = json.loads(previous_path.read_text()) if epoch > 0 and previous_path.exists() else {}
        for component in ("actor", "critic"):
            current, before = entries(state.get(component)), entries(previous.get(component))
            if not current and not before:
                continue
            changed = {key: entry for key, entry in current.items() if all_entries or before.get(key) != entry}
            print(f"-- epoch {epoch}, {component}: {len(current)} entries, {len(changed)} {'shown' if all_entries else 'changed'}")
            for key, entry in changed.items():
                old = before.get(key)
                print(f"   {'new' if old is None else 'changed' if old != entry else 'same'}: {cut(key, limit)}")
                if "text" in entry:
                    actions = ", ".join(f"{move} {count}" for move, count in entry.get("actions", {}).items())
                    print(f"    visits {entry['visits']}, writes {entry['writes']}, actions: {actions or '-'}")
                    print(block("text", entry["text"] or "(empty)", limit))
                    if old is not None and old.get("text") != entry["text"]:
                        print(block("was", old.get("text") or "(empty)", limit))
                else:
                    print(f"    {estimates(entry)}" + (f" (was {estimates(old)})" if old is not None and old != entry else ""))


def critic_assessments(calls: list[dict[str, Any]], keys: list[str]) -> dict[str, str]:
    # the critic's answer for each key; the calls run at once and are recorded as they finish, so each is matched by its prompt
    assessed: dict[str, str] = {}
    for call in calls:
        if call["role"] != "critic" or call.get("failure"):
            continue
        key = next((key for key in keys if call["prompt_message"].startswith(f"The situation:\n{key}\n\n")), None)
        if key is not None:
            assessed[key] = call["full_output"]
    return assessed


def call_summary(calls: list[dict[str, Any]]) -> str:
    roles = Counter(call["role"] for call in calls)
    failures = sum(1 for call in calls if call.get("failure"))
    tokens = sum(call["input_tokens"] for call in calls), sum(call["output_tokens"] for call in calls)
    parts = ", ".join(f"{role} {count}" for role, count in roles.items())
    return f"{len(calls)} calls ({parts}); tokens in {tokens[0]}, out {tokens[1]}" + (f"; {failures} unusable answers" if failures else "")


def show_update_part(name: str, part: dict[str, Any] | None, limit: int | None) -> None:
    # what one component learned from the game: the memory writes, or the table samples
    if part is None:
        return
    for write in part.get("writes", []):
        if not write["lesson"] and write["old"] == write["new"]:
            continue
        print(f"    {name} write (alpha {write['alpha']}{', cut to the budget' if write['cut'] else ''}): {cut(write['key'], limit)}")
        print(block("lesson", write["lesson"] or "(none)", limit, 6))
        # the old entry after Apply shortened it to make room; shown only when it was shortened
        if write["compressed"] != write["old"]:
            print(block("old, shortened", write["compressed"] or "(empty)", limit, 6))
        print(block("new entry", write["new"] or "(empty)", limit, 6))
    for sample in part.get("samples", []):
        action = f", {sample['action']}" if sample.get("action") else ""
        print(f"    {name} sample: target {sample['target']:+.3f}{action}: {cut(sample['key'], limit)}")


def show_games(
    harness: TrainingHarness, results: TrainingResults, run: TrainingRun, replayer: TurnBasedReplayer, phase: str, epochs: list[int], rounds: list[int] | None, limit: int | None
) -> None:
    print(f"== Games ({phase} phases)")
    trackers = phase_trackers(results, run, phase)
    for epoch in epochs:
        for tracker in trackers.get(epoch, []):
            for round_, steps in replay(replayer, tracker):
                if rounds is not None and round_ not in rounds:
                    continue
                game = tracker.games[round_ - 1]
                assert game is not None
                versus = f" vs {player_label(tracker.o)}" if phase == "test" else ""
                print(f"-- epoch {epoch}, {phase} game {round_}{versus} (seed {game.seed}): {game.get_result(tracker.i).name.lower()}")
                learner = harness.path_builder.learner_dir(harness.experiment, run, epoch) / f"game_{round_}"
                update = json.loads((learner / UPDATE_FILE).read_text()) if phase == "train" and (learner / UPDATE_FILE).exists() else None
                calls = json.loads((learner / LEARNER_CALLS).read_text()) if phase == "train" and (learner / LEARNER_CALLS).exists() else []
                feedback = (update or {}).get("actor", {}) or {}
                verdicts = feedback.get("feedback", [])
                keys = [(step.step.data or {}).get("key", "") for step in steps]
                assessed = critic_assessments(calls, keys)
                for index, replayed in enumerate(steps):
                    data = replayed.step.data or {}
                    mark = "optimal" if replayed.is_optimal else f"not optimal; optimal: {', '.join(replayed.optimal_moves)}"
                    print(f"   step {index + 1}: {replayed.step.move} ({mark})")
                    print(block("key", data.get("key"), limit))
                    print(block("memory read", data.get("context"), limit))
                    print(block("reasoning", data.get("reasoning_trace"), limit))
                    if keys[index] in assessed:
                        print(block("critic", assessed.pop(keys[index]), limit))
                    if index < len(verdicts):
                        verdict = verdicts[index]
                        print(f"    feedback: {verdict['grade']}: {cut(verdict['reason'], limit)}")
                        print(block("lesson", verdict.get("lesson"), limit, 6))
                        if verdict.get("critic_note"):
                            print(block("critic note", verdict["critic_note"], limit, 6))
                if phase == "train":
                    if update is None:
                        print(f"    no update recorded at {learner}")
                    else:
                        show_update_part("actor", update.get("actor"), limit)
                        show_update_part("critic", update.get("critic"), limit)
                    if calls:
                        print(f"    learner: {call_summary(calls)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_training_source_args(parser)
    parser.add_argument("--show", nargs="+", choices=SECTIONS, default=["summary", "moves"], help="sections to print (default summary moves)")
    parser.add_argument("--phase", choices=("train", "test"), help="the games the moves and games sections read (default: test for moves, train for games)")
    parser.add_argument("--epochs", nargs="+", type=int, help="epochs for the memory and games sections (default: all)")
    parser.add_argument("--rounds", nargs="+", type=int, help="game rounds for the games section (default: all)")
    parser.add_argument("--top", type=int, default=20, help="situations in the moves section, most visited first (default 20)")
    parser.add_argument("--all-entries", action="store_true", help="memory section: print unchanged entries too")
    parser.add_argument("--max-chars", type=int, default=300, help="cut each text to one line of this length (default 300)")
    parser.add_argument("--full", action="store_true", help="print texts whole, line breaks included")
    args = parser.parse_args()

    op = build_op()
    harness = training_from_args(op, args)
    results = harness.get_results()
    if not results.runs:
        raise SystemExit("the training selection is empty; check --experiment / --games / --trainees")
    testers = [op.registry.player_config(value) for value in harness.tester_configs]
    analysis = TrainingAnalysis(results, op.registry)
    limit = None if args.full else args.max_chars

    for run in results.runs:
        print(f"\n######## {game_title(run.game)}: {player_label(run.trainee)}")
        print(f"trainer {run.trainer.to_string()}; {run.training_config.to_string()}; replicate {run.replicate}")
        print(f"records: {harness.path_builder.run_dir(harness.experiment, run)}")
        replayer = build_replayer(op.registry, run.game)
        epochs = args.epochs if args.epochs is not None else list(range(run.training_config.num_epochs + 1))
        if "summary" in args.show:
            show_summary(analysis, run, testers, exact_optimum(op, run))
        if "moves" in args.show:
            show_moves(results, run, replayer, args.phase or "test", args.top, limit)
        if "memory" in args.show:
            show_memory(harness, run, epochs, args.all_entries, limit)
        if "games" in args.show:
            show_games(harness, results, run, replayer, args.phase or "train", [epoch for epoch in epochs if epoch > 0 or args.phase == "test"], args.rounds, limit)


if __name__ == "__main__":
    main()

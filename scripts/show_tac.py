"""Show what a TAC training run did, to find where and why its test results drop.

Sections (--show, default summary moves):
    summary  per epoch and tester: win rate, optimal decisions and fails with their intervals; for a single-player game also the exact
             value of the optimal policy, and whether the epoch's interval excludes it
    moves    per situation (the TAC key): the moves played in each epoch, the solver's optimal moves marked with *
    memory   per checkpoint: the entries of each component's memory that changed since the previous checkpoint
    games    per game: the actor's key, the memory it read, its reasoning and move against the solver's, the feedback, the memory
             writes, the critic's assessments and a summary of the learner's LLM calls
    feedback per situation: the grades of the feedback model on the training moves, split by whether the solver calls the move optimal
    reference per epoch and situation: the counts the player recorded beside the text (visits, share, mean outcome and its spread per move,
             Q* where solved) and how often the feedback's grade has the sign of the counted advantage (mean outcome of the move minus that of
             the situation, from the counts before the game) and agrees with the solver (Q* = V*)
    regret   across runs: the runs that differ only in a bandit's shift are replicates of one condition. Per condition the cumulative
             regret over the training games (the sum of V* - Q*(a) of the moves played, as in the bandit literature's pseudo-regret), the
             share of optimal games among the last 25, the moves tried, the runs that never play optimally in the second half (suffix
             failures), and the test games' regret per epoch (V* - V of the memory's policy, without exploration); means with 95% bootstrap
             intervals over the runs. --plot <png> draws the cumulative regret and the test regret

Examples:
    uv run python scripts/show_tac.py --experiment tac_monty_hall
    uv run python scripts/show_tac.py --experiment tac_monty_hall --show memory --epochs 3 4
    uv run python scripts/show_tac.py --experiment tac_monty_hall --show games --epochs 4 --rounds 1 2 --full
    uv run python scripts/show_tac.py --experiment tac_monty_hall --show games --phase test --epochs 5
    uv run python scripts/show_tac.py --experiment tac_bandit --show regret --plot plots/tac_bandit/regret.png
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from plybench.analysis.replay import ReplayedStep, TurnBasedReplayer, build_replayer
from plybench.analysis.statistics.bundle import CIBundle
from plybench.analysis.statistics.distribution import Distribution
from plybench.analysis.statistics.intervals import bootstrap_ci
from plybench.analysis.training import TrainingAnalysis
from plybench.analysis.visual import CATEGORICAL, Axis, BaselineLayer, CategoryTicks, Figure, Layer, LineLayer, Panel, StyleOverride, game_title, player_label, render
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

from plyagents.tac.model.counts import MoveCounts, counts_from_dict
from plyagents.tac.model.feedback import GRADES
from plyagents.tac.player import STATE_FILE, UPDATE_FILE

SECTIONS = ("summary", "moves", "memory", "games", "feedback", "reference", "regret")
# the training games at the end of a run whose share of optimal games is reported
LAST_GAMES = 25
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


def tables(state: dict[str, Any]) -> dict[str, Any]:
    # the parts of a checkpoint the memory section prints, by name: the tabular critic's V and Q apart (its state holds both)
    critic = state.get("critic")
    if isinstance(critic, dict) and set(critic) == {"values", "action_values"}:
        return {"actor": state.get("actor"), "critic": critic["values"], "critic Q": critic["action_values"]}
    return {"actor": state.get("actor"), "critic": critic}


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
        for component, part in tables(state).items():
            current, before = entries(part), entries(tables(previous).get(component))
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


def show_feedback(harness: TrainingHarness, results: TrainingResults, run: TrainingRun, replayer: TurnBasedReplayer, epochs: list[int], limit: int | None) -> None:
    print("== Feedback grades against the solver (training games; agree: good or strong on an optimal move, mistake or dubious on another)")
    # key -> whether the solver calls the move optimal -> grade counts, keys in the order of first visit
    counts: dict[str, dict[bool, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for epoch, trackers in phase_trackers(results, run, "train").items():
        if epoch not in epochs:
            continue
        for tracker in trackers:
            for round_, steps in replay(replayer, tracker):
                path = harness.path_builder.learner_dir(harness.experiment, run, epoch) / f"game_{round_}" / UPDATE_FILE
                verdicts = (json.loads(path.read_text()).get("actor") or {}).get("feedback", []) if path.exists() else []
                for index, replayed in enumerate(steps):
                    key = (replayed.step.data or {}).get("key", replayed.step.observation)
                    grade = (verdicts[index].get("grade") if index < len(verdicts) else None) or "none"
                    counts[key][replayed.is_optimal][grade] += 1
    grade_table(counts, limit)


def advantage_agrees(grade: str, advantage: float) -> bool:
    # as reask_feedback.py's agreement with the TD error: good or strong where it is positive, mistake or dubious where negative, neutral at 0
    if abs(advantage) < 1e-9:
        return grade == "neutral"
    return grade in (("good", "strong") if advantage > 0 else ("mistake", "dubious"))


def counted_advantage(counts: dict[str, MoveCounts], move: str) -> float | None:
    # the move's mean outcome minus the situation's, from the counts before the game; None while the move has no visit
    played = counts.get(move)
    if played is None or played.visits == 0:
        return None
    return played.total / played.visits - sum(other.total for other in counts.values()) / sum(other.visits for other in counts.values())


def show_reference(harness: TrainingHarness, run: TrainingRun, epochs: list[int], limit: int | None) -> None:
    print("== Reference per epoch: the counts after the epoch's training games, and the feedback's grades against the counted advantage and the solver")
    for epoch in [epoch for epoch in epochs if epoch > 0]:
        path = harness.path_builder.checkpoint(harness.experiment, run, epoch) / STATE_FILE
        state = json.loads(path.read_text()) if path.exists() else {}
        if "reference" not in state:
            print(f"-- epoch {epoch}: no reference in {path} (a run from before it was recorded)")
            continue
        learner = harness.path_builder.learner_dir(harness.experiment, run, epoch)
        updates = sorted(learner.glob(f"game_*/{UPDATE_FILE}"), key=lambda update: int(update.parent.name.removeprefix("game_")))
        # key -> the solved values of its legal moves, and the agreement tallies (graded, with the counted sign, counted at all, with the solver, solved at all)
        solved: dict[str, dict[str, float]] = {}
        tally: dict[str, Counter[str]] = defaultdict(Counter)
        for update in updates:
            for step in (json.loads(update.read_text()).get("reference") or {}).get("steps", []):
                key, move, grade = step["key"], step["move"], step["grade"]
                if step["solved"] is not None and step["solved"]["action_values"] is not None:
                    solved[key] = dict(zip(step["legal"], step["solved"]["action_values"]))
                if move is None or grade is None:
                    continue
                tally[key]["graded"] += 1
                counted = counted_advantage(counts_from_dict(step["counts"]), move)
                if counted is not None:
                    tally[key]["counted"] += 1
                    tally[key]["counted_agree"] += advantage_agrees(grade, counted)
                if step["solved"] is not None and step["solved"]["value"] is not None:
                    optimal = abs(step["solved"]["action_values"][step["legal"].index(move)] - step["solved"]["value"]) < 1e-9
                    tally[key]["solver"] += 1
                    tally[key]["solver_agree"] += grade in (("good", "strong") if optimal else ("mistake", "dubious"))
        print(f"-- epoch {epoch}: {len(updates)} training games")
        for key, raw in state["reference"].items():
            counts = counts_from_dict(raw)
            total = sum(move_counts.visits for move_counts in counts.values())
            values = solved.get(key, {})
            best = max(values.values(), default=None)
            print(f"   {cut(key, limit)}")
            print(f"     {'move':<16}{'visits':>8}{'share':>8}{'mean':>8}{'sd':>8}{'Q*':>8}")
            for move, move_counts in counts.items():
                mean = move_counts.total / move_counts.visits
                spread = max(move_counts.squares / move_counts.visits - mean * mean, 0.0) ** 0.5
                star = f"{values[move]:+.2f}{'*' if values[move] == best else ' '}" if move in values else "-"
                print(f"     {move:<16}{move_counts.visits:>8}{move_counts.visits / total:>8.2f}{mean:>+8.2f}{spread:>8.2f}{star:>8}")
            counted = tally[key]
            if counted["graded"]:
                shares = [f"{name} {counted[f'{name}_agree'] / counted[name]:.2f} of {counted[name]}" if counted[name] else f"{name} -" for name in ("counted", "solver")]
                print(f"     feedback: {counted['graded']} graded this epoch; agrees with the counted advantage's sign and with the solver: {', '.join(shares)}")


@dataclass
class RunRegret:
    # one run's regret per game: the training games in the order played, the test games per epoch; a game's regret is the sum of its steps'
    train: list[float] = field(default_factory=list)
    train_optimal: list[bool] = field(default_factory=list)
    moves: set[str] = field(default_factory=set)
    test: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))


def replicate_group(run: TrainingRun) -> tuple[str, str]:
    # the shift of a bandit moves the best button, not the problem: runs that differ only in it are replicates of one condition
    return re.sub(r",shift=\d+", "", run.game.to_string()), run.trainee.to_string()


def run_regret(results: TrainingResults, run: TrainingRun, replayer: TurnBasedReplayer) -> RunRegret:
    regret = RunRegret()
    for _, trackers in sorted(phase_trackers(results, run, "train").items()):
        for tracker in trackers:
            for _, steps in replay(replayer, tracker):
                regret.train.append(sum(replayed.regret for replayed in steps))
                regret.train_optimal.append(all(replayed.is_optimal for replayed in steps))
                regret.moves |= {replayed.step.move for replayed in steps}
    for epoch, trackers in phase_trackers(results, run, "test").items():
        for tracker in trackers:
            regret.test[epoch] += [sum(replayed.regret for replayed in steps) for _, steps in replay(replayer, tracker)]
    return regret


def uniform_regret(op: PlyBench, run: TrainingRun, replayer: TurnBasedReplayer) -> float | None:
    # the regret per game of the uniform policy at the first decision, exact (all of it in a one-decision game such as the bandit); None when chance moves first
    engine = op.registry.build_engine(run.game)
    engine.reset()
    if engine.game.is_chance_node():
        return None
    pid = engine.game.get_player()
    legal = [engine.action_class.from_openspiel(move, engine.interface_transformer) for move in engine.game.get_legal_moves(pid)]
    _, _, verdict = replayer.probe(engine.game.serialize_state(), legal[0].to_llm().string)
    return mean(verdict.V() - verdict.Q(move.number) for move in legal)


def over_runs(values: list[float]) -> tuple[float, float, float]:
    # the mean over the runs and its 95% BCa bootstrap interval (seeded): regret is bounded below and skewed over few runs, where a t interval
    # reaches below 0
    ci = bootstrap_ci(Distribution.from_values(values))
    return ci.value, ci.lower, ci.upper


@dataclass
class Curve:
    # one condition on one game, for the plot: its label, the cumulative regret per training game and the test regret per epoch (mean, low, high),
    # and each run's cumulative regret at the end of training
    game: str
    label: str
    cumulative: list[tuple[float, float, float]]
    test: list[tuple[int, tuple[float, float, float]]]
    uniform: float | None
    finals: list[float]


def condition_label(trainee: PlayerConfig) -> str:
    # the trainee's config without its model names, which every condition shares and which make the labels long; player_label names only the
    # component types, so the conditions of one experiment (selectors, options) would share a label
    text = re.sub(r"model=[^,\[\]]+(\[[^\]]*\])?,?", "", trainee.params.to_string())
    return re.sub(r"\[\]", "", text.replace("[,", "[").replace(",]", "]"))


def show_regret(op: PlyBench, results: TrainingResults, plot: Path | None) -> None:
    print("== Regret per condition (runs differing only in a bandit's shift pooled; means over runs with 95% BCa bootstrap intervals)")
    groups: dict[tuple[str, str], list[RunRegret]] = defaultdict(list)
    labels: dict[tuple[str, str], str] = {}
    uniform: dict[tuple[str, str], float | None] = {}
    for run in results.runs:
        group = replicate_group(run)
        replayer = build_replayer(op.registry, run.game)
        groups[group].append(run_regret(results, run, replayer))
        labels[group] = condition_label(run.trainee)
        uniform.setdefault(group, uniform_regret(op, run, replayer))
    curves: list[Curve] = []
    for group, runs in groups.items():
        game, trainee = group
        games = min(len(run.train) for run in runs)
        print(f"-- {game}: {trainee}\n   {len(runs)} runs, {games} training games each")
        if games == 0:
            continue
        cumulative = [over_runs([sum(run.train[: index + 1]) for run in runs]) for index in range(games)]
        value, low, high = cumulative[-1]
        reference = uniform[group]
        print(f"   cumulative regret: {value:.2f} [{low:.2f}, {high:.2f}]" + (f" (uniform policy: {reference * games:.2f})" if reference is not None else ""))
        last = min(LAST_GAMES, games)
        value, low, high = over_runs([mean(run.train_optimal[games - last : games]) for run in runs])
        print(f"   optimal games among the last {last}: {value:.2f} [{low:.2f}, {high:.2f}]")
        print(
            f"   moves tried: {mean(len(run.moves) for run in runs):.1f} on average; suffix failures: {sum(not any(run.train_optimal[games // 2 : games]) for run in runs)} of {len(runs)} runs"
        )
        epochs = sorted(set.intersection(*(set(run.test) for run in runs)))
        test = [(epoch, over_runs([mean(run.test[epoch]) for run in runs])) for epoch in epochs if all(run.test[epoch] for run in runs)]
        print("   test regret per game: " + ", ".join(f"epoch {epoch} {value:.2f} [{low:.2f}, {high:.2f}]" for epoch, (value, low, high) in test))
        curves.append(Curve(game, labels[group], cumulative, test, reference, [sum(run.train[:games]) for run in runs]))
    if plot is not None and curves:
        print(f"   plot: {plot_regret(curves, plot)}")


def plot_regret(curves: list[Curve], path: Path) -> Path:
    # one row per game: the cumulative regret over the training games, the test regret per epoch, and each run's cumulative regret at the end (one
    # dot per run, the mean with its interval beside them: a mean over few runs hides whether all runs did a little worse or a few got stuck). A
    # condition keeps its colour in every row, in the order of the experiment file; the uniform policy in grey where it is known (exact for a
    # one-decision game); bands and whiskers are 95% bootstrap intervals over the runs
    colours = {label: CATEGORICAL[index % len(CATEGORICAL)] for index, label in enumerate(dict.fromkeys(curve.label for curve in curves))}
    grey = StyleOverride(color="#898781", linestyle="--")
    panels = []
    for game in dict.fromkeys(curve.game for curve in curves):
        cumulative_layers: list[Layer] = []
        test_layers: list[Layer] = []
        rows = [curve for curve in curves if curve.game == game]
        for curve in rows:
            style = StyleOverride(color=colours[curve.label])
            games = tuple(float(number) for number in range(1, len(curve.cumulative) + 1))
            bands = tuple((low, high) for _, low, high in curve.cumulative)
            cumulative_layers.append(LineLayer(games, tuple(value for value, _, _ in curve.cumulative), curve.label, bands, False, style=style))
            if curve.test:
                epochs = tuple(float(epoch) for epoch, _ in curve.test)
                bands = tuple((low, high) for _, (_, low, high) in curve.test)
                test_layers.append(LineLayer(epochs, tuple(value for _, (value, _, _) in curve.test), curve.label, bands, style=style))
        reference = next((curve.uniform for curve in rows if curve.uniform is not None), None)
        if reference is not None:
            longest = max(len(curve.cumulative) for curve in rows)
            cumulative_layers.append(LineLayer((0.0, float(longest)), (0.0, reference * longest), "uniform policy", show_markers=False, style=grey))
            test_layers.append(BaselineLayer(reference, "uniform policy", in_legend=True, style=grey))
        run_layers: list[Layer] = []
        for index, curve in enumerate(rows):
            dots = StyleOverride(color=colours[curve.label], linestyle="none")
            spread = tuple(index - 0.2 + 0.4 * number / max(len(curve.finals) - 1, 1) for number in range(len(curve.finals)))
            run_layers.append(LineLayer(spread, tuple(curve.finals), curve.label, style=dots))
            value, low, high = curve.cumulative[-1]
            run_layers.append(LineLayer((index + 0.35,), (value,), "", ((low, high),), style=StyleOverride(color=colours[curve.label], marker="D")))
        if reference is not None:
            run_layers.append(BaselineLayer(reference * max(len(curve.cumulative) for curve in rows), "uniform policy", style=grey))
        ticks = CategoryTicks(tuple(str(number) for number in range(1, len(rows) + 1)))
        panels += [
            Panel(cumulative_layers, Axis("cumulative regret", include_zero=True), Axis("training game"), f"{game}: training games"),
            Panel(test_layers, Axis("regret per game", include_zero=True), Axis("epoch"), f"{game}: test games"),
            Panel(run_layers, Axis("cumulative regret at the end", include_zero=True), Axis("condition (legend order)", ticks=ticks), f"{game}: each run"),
        ]
    return render(Figure(panels, ncols=3, share_x=False), path)


def grade_table(counts: dict[str, dict[bool, Counter[str]]], limit: int | None) -> None:
    # per key: the grades given to optimal moves and to other moves, and the share that agrees with the solver; also used by reask_feedback.py
    columns = (*GRADES, "none")
    for key, by_optimal in counts.items():
        print(f"   {cut(key, limit)}")
        print(f"     {'move':<12}" + "".join(f"{grade:>9}" for grade in columns) + f"{'agree':>9}")
        for optimal in (True, False):
            counter = by_optimal.get(optimal)
            if not counter:
                continue
            agreeing = ("good", "strong") if optimal else ("mistake", "dubious")
            share = sum(counter[grade] for grade in agreeing) / sum(counter.values())
            label = "optimal" if optimal else "not optimal"
            print(f"     {label:<12}" + "".join(f"{counter.get(grade, 0) or '':>9}" for grade in columns) + f"{share:>9.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_training_source_args(parser)
    parser.add_argument("--show", nargs="+", choices=SECTIONS, default=["summary", "moves"], help="sections to print (default summary moves)")
    parser.add_argument("--phase", choices=("train", "test"), help="the games the moves and games sections read (default: test for moves, train for games)")
    parser.add_argument("--epochs", nargs="+", type=int, help="epochs for the memory, games and feedback sections (default: all)")
    parser.add_argument("--rounds", nargs="+", type=int, help="game rounds for the games section (default: all)")
    parser.add_argument("--top", type=int, default=20, help="situations in the moves section, most visited first (default 20)")
    parser.add_argument("--all-entries", action="store_true", help="memory section: print unchanged entries too")
    parser.add_argument("--max-chars", type=int, default=300, help="cut each text to one line of this length (default 300)")
    parser.add_argument("--full", action="store_true", help="print texts whole, line breaks included")
    parser.add_argument("--plot", type=Path, help="regret section: draw the cumulative and the test regret into this PNG")
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
        if "feedback" in args.show:
            show_feedback(harness, results, run, replayer, epochs, limit)
        if "reference" in args.show:
            show_reference(harness, run, epochs, limit)
    if "regret" in args.show:
        show_regret(op, results, args.plot)


if __name__ == "__main__":
    main()

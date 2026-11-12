"""Ask the LLM feedback model again about the training games of a finished TAC run, with other information than it had, and grade its answers.

Offline: no game is played. Each training game that had a feedback call is asked again, and the new grades are compared with the solver's verdict on
each move (as the feedback section of show_tac.py does for the recorded grades). Calls the model: costs money.

Variants of the first round, each the recorded call changed:
    as_run   the recorded call unchanged: the noise of the grades, the baseline for the others
    caution  the system message also says the rules may leave parts of the game unstated, so a known puzzle's answer may not hold
    history  the prompt also shows, for each situation of the game, the moves the agent played there in the run's earlier training games
             and their outcomes: the evidence across games the feedback model never sees in a run

Levels of the information ladder (experiments/tac-feedback-fixes.md in the phd repository), each call built again from the recorded steps by the
feedback model's own code with the inputs of the level (LLMFeedback, inputs=...), each level the one before plus one input:
    L0  the moves and the outcome (REINFORCE)            L3  + the critic's value of every legal move (all-action policy gradient)
    L1  + the actor's memory entry it read (theta)       L4  + the earlier games in the same situations (the batch)
    L2  + the critic's value before, after, change (TD)  L5  + the actor's (and an LLM critic's) reasoning
The levels from L2 on are asked once per critic (--critics): tabular (Monte Carlo V and Q, replayed over the run's training games in the order they
were played, so each call sees only the games before it) and oracle (V* and Q* of the solved game: the bound, privileged). --no-q leaves L3 out and q
out of L4 and L5.
    obfuscated  L0 with the game renamed (doors to boxes, the host to an operator, the car to a prize): no cue of the known puzzle
--inputs asks explicit sets of inputs (as the feedback's inputs=..., e.g. memory+history) besides the ladder; a set without delta and q is asked
with no critic and named <inputs>_nocritic (the text-only runs: no number from another method in the call), one with them once per critic.

--sample-seeds asks each call again with other seeds (0 is the game's own, which reproduces the recorded answers): the noise of the grades.

Each answer is saved as its own file, <out>/<name>/epoch_<e>_game_<r>.json (the grades, the keys and the solver's verdicts, the call), and an
answer already saved is not asked again, so a stopped run continues. The name is the variant, the level's inputs and critic (e.g.
memory+delta_tabular), or obfuscated, with _seed<n> for another seed. A plyservice scripts job (see its README): --check --json prints the plan and
checks that the calls are rebuilt exactly (the run's own inputs give its recorded prompts); --job-report <path> keeps a report (plybench.report) of the
names this command asks (those with answers left when it starts) while it works: their progress and the tables of their saved answers.

Examples:
    uv run python scripts/reask_feedback.py --experiment tac_monty_hall_ladder --trainees "<R3 config>" --out analysis/tac_monty_hall_ladder/reask_r3
    uv run python scripts/reask_feedback.py --experiment tac_monty_hall_ladder --trainees "<R3 config>" --out <dir> --variants --levels L0 L1 L2 L4 L5 --no-q
    uv run python scripts/reask_feedback.py --experiment tac_monty_hall_ladder --trainees "<R3 config>" --out <dir> --variants as_run --sample-seeds 1 2
    uv run python scripts/reask_feedback.py --experiment tac_monty_hall_ladder --trainees "<R3 config>" --out <dir> --levels L2 --dry-run
    uv run python scripts/reask_feedback.py --experiment tac_monty_hall_ladder --trainees "<R3 config>" --out <dir> --variants --inputs history memory+history
    uv run python scripts/reask_feedback.py --experiment tac_monty_hall_ladder --trainees "<R3 config>" --out <dir> --job-report <dir>/report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from plybench.analysis.replay import build_replayer
from plybench.cli.shared import add_training_source_args, build_op, training_from_args
from plybench.configs.training_run import TrainingRun
from plybench.core.game import TurnBasedGame
from plybench.harness.training.training import TrainingHarness
from plybench.player.learnable import LEARNER_CALLS
from plybench.report import Item, Report, ReportFile
from plybench.trackers.llm_call import LLMCall
from plybench.utils.seed import derive_seed
from plyllm import ModelConfig
from show_tac import grade_table

from plyagents.tac.configs.components import Services
from plyagents.tac.critics.build import CRITICS
from plyagents.tac.enums import CriticType
from plyagents.tac.feedback.llm import LLMFeedback, LLMFeedbackParams, _Verdicts, history_block, read_inputs
from plyagents.tac.model.feedback import GRADES, StepFeedback
from plyagents.tac.model.reference import Reference
from plyagents.tac.model.step import TACStep
from plyagents.tac.params import TACParams
from plyagents.tac.rewards.build import REWARDS
from plyagents.tac.utils.calls import Calls

VARIANTS = ("as_run", "caution", "history")
CAUTION = "\n\nThe rules may leave parts of the game unstated. Do not assume they work as in a puzzle or game you know; judge each move by what the games showed."
# the prompt's line after the moves; the history goes before it
OUTCOME = "\n\nOutcome for the agent:"
# each level the one before plus one input; the levels showing the critic's numbers are asked once per critic
LEVELS = {"L0": "none", "L1": "memory", "L2": "memory+delta", "L3": "memory+delta+q", "L4": "memory+delta+q+history", "L5": "memory+delta+q+history+reasoning"}
CRITICS_OF_LEVELS = {"tabular": "tabular", "oracle": "oracle"}
OBFUSCATED = "obfuscated"
# Monty Hall's words and neutral ones, whole sentences first, then words (whole words, any case); afterwards no cue may be left
RENAMES = (
    ("You are a contestant on a game show.", "You take part in a task."),
    ("A car is behind one door, and a goat is behind each of the others.", "A prize is in one box, and each of the others is empty."),
    ("You win the car if your final door hides it.", "You win the prize if your final box holds it."),
    ("which hides a goat", "which is empty"),
    ("car", "prize"),
    ("doors", "boxes"),
    ("door", "box"),
    ("host", "operator"),
)
CUES = ("door", "doors", "host", "car", "goat", "goats", "contestant", "game show", "monty", "hall", "behind")


@dataclass
class Game:
    # one recorded training game: where it is, the agent's steps as the player recorded them, the key, move and solver's verdict per step, the
    # result and the outcome, and its last usable feedback call (None for a game without one: it is replayed, not asked)
    epoch: int
    round: int
    seed: int | None
    steps: list[TACStep]
    keys: list[str]
    moves: list[str]
    optimal: list[bool]
    result: str
    outcome: float
    system: str | None
    prompt: str | None


@dataclass(frozen=True)
class Ask:
    # one answer to ask: under which name it is saved, for which game, the call, and the seed of the game it is asked with
    name: str
    game: Game
    system: str
    prompt: str
    seed: int | None


def load_games(harness: TrainingHarness, run: TrainingRun, services: Services) -> list[Game]:
    # every training game of the run in the order it was played (the order the updates were committed in), each with its last usable feedback call
    op, games = harness.op, []
    replayer = build_replayer(op.registry, run.game)
    shaper = REWARDS.build(params(run).reward, services)
    for epoch in harness.get_results().for_run(run):
        tracker = epoch.train
        if tracker is None:
            continue
        for round_, record in enumerate(tracker.games, start=1):
            if record is None:
                continue
            path = harness.path_builder.learner_dir(harness.experiment, run, epoch.epoch.index) / f"game_{round_}" / LEARNER_CALLS
            calls = [call for call in json.loads(path.read_text()) if call["role"] == "feedback" and not call.get("failure")] if path.exists() else []
            replayed = replayer.replay_steps(record, tracker.i)
            steps = [TACStep.from_dict(step.step.data or {}) for step in replayed]
            result = record.get_result(tracker.i)
            games.append(
                Game(
                    epoch.epoch.index,
                    round_,
                    record.seed,
                    steps,
                    [step.key for step in steps],
                    [step.step.move for step in replayed],
                    [step.is_optimal for step in replayed],
                    result.name.lower(),
                    shaper.outcome(result, steps),
                    calls[-1]["system_message"] if calls else None,
                    calls[-1]["prompt_message"] if calls else None,
                )
            )
    return games


def params(run: TrainingRun) -> TACParams:
    return TACParams.from_string(run.trainee.to_string().partition(":")[2])


def feedback_params(run: TrainingRun) -> LLMFeedbackParams:
    # the run's LLM feedback, its model with options (the recorded calls name the model without them)
    feedback = params(run).feedback.params
    if not isinstance(feedback, LLMFeedbackParams) or feedback.model is None:
        raise SystemExit(f"the run's feedback names no LLM model: {run.trainee.to_string()}")
    return feedback


def history(game: Game, earlier: list[Game]) -> str:
    # the first round's history: per step, the moves played in the same situation in the earlier games that had a feedback call, with their results
    seen = Reference()
    for other in earlier:
        for key, move in zip(other.keys, other.moves):
            seen.add(key, move, other.outcome)
    return history_block(seen.counts, game.keys)


def changed(game: Game, earlier: list[Game], variant: str) -> tuple[str, str]:
    # the system message and the prompt of the recorded call, changed for a variant of the first round
    assert game.system is not None and game.prompt is not None
    if variant == "caution":
        return game.system + CAUTION, game.prompt
    if variant == "history":
        if OUTCOME not in game.prompt:
            raise ValueError(f"epoch {game.epoch}, game {game.round}: the recorded prompt has no outcome line to put the history before")
        head, _, tail = game.prompt.partition(OUTCOME)
        return game.system, head + history(game, earlier) + OUTCOME + tail
    return game.system, game.prompt


async def rebuilt(run: TrainingRun, games: list[Game], inputs: str, critic: str, game: TurnBasedGame, services: Services) -> dict[tuple[int, int], tuple[str, str]]:
    # the calls of the games that had one, built by the feedback model's code with these inputs and this critic; the critic and the history are
    # replayed over every training game in order, each game assessed with the state before its own update, as in training
    feedback_config = feedback_params(run)
    assert feedback_config.model is not None
    feedback = LLMFeedback(feedback_config.model, feedback_config.lesson_chars, LLMFeedbackParams(inputs=inputs).shown)
    judge = CRITICS.build(CRITICS.config(critic), services)
    judge.prepare(game)
    calls = Calls(services.llm, None, lambda call: None)
    built: dict[tuple[int, int], tuple[str, str]] = {}
    reference = Reference()
    for played in games:
        assessments = await judge.assess(played.steps, calls)
        if played.prompt is not None:
            history = {key: reference.before(key) for key in played.keys}
            built[played.epoch, played.round] = (feedback.system(played.steps), feedback.prompt(played.steps, assessments, played.outcome, history))
        update = await judge.propose(played.steps, played.outcome, [StepFeedback()] * len(played.steps), calls)
        if update is not None:
            judge.commit(update)
        for step in played.steps:
            if step.action is not None:
                reference.add(step.key, step.action, played.outcome)
    return built


async def oracle_deltas(games: list[Game], game: TurnBasedGame, services: Services) -> dict[tuple[int, int], list[float]]:
    # per step the TD error under V*: V*(s_t+1) - V*(s_t), the outcome after the last move; its sign is what the classical estimator would use
    oracle = CRITICS.build(CRITICS.config("oracle"), services)
    oracle.prepare(game)
    calls = Calls(services.llm, None, lambda call: None)
    deltas: dict[tuple[int, int], list[float]] = {}
    for played in games:
        values = [assessment.value for assessment in await oracle.assess(played.steps, calls)]
        after = [*values[1:], played.outcome]
        deltas[played.epoch, played.round] = [next_value - value for value, next_value in zip(values, after) if value is not None and next_value is not None]
    return deltas


def obfuscate(text: str) -> str:
    # the call without the known puzzle's words; fails if a cue is left, rather than asking a call that still names it
    for old, new in RENAMES:
        text = re.sub(rf"(?<!\w){re.escape(old)}(?!\w)", lambda match, new=new: new[0].upper() + new[1:] if match.group(0)[0].isupper() else new, text, flags=re.IGNORECASE)
    left = [cue for cue in CUES if re.search(rf"\b{re.escape(cue)}\b", text, flags=re.IGNORECASE)]
    if left:
        raise ValueError(f"the obfuscated call still has {left}: add them to RENAMES")
    return text


def level_names(levels: list[str], critics: list[str], no_q: bool, explicit: list[str] | None = None) -> list[tuple[str, str, str]]:
    # (name, inputs, critic) per level and critic, without duplicates (L3 without q is L2); the levels without the critic's numbers once, with no critic.
    # The explicit sets (--inputs) the same, but named <inputs>_nocritic without a critic, apart from the first round's variants (history)
    names: dict[str, tuple[str, str, str]] = {}
    sets = [("+".join(name for name in LEVELS[level].split("+") if not (no_q and name == "q")), "") for level in levels]
    for text, suffix in sets + [(text, "_nocritic") for text in explicit or []]:
        inputs = read_inputs(text)
        shown = LLMFeedbackParams(inputs=inputs).shown
        for critic in critics if shown & {"delta", "q"} else ["none"]:
            name = f"{inputs}{suffix}" if critic == "none" else f"{inputs}_{critic}"
            names.setdefault(name, (name, inputs, "none" if critic == "none" else CRITICS_OF_LEVELS[critic]))
    return list(names.values())


async def plan(args: argparse.Namespace, run: TrainingRun, every: list[Game], game: TurnBasedGame, services: Services) -> tuple[list[Ask], list[str]]:
    # every answer asked, before any call, and the problems that stop the run: a call that is not rebuilt exactly is a bug in the rebuild
    asked = [played for played in every if played.prompt is not None and (args.epochs is None or played.epoch in args.epochs)]
    with_calls = [played for played in every if played.prompt is not None]
    calls: dict[str, dict[tuple[int, int], tuple[str, str]]] = {}
    for variant in args.variants:
        calls[variant] = {(played.epoch, played.round): changed(played, with_calls[: with_calls.index(played)], variant) for played in asked}
    for name, inputs, critic in level_names(args.levels, args.critics, args.no_q, args.inputs):
        calls[name] = await rebuilt(run, every, inputs, critic, game, services)
    if args.obfuscated:
        plain = await rebuilt(run, every, "none", "none", game, services)
        calls[OBFUSCATED] = {where: (obfuscate(system), obfuscate(prompt)) for where, (system, prompt) in plain.items()}
    problems = await rebuild_problems(run, every, game, services) if args.levels or args.inputs or args.obfuscated else []
    asks = [
        Ask(name if sample == 0 else f"{name}_seed{sample}", played, *calls[name][played.epoch, played.round], seed_of(played, sample))
        for name in calls
        for sample in args.sample_seeds
        for played in asked
    ]
    return asks, problems


async def rebuild_problems(run: TrainingRun, every: list[Game], game: TurnBasedGame, services: Services) -> list[str]:
    # the run's own inputs and critic must give back its recorded calls, else the levels are not built as the training builds them
    run_params = params(run)
    if run_params.critic.kind in (CriticType.LLM_V, CriticType.LLM_Q):
        return []
    own = await rebuilt(run, every, feedback_params(run).inputs, run_params.critic.to_string(), game, services)
    differ = [where for where, call in own.items() if call != next((played.system, played.prompt) for played in every if (played.epoch, played.round) == where)]
    if not differ:
        return []
    epoch, round_ = differ[0]
    return [f"the rebuilt call differs from the recorded one in {len(differ)} of {len(own)} games (first: epoch {epoch}, game {round_}); see --dry-run"]


def seed_of(game: Game, sample: int) -> int | None:
    # 0: the game's own seed, which reproduces the recorded answer; another: a seed derived from it
    if sample == 0 or game.seed is None:
        return game.seed
    return derive_seed(game.seed, "reask", sample)


def answer_path(out: Path, name: str, game: Game) -> Path:
    return out / name / f"epoch_{game.epoch}_game_{game.round}.json"


async def answer(calls: Calls, model: ModelConfig, ask: Ask, deltas: list[float], path: Path, records: list[LLMCall]) -> list[str]:
    # one answer, saved as soon as it arrives (a stopped run keeps it), with the keys, the solver's verdicts and V*'s TD errors the report needs;
    # the new grade of every step, "none" for a step the answer leaves out
    reply = await calls.structured("feedback", model, ask.system, ask.prompt, _Verdicts)
    by_step: dict[int, str] = {}
    for verdict in reply.verdicts:
        by_step.setdefault(verdict.step, verdict.grade)
    graded = [by_step.get(number, "none") for number in range(1, len(ask.game.keys) + 1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    game = ask.game
    saved = {"epoch": game.epoch, "round": game.round, "variant": ask.name, "keys": game.keys, "optimal": game.optimal, "delta": deltas, "grades": graded}
    path.write_text(json.dumps(saved | {"calls": [record.to_dict() for record in records]}, indent=1))
    return graded


def agree(optimal: bool, grade: str) -> bool:
    # as show_tac.py's feedback section: good or strong on an optimal move, mistake or dubious on another
    return grade in (("good", "strong") if optimal else ("mistake", "dubious"))


def agree_delta(delta: float, grade: str) -> bool:
    # with the sign of V*'s TD error for this game: good or strong where it rose, mistake or dubious where it fell, neutral where it stayed
    if abs(delta) < 1e-9:
        return grade == "neutral"
    return grade in (("good", "strong") if delta > 0 else ("mistake", "dubious"))


def report(out: Path, planned: dict[str, int]) -> Report:
    # the report of the names this command asks (planned: answers per name): one item per name, the agreement with the solver (and with V*'s TD error)
    # per name, then per name, situation and kind of move the grades; answers of other names in the folder are left out
    graded: dict[str, list[tuple[str, bool, str, bool, float | None]]] = defaultdict(list)
    answers: dict[str, int] = defaultdict(int)
    for path in sorted(out.glob("*/epoch_*_game_*.json")):
        if path.parent.name not in planned:
            continue
        saved = json.loads(path.read_text())
        answers[saved["variant"]] += 1
        deltas = saved.get("delta") or [None] * len(saved["keys"])
        for index, (key, optimal, grade, delta) in enumerate(zip(saved["keys"], saved["optimal"], saved["grades"], deltas)):
            graded[saved["variant"]].append((key, optimal, grade, index > 0, delta))

    def share(steps: list[tuple[str, bool, str, bool, float | None]]) -> float | None:
        return sum(agree(optimal, grade) for _, optimal, grade, _, _ in steps) / len(steps) if steps else None

    def share_delta(steps: list[tuple[str, bool, str, bool, float | None]]) -> float | None:
        known = [(delta, grade) for _, _, grade, _, delta in steps if delta is not None]
        return sum(agree_delta(delta, grade) for delta, grade in known) / len(known) if known else None

    names = sorted(answers, key=lambda name: (name not in VARIANTS, VARIANTS.index(name) if name in VARIANTS else 0, name))
    summary = [
        {
            "item": name,
            "answers": answers[name],
            "optimal": share([step for step in graded[name] if step[1]]),
            "other": share([step for step in graded[name] if not step[1]]),
            "all": share(graded[name]),
            "later": share([step for step in graded[name] if step[3]]),
            "delta": share_delta(graded[name]),
        }
        for name in names
    ]
    rows: dict[tuple[str, str, bool], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for name in names:
        for key, optimal, grade, _, _ in graded[name]:
            rows[name, key, optimal][grade] += 1
    detail = [
        {"item": name, "situation": key, "move": "optimal" if optimal else "not optimal", "n": sum(counts.values())}
        | {grade: counts.get(grade, 0) for grade in (*GRADES, "none")}
        | {"agree": sum(count for grade, count in counts.items() if agree(optimal, grade)) / sum(counts.values())}
        for (name, key, optimal), counts in rows.items()
    ]
    rate = {"rate": True}
    columns = (("optimal", "optimal moves"), ("other", "other moves"), ("all", "all moves"), ("later", "all but each game's first"), ("delta", "with V*'s TD error"))
    result = Report(unit="answers", items=[Item(name, {"variant": name}, total, answers.get(name, 0)) for name, total in planned.items()])
    result.add_table(
        "agreement with the solver, per variant and level",
        [{"key": "item", "label": "variant", "name": True}, {"key": "answers", "label": "answers", "digits": 0}] + [{"key": key, "label": label} | rate for key, label in columns],
        summary,
        note="share of graded moves where the feedback agrees: good or strong on an optimal move, mistake or dubious on another; the last column against the"
        " sign of V*'s TD error in the same game (neutral where it is 0), what the classical estimator would use",
    )
    result.add_table(
        "grades per situation",
        [{"key": "item", "label": "variant", "name": True}, {"key": "situation", "label": "situation"}, {"key": "move", "label": "move"}]
        + [{"key": key, "label": key, "digits": 0} for key in ("n", *GRADES, "none")]
        + [{"key": "agree", "label": "agree"} | rate],
        detail,
    )
    return result


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_training_source_args(parser)
    parser.add_argument("--out", type=Path, required=True, help="folder of the answers, one file each (relative to the repository: analysis/...)")
    parser.add_argument(
        "--variants",
        nargs="*",
        choices=VARIANTS,
        default=["as_run", "history"],
        help="variants of the first round, each over the same games (default as_run history; none: --variants alone)",
    )
    parser.add_argument("--levels", nargs="*", choices=list(LEVELS), default=[], help="levels of the information ladder (default none)")
    parser.add_argument("--critics", nargs="+", choices=list(CRITICS_OF_LEVELS), default=list(CRITICS_OF_LEVELS), help="critics of the levels from L2 (default both)")
    parser.add_argument("--no-q", action="store_true", help="leave the critic's values of the other moves out of the levels (L3 is then L2)")
    parser.add_argument("--obfuscated", action="store_true", help="also ask L0 with the game renamed")
    parser.add_argument(
        "--inputs", nargs="+", default=[], help="explicit sets of the feedback's inputs (e.g. memory+history), asked as a level; without delta and q with no critic"
    )
    parser.add_argument("--sample-seeds", nargs="+", type=int, default=[0], help="seeds to ask each call with: 0 the game's own, others derived from it (default 0)")
    parser.add_argument("--epochs", nargs="+", type=int, help="epochs whose training games are asked again (default: all); the history and the critic still see every earlier game")
    parser.add_argument("--concurrency", type=int, default=10, help="requests in flight per provider (default 10)")
    parser.add_argument(
        "--check", action="store_true", help="print what would be asked (answers planned and left, the model, the outputs) and check the rebuild, then exit; no calls"
    )
    parser.add_argument("--json", action="store_true", help="with --check: print the plan as JSON")
    parser.add_argument("--job-report", type=Path, help="keep a report (plybench.report) of the names this command asks while it works, in this JSON file")
    parser.add_argument("--dry-run", action="store_true", help="print the call of the last selected game per name, no calls")
    parser.add_argument("--max-chars", type=int, default=70, help="cut each key to this length (default 70)")
    args = parser.parse_args()

    if not (args.variants or args.levels or args.inputs or args.obfuscated):
        raise SystemExit("nothing to ask: name --variants, --levels, --inputs or --obfuscated")
    op = build_op(concurrency=args.concurrency)
    services = Services(op.llm, op.registry.solvable)
    harness = training_from_args(op, args)
    runs = harness.get_results().runs
    if len(runs) != 1:
        raise SystemExit(f"select exactly one run with --experiment and --trainees (selected {len(runs)})")
    run = runs[0]
    feedback_model = feedback_params(run).model
    assert feedback_model is not None
    model = feedback_model.config
    engine = op.registry.build_engine(run.game)
    engine.reset()
    every = load_games(harness, run, services)
    asks, problems = await plan(args, run, every, engine.game, services)
    if not asks:
        problems.append(f"no training game of {run.trainee.to_string()} has a recorded feedback call")
    left = [ask for ask in asks if not answer_path(args.out, ask.name, ask.game).exists()]

    if args.check:
        plan_json = {"planned": len(asks), "remaining": len(left), "unit": "answers", "models": [f"{model.provider.value}:{model.model_name}"], "outputs": [str(args.out)]}
        print(
            json.dumps(plan_json | {"problems": problems})
            if args.json
            else f"{len(left)} of {len(asks)} answers to ask, model {model.to_string()}, into {args.out}; problems: {problems or 'none'}"
        )
        raise SystemExit(1 if problems else 0)
    if problems:
        raise SystemExit("; ".join(problems))
    names = list(dict.fromkeys(ask.name for ask in asks))
    print(
        f"{run.trainee.to_string()}\n{len(every)} training games, {len({id(ask.game) for ask in asks})} with a recorded feedback call; {len(left)} of {len(asks)} answers to ask",
        flush=True,
    )
    if args.dry_run:
        for name in names:
            ask = [ask for ask in asks if ask.name == name][-1]
            print(f"\n== {name}: epoch {ask.game.epoch}, game {ask.game.round}\n-- system\n{ask.system}\n-- prompt\n{ask.prompt}")
        return

    deltas = await oracle_deltas(every, engine.game, services)
    done, failed = len(asks) - len(left), 0
    # the report's items: the names with answers left now, each with all its answers (a name saved in full is not shown)
    open_names = {ask.name for ask in left}
    planned = {name: sum(1 for ask in asks if ask.name == name) for name in names if name in open_names}
    report_file = ReportFile(args.job_report, lambda: report(args.out, planned)) if args.job_report is not None else None
    if report_file is not None:
        report_file.write()

    async def one(ask: Ask) -> None:
        nonlocal done, failed
        records: list[LLMCall] = []
        try:
            await answer(Calls(op.llm, ask.seed, records.append), model, ask, deltas[ask.game.epoch, ask.game.round], answer_path(args.out, ask.name, ask.game), records)
        except Exception as error:
            # left unsaved, so a run again asks it again; the run ends with an error once the others are done
            failed += 1
            print(f"warning: {ask.name}, epoch {ask.game.epoch}, game {ask.game.round}: {type(error).__name__}: {error}", flush=True)
            return
        done += 1
        print(f"progress: {done}/{len(asks)}", flush=True)
        if report_file is not None:
            report_file.changed()

    completed = False
    try:
        await asyncio.gather(*(one(ask) for ask in left))
        completed = True
    finally:
        if report_file is not None:
            report_file.close(finished=completed and failed == 0)
    for name in names:
        print(f"\n== {name}: grades against the solver (agree: good or strong on an optimal move, mistake or dubious on another)")
        counts: dict[str, dict[bool, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
        for ask in asks:
            path = answer_path(args.out, name, ask.game)
            if ask.name == name and path.exists():
                for key, optimal, grade in zip(ask.game.keys, ask.game.optimal, json.loads(path.read_text())["grades"]):
                    counts[key][optimal][grade] += 1
        grade_table(counts, args.max_chars)
        graded = [(optimal, grade) for by_optimal in counts.values() for optimal, counter in by_optimal.items() for grade in counter.elements()]
        print(f"   agree on all moves: {sum(agree(*pair) for pair in graded)} of {len(graded)}")
    if failed:
        raise SystemExit(f"{failed} answers failed; run the same command again to ask them")


if __name__ == "__main__":
    asyncio.run(main())

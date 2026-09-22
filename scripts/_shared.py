"""Shared helpers for the repo-local scripts (not part of the installed library).

Everything operates relative to the current working directory, which is the path convention plybench
itself uses: `experiments/training/` for the definitions, `results/training/` for the transcripts and
checkpoints.
"""

from __future__ import annotations

import argparse

from plybench.analysis.training import TrainingAnalysis
from plybench.app import PlyBench
from plybench.common.enums import MetricName
from plybench.harness.training.results import TrainingResults
from plybench.harness.training.training import TrainingHarness
from plybench.llm import ModelLimits, Provider
from plybench.registry import Registry

from game_agents.reasoning_bank.register import register_agents

# Per-model quotas for *this* account, read off the provider's console. They live here rather than in
# the library because they are account-specific; a model absent from this map is governed only by the
# provider-wide semaphore.
LIMITS: dict[Provider, dict[str, ModelLimits]] = {}


def build_op(concurrency: int | None = None) -> PlyBench:
    op = PlyBench(concurrency=concurrency)
    available = set(op.llm.available_providers)
    for provider, models in LIMITS.items():
        if provider in available:
            for model_name, limits in models.items():
                op.llm.set_model_limits(provider, model_name, limits)
    return register_agents(op)


def add_training_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", help="experiment name under experiments/training/<name>.json")
    parser.add_argument("--name", help="experiment name for an inline training run")
    parser.add_argument("--games", nargs="+", metavar="CONFIG", help="game config strings, e.g. connect_four:")
    parser.add_argument("--trainees", nargs="+", metavar="CONFIG", help="learnable player config strings")
    parser.add_argument("--trainers", nargs="+", metavar="CONFIG", help="opponent config strings for the training phases")
    parser.add_argument("--testers", nargs="+", metavar="CONFIG", help="opponent config strings for the evaluation phases")
    parser.add_argument("--schedules", nargs="+", metavar="CONFIG", help="schedules, e.g. num_epochs=5,num_training_games=20,num_test_games=50")
    parser.add_argument("--replicates", type=int, default=1, help="independent repeats of each run (inline only; default 1)")


def training_from_args(op: PlyBench, args: argparse.Namespace) -> TrainingHarness:
    if args.experiment:
        return TrainingHarness.load_experiment(
            op,
            args.experiment,
            game_override=args.games,
            player_override=args.trainees,
            trainer_override=args.trainers,
            tester_override=args.testers,
            training_override=args.schedules,
        )
    if not (args.games and args.trainees and args.trainers and args.testers and args.schedules):
        raise SystemExit("provide --experiment, or all of --games / --trainees / --trainers / --testers / --schedules (+ optional --replicates)")
    return TrainingHarness(args.name or "training", op, args.games, args.trainees, args.trainers, args.testers, args.schedules, args.replicates)


def report_curves(results: TrainingResults, registry: Registry, tester_configs: list[str], metric: MetricName) -> None:
    analysis = TrainingAnalysis(results, registry)
    for tester_config in tester_configs:
        tester = registry.player_config(tester_config)
        for run in results.runs:
            curve = analysis.curve(run, tester)
            print(f"\n{run.game.path}: {run.trainee.path} vs {run.trainer.path} rep{run.replicate} - {metric.value} vs {tester_config}")
            for point in curve.points:
                bundle = point.bundle(metric)
                interval = f" [{bundle.wilson.lower:.2f}, {bundle.wilson.upper:.2f}]" if bundle.wilson is not None else ""
                print(f"  epoch {point.epoch:>2}: {bundle.value:.3f}{interval}  (n={bundle.n})")
            print(f"  delta: {curve.delta(metric):+.3f}")

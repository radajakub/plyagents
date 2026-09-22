"""Train an agent on a game: every epoch plays a training phase against the trainer, checkpoints what
it learned, and evaluates that frozen checkpoint against every tester. Resumable -- completed epochs
and rounds are skipped, and an interrupted epoch replays its recorded games through the learner.

Examples:
    uv run python scripts/train.py --experiment smoke
    uv run python scripts/train.py --experiment connect_four --concurrency 4
    uv run python scripts/train.py --experiment connect_four --notify
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import clankers  # noqa: E402
from _shared import add_training_source_args, build_op, report_curves, training_from_args  # noqa: E402
from clankers.core.models import describe  # noqa: E402
from plybench.callbacks.console_callbacks import console_training_callbacks  # noqa: E402
from plybench.callbacks.notification_callbacks import notification_training_callbacks  # noqa: E402
from plybench.callbacks.training_callbacks import TrainingCallbacks  # noqa: E402
from plybench.common.enums import MetricName  # noqa: E402
from plybench.llm import DEFAULT_CONCURRENCY  # noqa: E402


def warn_if_unconfigured() -> None:
    # clankers builds its backend lazily and only warns when a send fails, so a missing
    # NTFY_URL/NTFY_TOPIC would otherwise go unnoticed until the first epoch ends
    try:
        _ = clankers.default().backend
    except ValueError as error:
        print(f"warning: --notify set but notifications are not configured ({error}); they will be skipped")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_training_source_args(parser)
    parser.add_argument("--sync", action="store_true", help="run the runs sequentially instead of concurrently (epochs within a run are always sequential)")
    parser.add_argument("--concurrency", type=int, help=f"max in-flight requests per provider (default: {DEFAULT_CONCURRENCY})")
    parser.add_argument("--rounds-concurrency", type=int, help="max concurrent rounds per evaluation phase (default: --concurrency)")
    parser.add_argument("--runs-concurrency", type=int, help="max runs in flight at once (default: all of them)")
    parser.add_argument("--metric", type=str, default=MetricName.WIN_RATE.value, choices=[name.value for name in MetricName], help="metric to print the learning curve on")
    parser.add_argument("--notify", action="store_true", help="push a notification on each epoch end, a final summary, and on failure (needs NTFY_URL and NTFY_TOPIC)")
    args = parser.parse_args()

    concurrency = args.concurrency if args.concurrency is not None else DEFAULT_CONCURRENCY
    rounds_concurrency = args.rounds_concurrency if args.rounds_concurrency is not None else concurrency

    op = build_op(concurrency=concurrency)
    harness = training_from_args(op, args)

    callbacks = console_training_callbacks()
    notify: AbstractContextManager[object] = nullcontext()
    if args.notify:
        warn_if_unconfigured()
        notification_callbacks, progress = notification_training_callbacks(harness.experiment)
        callbacks = TrainingCallbacks.combine(callbacks, notification_callbacks)
        # clankers reports the run's start, duration and outcome; the builders add how far it got
        label = f"[{harness.experiment}] training"
        notify = clankers.Engage(
            label,
            success=lambda: f"{label} finished: {progress()}",
            failure=lambda error: f"{label} crashed at {progress()}: {describe(error)}",
        )

    with notify:
        results = asyncio.run(harness.run(sync=args.sync, concurrency=rounds_concurrency, runs_concurrency=args.runs_concurrency, training_callbacks=callbacks))

    metric = MetricName.from_value(args.metric)
    if metric is None:
        parser.error(f"Invalid metric: {args.metric}")

    report_curves(results, op.registry, harness.tester_configs, metric)
    print(f"\nresults under results/training/{harness.experiment}/")


if __name__ == "__main__":
    main()

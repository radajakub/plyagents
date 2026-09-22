"""Print the learning curves of an experiment that has already been run, without replaying anything.

Examples:
    uv run python scripts/analyse.py --experiment connect_four
    uv run python scripts/analyse.py --experiment connect_four --metric output_tokens_per_game
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _shared import add_training_source_args, build_op, report_curves, training_from_args  # noqa: E402
from plybench.common.enums import MetricName  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_training_source_args(parser)
    parser.add_argument("--metric", type=str, default=MetricName.WIN_RATE.value, choices=[name.value for name in MetricName], help="metric to print the learning curve on")
    args = parser.parse_args()

    metric = MetricName.from_value(args.metric)
    if metric is None:
        parser.error(f"Invalid metric: {args.metric}")

    op = build_op()
    harness = training_from_args(op, args)
    report_curves(harness.get_results(), op.registry, harness.tester_configs, metric)


if __name__ == "__main__":
    main()

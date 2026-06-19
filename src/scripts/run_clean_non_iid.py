"""Run the clean Non-IID CICIoT2023 experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.main import run_training_pipeline
from src.scripts.common import add_common_experiment_args, build_framework_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_experiment_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_framework_config(
        args,
        data_mode="benign",
        checkpoint_algorithm="FFNN_Clean_NonIID",
    )
    run_training_pipeline(
        config=config,
        num_rounds=args.num_rounds,
        perform_evaluation=args.perform_evaluation,
    )


if __name__ == "__main__":
    main()

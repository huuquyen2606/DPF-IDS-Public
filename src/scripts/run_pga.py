"""Run the PGA-based untargeted model-poisoning experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.main import run_training_pipeline
from src.scripts.common import add_common_experiment_args, build_framework_config, require_poison_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_experiment_args(parser)
    parser.add_argument("--pga-gamma", type=float, default=1e4)
    parser.add_argument("--pga-projection-radius", type=float, default=10.0)
    parser.add_argument("--pga-tau-mode", choices=("fixed", "self_benign"), default="fixed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_poison_index(args, "PGA")
    config = build_framework_config(
        args,
        data_mode="data_poison",
        checkpoint_algorithm="FFNN_PGA",
        enable_pga_attack=True,
    )
    config.pga_gamma = args.pga_gamma
    config.pga_projection_radius = args.pga_projection_radius
    config.pga_tau_mode = args.pga_tau_mode
    run_training_pipeline(
        config=config,
        num_rounds=args.num_rounds,
        perform_evaluation=args.perform_evaluation,
    )


if __name__ == "__main__":
    main()

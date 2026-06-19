"""Run the BoTPA targeted poisoning experiment."""

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
    parser.add_argument("--botpa-source-class", type=int, default=2)
    parser.add_argument("--botpa-target-class", type=int, default=0)
    parser.add_argument("--botpa-intermediate-classes", type=int, default=3)
    parser.add_argument("--botpa-gamma", type=float, default=1.0)
    parser.add_argument("--evaluate-asr-each-round", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_poison_index(args, "BoTPA")
    config = build_framework_config(
        args,
        data_mode="data_poison",
        checkpoint_algorithm="FFNN_BoTPA",
        enable_botpa_attack=True,
    )
    config.botpa_source_class = args.botpa_source_class
    config.botpa_target_class = args.botpa_target_class
    config.botpa_num_intermediate_classes = args.botpa_intermediate_classes
    config.botpa_gamma = args.botpa_gamma
    config.evaluate_botpa_asr_each_round = args.evaluate_asr_each_round
    run_training_pipeline(
        config=config,
        num_rounds=args.num_rounds,
        perform_evaluation=args.perform_evaluation,
    )


if __name__ == "__main__":
    main()

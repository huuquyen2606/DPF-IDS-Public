"""Shared CLI helpers for DPF-IDS experiment scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.main import FrameworkConfig


def add_common_experiment_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--train-data-dir", required=True, help="Directory containing collaborator client_*.pt files.")
    parser.add_argument("--test-data-path", default="", help="Path to the shared test.pt file.")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory where checkpoints and logs will be written.")
    parser.add_argument("--num-collaborators", type=int, choices=(200, 500), default=200)
    parser.add_argument("--num-rounds", type=int, default=11, help="Scheduled round count used by the paper setup.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perform-evaluation", action="store_true", help="Evaluate detected-benign client models during training.")
    parser.add_argument("--true-poison-index-path", default="", help="Optional path to a text file containing poisoned collaborator IDs.")
    parser.add_argument("--true-poison-index-base", default="auto")
    return parser


def build_framework_config(
    args: argparse.Namespace,
    *,
    data_mode: str,
    checkpoint_algorithm: str,
    enable_pga_attack: bool = False,
    enable_botpa_attack: bool = False,
) -> FrameworkConfig:
    return FrameworkConfig(
        data_mode=data_mode,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.local_epochs,
        num_clients=args.num_collaborators,
        checkpoint_dir=str(Path(args.checkpoint_dir)),
        checkpoint_algorithm=checkpoint_algorithm,
        data_train=str(Path(args.train_data_dir)),
        data_test=str(Path(args.test_data_path)) if args.test_data_path else "",
        true_poison_index_path=str(Path(args.true_poison_index_path)) if args.true_poison_index_path else "",
        true_poison_index_base=args.true_poison_index_base,
        seed=args.seed,
        enable_pga_attack=enable_pga_attack,
        enable_botpa_attack=enable_botpa_attack,
    )


def require_poison_index(args: argparse.Namespace, attack_name: str) -> None:
    if not args.true_poison_index_path:
        raise ValueError(f"{attack_name} requires --true-poison-index-path to identify compromised collaborators.")

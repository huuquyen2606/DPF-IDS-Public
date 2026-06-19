"""Analyze label distributions for clean and targeted label-flipping folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.attacks.label_flipping import PLACEHOLDER_OUTPUT_DIR, PLACEHOLDER_SOURCE_DIR, PLACEHOLDER_TARGETED_DIR, analyze_clean_targeted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", type=Path, required=True, help=f"Clean client .pt directory, e.g. {PLACEHOLDER_SOURCE_DIR}.")
    parser.add_argument("--targeted-dir", type=Path, required=True, help=f"Targeted poisoned client .pt directory, e.g. {PLACEHOLDER_TARGETED_DIR}.")
    parser.add_argument("--output-dir", type=Path, required=True, help=f"Output analysis directory, e.g. {PLACEHOLDER_OUTPUT_DIR}.")
    parser.add_argument("--num-classes", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_classes <= 0:
        raise ValueError("--num-classes must be > 0")
    analyze_clean_targeted(
        clean_dir=args.clean_dir,
        targeted_dir=args.targeted_dir,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
    )


if __name__ == "__main__":
    main()

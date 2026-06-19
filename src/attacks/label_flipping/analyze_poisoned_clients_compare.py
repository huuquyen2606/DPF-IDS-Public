"""Compare poisoned collaborators across clean, targeted, and untargeted folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.attacks.label_flipping import (
    PLACEHOLDER_OUTPUT_DIR,
    PLACEHOLDER_POISONED_LIST,
    PLACEHOLDER_SOURCE_DIR,
    PLACEHOLDER_TARGETED_DIR,
    PLACEHOLDER_UNTARGETED_DIR,
    analyze_poisoned_clients_compare,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poisoned-list", type=Path, required=True, help=f"Text file with poisoned client IDs, e.g. {PLACEHOLDER_POISONED_LIST}.")
    parser.add_argument("--clean-dir", type=Path, required=True, help=f"Clean client .pt directory, e.g. {PLACEHOLDER_SOURCE_DIR}.")
    parser.add_argument("--targeted-dir", type=Path, required=True, help=f"Targeted poisoned client .pt directory, e.g. {PLACEHOLDER_TARGETED_DIR}.")
    parser.add_argument("--untargeted-dir", type=Path, required=True, help=f"Untargeted poisoned client .pt directory, e.g. {PLACEHOLDER_UNTARGETED_DIR}.")
    parser.add_argument("--output-dir", type=Path, required=True, help=f"Output analysis directory, e.g. {PLACEHOLDER_OUTPUT_DIR}.")
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_classes <= 0:
        raise ValueError("--num-classes must be > 0")
    analyze_poisoned_clients_compare(
        poisoned_list=args.poisoned_list,
        clean_dir=args.clean_dir,
        targeted_dir=args.targeted_dir,
        untargeted_dir=args.untargeted_dir,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()

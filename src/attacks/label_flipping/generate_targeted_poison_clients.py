"""Generate targeted label-flipping collaborator files.

Example:
    python src/attacks/label_flipping/generate_targeted_poison_clients.py \
        --case-name Targeted_Attack_200c_20p \
        --source-dir /path/to/processed/CICIoT2023/Clients_200_NonIID_pt \
        --poisoned-list /path/to/poisoned_200c_20p.txt \
        --output-dir /path/to/processed/CICIoT2023/Targeted_Attack_200c_20p
"""

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
    generate_targeted_clients_for_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate targeted label-flipping poisoned client .pt files.")
    parser.add_argument("--case-name", required=True, help="Experiment name, for example Targeted_Attack_200c_20p.")
    parser.add_argument("--source-dir", type=Path, required=True, help=f"Clean client .pt directory, e.g. {PLACEHOLDER_SOURCE_DIR}.")
    parser.add_argument("--poisoned-list", type=Path, required=True, help=f"Text file with poisoned client IDs, e.g. {PLACEHOLDER_POISONED_LIST}.")
    parser.add_argument("--output-dir", type=Path, required=True, help=f"Output directory, e.g. {PLACEHOLDER_OUTPUT_DIR}.")
    parser.add_argument("--target-class", type=int, default=0, help="Target label used for majority-class flipping.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_targeted_clients_for_case(
        case_name=args.case_name,
        source_dir=args.source_dir,
        poisoned_list_path=args.poisoned_list,
        output_dir=args.output_dir,
        target_class=args.target_class,
    )


if __name__ == "__main__":
    main()

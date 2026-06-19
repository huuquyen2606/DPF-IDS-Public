"""Label-flipping data-poisoning utilities."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import pandas as pd
import torch

from ..data.dataset import _safe_torch_load, unpack_xy


def flip_label_value(label: int, mapping: Mapping[int, int] | None = None, num_classes: int | None = None) -> int:
    label = int(label)
    if mapping is not None:
        return int(mapping.get(label, label))
    if num_classes is None:
        raise ValueError("num_classes is required when mapping is not provided")
    return int((label + 1) % int(num_classes))


def flip_labels_tensor(labels: torch.Tensor, mapping: Mapping[int, int] | None = None, num_classes: int | None = None) -> torch.Tensor:
    flipped = labels.detach().clone().long().view(-1)
    for idx, value in enumerate(flipped.tolist()):
        flipped[idx] = flip_label_value(value, mapping=mapping, num_classes=num_classes)
    return flipped


def poison_dataframe_labels(
    df: pd.DataFrame,
    mapping: Mapping[int, int] | None = None,
    num_classes: int | None = None,
    label_col: str = "Label",
    poison_fraction: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    if label_col not in df.columns:
        raise KeyError(f"Missing label column: {label_col}")
    if not 0.0 <= float(poison_fraction) <= 1.0:
        raise ValueError("poison_fraction must be in [0, 1]")

    poisoned = df.copy()
    indices = poisoned.sample(frac=float(poison_fraction), random_state=int(seed)).index if poison_fraction < 1.0 else poisoned.index
    poisoned.loc[indices, label_col] = poisoned.loc[indices, label_col].map(
        lambda value: flip_label_value(value, mapping=mapping, num_classes=num_classes)
    )
    return poisoned


def poison_csv_file(
    input_path: str,
    output_path: str,
    mapping: Mapping[int, int] | None = None,
    num_classes: int | None = None,
    label_col: str = "Label",
    poison_fraction: float = 1.0,
    seed: int = 42,
) -> str:
    df = pd.read_csv(input_path)
    poisoned = poison_dataframe_labels(
        df,
        mapping=mapping,
        num_classes=num_classes,
        label_col=label_col,
        poison_fraction=poison_fraction,
        seed=seed,
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    poisoned.to_csv(output_path, index=False)
    return output_path


def poison_pt_file(
    input_path: str,
    output_path: str,
    mapping: Mapping[int, int] | None = None,
    num_classes: int | None = None,
) -> str:
    X, y = unpack_xy(_safe_torch_load(input_path, map_location="cpu"))
    poisoned_y = flip_labels_tensor(y, mapping=mapping, num_classes=num_classes)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save((X, poisoned_y), output_path)
    return output_path


def poison_selected_clients(
    input_dir: str,
    output_dir: str,
    client_ids: Sequence[int],
    mapping: Mapping[int, int] | None = None,
    num_classes: int | None = None,
) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    client_set = {int(cid) for cid in client_ids}
    written = []
    for name in sorted(os.listdir(input_dir)):
        in_path = os.path.join(input_dir, name)
        out_path = os.path.join(output_dir, name)
        stem_digits = "".join(ch for ch in os.path.splitext(name)[0] if ch.isdigit())
        cid = int(stem_digits) if stem_digits else None
        if cid in client_set and name.lower().endswith(".pt"):
            written.append(poison_pt_file(in_path, out_path, mapping=mapping, num_classes=num_classes))
        elif cid in client_set and name.lower().endswith(".csv"):
            written.append(poison_csv_file(in_path, out_path, mapping=mapping, num_classes=num_classes))
        elif os.path.isfile(in_path):
            if name.lower().endswith(".pt"):
                X, y = unpack_xy(_safe_torch_load(in_path, map_location="cpu"))
                torch.save((X, y), out_path)
            elif name.lower().endswith(".csv"):
                pd.read_csv(in_path).to_csv(out_path, index=False)
            else:
                continue
            written.append(out_path)
    return written

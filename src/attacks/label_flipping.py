"""Label-flipping data-poisoning utilities.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import TensorDataset

from ..data.dataset import _safe_torch_load, unpack_xy

PLACEHOLDER_SOURCE_DIR = Path("/path/to/processed/CICIoT2023/client_pt_files")
PLACEHOLDER_TARGETED_DIR = Path("/path/to/processed/CICIoT2023/label_flipping_targeted")
PLACEHOLDER_UNTARGETED_DIR = Path("/path/to/processed/CICIoT2023/label_flipping_untargeted")
PLACEHOLDER_POISONED_LIST = Path("/path/to/poisoned_collaborators.txt")
PLACEHOLDER_OUTPUT_DIR = Path("/path/to/output/label_flipping_analysis")

CLIENT_PATTERN = re.compile(r"^client[_\-]?0*(\d+)\.pt$")
LABEL_KEYS = {"y", "label", "labels", "target", "targets"}


def safe_torch_load(path: str | Path) -> Any:
    """Load a trusted local PyTorch file across PyTorch versions."""
    return _safe_torch_load(str(path), map_location="cpu")


def parse_client_id(path: str | Path) -> int | None:
    """Extract the client id from names such as client_1.pt or client_001.pt."""
    match = CLIENT_PATTERN.match(Path(path).name)
    if match is None:
        return None
    return int(match.group(1))


def extract_client_id(path: str | Path) -> int:
    client_id = parse_client_id(path)
    if client_id is None:
        raise ValueError(f"Invalid client filename: {path}")
    return client_id


def read_poisoned_client_ids(list_path: str | Path) -> list[int]:
    """Read integer collaborator ids from a text file.

    The file may use whitespace, comma, semicolon, or newline separators.
    Non-integer tokens are ignored.
    """
    path = Path(list_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing poisoned-client list: {path}")

    client_ids: list[int] = []
    for token in re.split(r"[\s,;]+", path.read_text(encoding="utf-8").strip()):
        if not token:
            continue
        try:
            client_ids.append(int(token))
        except ValueError:
            continue

    if not client_ids:
        raise ValueError(f"No valid collaborator ids found in: {path}")
    return sorted(set(client_ids))


def load_client_file_map(source_dir: str | Path) -> dict[int, Path]:
    """Map collaborator id to its .pt file path."""
    folder = Path(source_dir)
    client_map: dict[int, Path] = {}
    for pt_path in sorted(folder.glob("client*.pt")):
        client_id = parse_client_id(pt_path)
        if client_id is not None:
            client_map[client_id] = pt_path

    if not client_map:
        raise FileNotFoundError(f"No client .pt files found in: {folder}")
    return client_map


def resolve_client_file(folder: str | Path, client_id: int) -> Path:
    """Resolve a collaborator file with flexible zero padding."""
    folder = Path(folder)
    candidates = [
        folder / f"client_{client_id}.pt",
        folder / f"client_{client_id:02d}.pt",
        folder / f"client_{client_id:03d}.pt",
        folder / f"client_{client_id:04d}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    pattern = re.compile(rf"^client[_\-]?0*{int(client_id)}\.pt$")
    for pt_path in folder.glob("client*.pt"):
        if pattern.match(pt_path.name):
            return pt_path
    raise FileNotFoundError(f"Could not find client_{client_id}.pt in {folder}")


def extract_labels(obj: Any, pt_path: str | Path | None = None) -> torch.Tensor:
    """Extract label tensor from common .pt formats."""
    if isinstance(obj, TensorDataset):
        if len(obj.tensors) < 2:
            raise ValueError(f"{pt_path} does not contain at least (X, y) tensors.")
        return obj.tensors[1]

    if isinstance(obj, (tuple, list)):
        if len(obj) < 2:
            raise ValueError(f"{pt_path} must contain at least (X, y).")
        return obj[1]

    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).strip().lower() in LABEL_KEYS:
                return value
        tensor_values = [value for value in obj.values() if torch.is_tensor(value)]
        if len(tensor_values) >= 2:
            return tensor_values[1]
        raise ValueError(f"Could not infer label tensor from {pt_path}.")

    raise TypeError(f"Unsupported .pt format: {type(obj)}")


def count_labels(labels: torch.Tensor | Sequence[int]) -> dict[int, int]:
    labels = torch.as_tensor(labels).long().view(-1).cpu()
    if labels.numel() == 0:
        return {}
    if int(labels.min().item()) < 0:
        raise ValueError("Negative labels are not supported by torch.bincount.")

    bincount = torch.bincount(labels)
    return {label: int(count) for label, count in enumerate(bincount.tolist()) if count > 0}


def flip_label_value(label: int, mapping: Mapping[int, int] | None = None, num_classes: int | None = None) -> int:
    """Flip one label using an explicit mapping or cyclic replacement."""
    label = int(label)
    if mapping is not None:
        return int(mapping.get(label, label))
    if num_classes is None:
        raise ValueError("num_classes is required when mapping is not provided.")
    return int((label + 1) % int(num_classes))


def flip_labels_tensor(
    labels: torch.Tensor,
    mapping: Mapping[int, int] | None = None,
    num_classes: int | None = None,
) -> torch.Tensor:
    """Flip all labels in a tensor using an explicit mapping or cyclic replacement."""
    flipped = labels.detach().clone().long().view(-1)
    for idx, value in enumerate(flipped.tolist()):
        flipped[idx] = flip_label_value(value, mapping=mapping, num_classes=num_classes)
    return flipped


def flip_majority_class_to_target(
    labels: torch.Tensor,
    target_class: int = 0,
) -> tuple[torch.Tensor, int, int, dict[int, int], dict[int, int]]:
    """Flip the local majority class to a target class.

    This implements the targeted label-flipping variant used by the added
    generation script: for each selected poisoned collaborator, find its local
    majority class and replace that class with ``target_class``.
    """
    y_clean = torch.as_tensor(labels).long().view(-1)
    counts_before = count_labels(y_clean)
    if not counts_before:
        return y_clean.clone(), -1, 0, counts_before, counts_before.copy()

    source_class = max(counts_before.items(), key=lambda item: (item[1], -item[0]))[0]
    y_poisoned = y_clean.clone()
    flipped_samples = 0
    if int(source_class) != int(target_class):
        source_indices = torch.where(y_poisoned == int(source_class))[0]
        y_poisoned[source_indices] = int(target_class)
        flipped_samples = int(source_indices.numel())

    counts_after = count_labels(y_poisoned)
    return y_poisoned, int(source_class), flipped_samples, counts_before, counts_after


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
    input_path: str | Path,
    output_path: str | Path,
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
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    poisoned.to_csv(output_path, index=False)
    return str(output_path)


def poison_pt_file(
    input_path: str | Path,
    output_path: str | Path,
    mapping: Mapping[int, int] | None = None,
    num_classes: int | None = None,
) -> str:
    X, y = unpack_xy(safe_torch_load(input_path))
    poisoned_y = flip_labels_tensor(y, mapping=mapping, num_classes=num_classes)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((X, poisoned_y), output_path)
    return str(output_path)


def poison_selected_clients(
    input_dir: str | Path,
    output_dir: str | Path,
    client_ids: Sequence[int],
    mapping: Mapping[int, int] | None = None,
    num_classes: int | None = None,
) -> list[str]:
    """Copy all clients, cyclic-label-flipping only selected collaborators."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client_set = {int(cid) for cid in client_ids}
    written = []

    for input_path in sorted(input_dir.iterdir()):
        if not input_path.is_file():
            continue
        output_path = output_dir / input_path.name
        client_id = parse_client_id(input_path)
        if client_id in client_set and input_path.suffix.lower() == ".pt":
            written.append(poison_pt_file(input_path, output_path, mapping=mapping, num_classes=num_classes))
        elif client_id in client_set and input_path.suffix.lower() == ".csv":
            written.append(poison_csv_file(input_path, output_path, mapping=mapping, num_classes=num_classes))
        elif input_path.suffix.lower() == ".pt":
            X, y = unpack_xy(safe_torch_load(input_path))
            torch.save((X, y), output_path)
            written.append(str(output_path))
        elif input_path.suffix.lower() == ".csv":
            pd.read_csv(input_path).to_csv(output_path, index=False)
            written.append(str(output_path))
    return written


def save_targeted_summary_csv(rows: list[dict[str, object]], csv_path: str | Path) -> None:
    fieldnames = [
        "client_id",
        "source_file",
        "total_samples",
        "majority_class",
        "majority_class_count",
        "target_class",
        "flipped_samples",
        "label_counts_before",
        "label_counts_after",
    ]
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_targeted_clients_for_case(
    case_name: str,
    source_dir: str | Path,
    poisoned_list_path: str | Path,
    output_dir: str | Path,
    target_class: int = 0,
) -> list[dict[str, object]]:
    """Generate one targeted label-flipping case.

    Clean collaborators are copied unchanged. Selected poisoned collaborators
    have their local majority class flipped to ``target_class``.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client_map = load_client_file_map(source_dir)
    poisoned_ids = read_poisoned_client_ids(poisoned_list_path)
    poisoned_id_set = set(poisoned_ids)
    all_client_ids = sorted(client_map)

    missing_ids = [cid for cid in poisoned_ids if cid not in client_map]
    if missing_ids:
        preview = ", ".join(str(cid) for cid in missing_ids[:10])
        raise FileNotFoundError(f"{case_name}: missing {len(missing_ids)} client ids in {source_dir}: {preview}")

    rows: list[dict[str, object]] = []
    copied_clean_clients = 0

    print(f"\n===== {case_name} =====")
    print(f"Source dir       : {source_dir}")
    print(f"Poisoned list    : {poisoned_list_path}")
    print(f"Output dir       : {output_dir}")
    print(f"Clients total    : {len(all_client_ids)}")
    print(f"Clients to poison: {len(poisoned_ids)}")

    for index, client_id in enumerate(all_client_ids, start=1):
        input_path = client_map[client_id]
        output_path = output_dir / input_path.name

        if client_id not in poisoned_id_set:
            shutil.copy2(input_path, output_path)
            copied_clean_clients += 1
        else:
            X, y = unpack_xy(safe_torch_load(input_path))
            (
                y_poisoned,
                majority_class,
                flipped_samples,
                counts_before,
                counts_after,
            ) = flip_majority_class_to_target(y, target_class=target_class)
            torch.save((X.float() if torch.is_tensor(X) else X, y_poisoned), output_path)
            rows.append(
                {
                    "client_id": int(client_id),
                    "source_file": str(input_path),
                    "total_samples": int(torch.as_tensor(y).numel()),
                    "majority_class": int(majority_class),
                    "majority_class_count": int(counts_before.get(majority_class, 0)),
                    "target_class": int(target_class),
                    "flipped_samples": int(flipped_samples),
                    "label_counts_before": json.dumps(counts_before, sort_keys=True),
                    "label_counts_after": json.dumps(counts_after, sort_keys=True),
                }
            )

        if index % 50 == 0 or index == len(all_client_ids):
            print(f"[{case_name}] processed {index}/{len(all_client_ids)} clients")

    summary_path = output_dir / "targeted_flip_summary.csv"
    save_targeted_summary_csv(rows, summary_path)
    total_flipped = sum(int(row["flipped_samples"]) for row in rows)
    print(f"[OK] Saved {len(rows)} poisoned clients to: {output_dir}")
    print(f"[OK] Copied {copied_clean_clients} clean clients to: {output_dir}")
    print(f"[OK] Saved summary to: {summary_path}")
    print(f"[OK] Total flipped samples: {total_flipped}")
    return rows


def run_label_flipping_cases(
    case_configs: Sequence[Mapping[str, str | Path]],
    output_root: str | Path,
    target_class: int = 0,
) -> dict[str, list[dict[str, object]]]:
    """Generate multiple targeted label-flipping cases."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows: dict[str, list[dict[str, object]]] = {}

    for case in case_configs:
        case_name = str(case["case_name"])
        rows = generate_targeted_clients_for_case(
            case_name=case_name,
            source_dir=Path(case["source_dir"]),
            poisoned_list_path=Path(case["poisoned_list_path"]),
            output_dir=output_root / case_name,
            target_class=target_class,
        )
        all_rows[case_name] = rows
    return all_rows


def build_label_count_row(client_id: int, file_name: str, counts: Mapping[int, int], num_classes: int) -> dict[str, int | str]:
    row: dict[str, int | str] = {
        "client_id": int(client_id),
        "file_name": file_name,
        "total_samples": int(sum(counts.values())),
    }
    for label in range(int(num_classes)):
        row[f"label_{label}"] = int(counts.get(label, 0))
    return row


def analyze_folder(
    folder: str | Path,
    prefix: str,
    output_dir: str | Path,
    num_classes: int,
    client_ids: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Save per-client and aggregate label-count CSV files for one folder."""
    folder = Path(folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    selected_ids = None if client_ids is None else {int(cid) for cid in client_ids}
    rows: list[dict[str, int | str]] = []
    for pt_path in sorted(folder.glob("client*.pt"), key=lambda path: parse_client_id(path) or 10**9):
        client_id = parse_client_id(pt_path)
        if client_id is None or (selected_ids is not None and client_id not in selected_ids):
            continue
        labels = extract_labels(safe_torch_load(pt_path), pt_path)
        counts = count_labels(labels)
        rows.append(build_label_count_row(client_id, pt_path.name, counts, num_classes))

    if not rows:
        raise FileNotFoundError(f"No matching client .pt files found in: {folder}")

    df = pd.DataFrame(rows).sort_values("client_id").reset_index(drop=True)
    per_client_path = output_dir / f"{prefix}_per_client_label_counts.csv"
    df.to_csv(per_client_path, index=False)

    summary: dict[str, int | str] = {
        "dataset": prefix,
        "num_clients": int(len(df)),
        "total_samples": int(df["total_samples"].sum()),
    }
    for label in range(int(num_classes)):
        summary[f"label_{label}"] = int(df[f"label_{label}"].sum())

    summary_path = output_dir / f"{prefix}_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"[OK] {prefix}: {len(df)} clients")
    print(f"     Saved per-client: {per_client_path}")
    print(f"     Saved summary:    {summary_path}")
    return df


def compare_clean_vs_targeted(
    df_clean: pd.DataFrame,
    df_targeted: pd.DataFrame,
    output_dir: str | Path,
    num_classes: int,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged = df_clean.merge(
        df_targeted,
        on="client_id",
        how="outer",
        suffixes=("_clean", "_targeted"),
    ).fillna(0)

    for label in range(int(num_classes)):
        merged[f"delta_label_{label}"] = merged[f"label_{label}_targeted"] - merged[f"label_{label}_clean"]
    merged["delta_total_samples"] = merged["total_samples_targeted"] - merged["total_samples_clean"]

    compare_path = output_dir / "clean_vs_targeted_client_diff.csv"
    merged.sort_values("client_id").to_csv(compare_path, index=False)
    print(f"[OK] Saved compare file: {compare_path}")
    return merged


def analyze_clean_targeted(
    clean_dir: str | Path,
    targeted_dir: str | Path,
    output_dir: str | Path,
    num_classes: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean_df = analyze_folder(clean_dir, "clean", output_dir, num_classes)
    targeted_df = analyze_folder(targeted_dir, "targeted", output_dir, num_classes)
    comparison_df = compare_clean_vs_targeted(clean_df, targeted_df, output_dir, num_classes)
    return clean_df, targeted_df, comparison_df


def analyze_dataset_for_clients(
    dataset_name: str,
    folder: str | Path,
    client_ids: Sequence[int],
    num_classes: int,
) -> pd.DataFrame:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Missing {dataset_name} folder: {folder}")

    rows: list[dict[str, int | str]] = []
    for client_id in sorted(set(int(cid) for cid in client_ids)):
        pt_path = resolve_client_file(folder, client_id)
        labels = extract_labels(safe_torch_load(pt_path), pt_path)
        counts = count_labels(labels)
        row = {
            "dataset": dataset_name,
            "client_id": int(client_id),
            "file_name": pt_path.name,
            "total_samples": int(sum(counts.values())),
        }
        for label in range(int(num_classes)):
            row[f"label_{label}"] = int(counts.get(label, 0))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("client_id").reset_index(drop=True)


def build_poisoned_clients_comparison(df_counts: pd.DataFrame, client_ids: Sequence[int], num_classes: int) -> pd.DataFrame:
    clean = df_counts[df_counts["dataset"] == "clean"].drop(columns=["dataset", "file_name"]).set_index("client_id")
    targeted = df_counts[df_counts["dataset"] == "targeted"].drop(columns=["dataset", "file_name"]).set_index("client_id")
    untargeted = df_counts[df_counts["dataset"] == "untargeted"].drop(columns=["dataset", "file_name"]).set_index("client_id")

    merged = clean.join(targeted, lsuffix="_clean", rsuffix="_targeted", how="inner")
    merged = merged.join(untargeted.add_suffix("_untargeted"), how="inner").reset_index()

    merged["delta_total_targeted_vs_clean"] = merged["total_samples_targeted"] - merged["total_samples_clean"]
    merged["delta_total_untargeted_vs_clean"] = merged["total_samples_untargeted"] - merged["total_samples_clean"]
    for label in range(int(num_classes)):
        merged[f"delta_label_{label}_targeted_vs_clean"] = merged[f"label_{label}_targeted"] - merged[f"label_{label}_clean"]
        merged[f"delta_label_{label}_untargeted_vs_clean"] = merged[f"label_{label}_untargeted"] - merged[f"label_{label}_clean"]

    targeted_cols = [f"delta_label_{label}_targeted_vs_clean" for label in range(int(num_classes))]
    untargeted_cols = [f"delta_label_{label}_untargeted_vs_clean" for label in range(int(num_classes))]
    merged["l1_shift_targeted_vs_clean"] = merged[targeted_cols].abs().sum(axis=1)
    merged["l1_shift_untargeted_vs_clean"] = merged[untargeted_cols].abs().sum(axis=1)

    selected_ids = set(int(cid) for cid in client_ids)
    return merged[merged["client_id"].isin(selected_ids)].sort_values("client_id").reset_index(drop=True)


def build_poisoned_clients_summary(df_comparison: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    metrics = [
        "l1_shift_targeted_vs_clean",
        "l1_shift_untargeted_vs_clean",
        "delta_total_targeted_vs_clean",
        "delta_total_untargeted_vs_clean",
    ]
    for label in range(int(num_classes)):
        metrics.append(f"delta_label_{label}_targeted_vs_clean")
        metrics.append(f"delta_label_{label}_untargeted_vs_clean")

    for metric in metrics:
        rows.append(
            {
                "metric": metric,
                "mean": float(df_comparison[metric].mean()),
                "median": float(df_comparison[metric].median()),
                "min": float(df_comparison[metric].min()),
                "max": float(df_comparison[metric].max()),
            }
        )
    return pd.DataFrame(rows)


def _plot_outputs(df_counts: pd.DataFrame, df_comparison: pd.DataFrame, num_classes: int, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    pivot_df = df_counts.pivot(index="client_id", columns="dataset", values="total_samples")
    pivot_df = pivot_df[["clean", "targeted", "untargeted"]]
    x = np.arange(len(pivot_df))
    width = 0.28

    fig, ax = plt.subplots(figsize=(20, 7))
    ax.bar(x - width, pivot_df["clean"], width=width, label="Clean")
    ax.bar(x, pivot_df["targeted"], width=width, label="Targeted")
    ax.bar(x + width, pivot_df["untargeted"], width=width, label="Untargeted")
    ax.set_title("Total samples per poisoned client across datasets")
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Samples")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index.astype(int), rotation=90)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "plot_total_samples_grouped.png", dpi=180)
    plt.close(fig)

    for suffix, title, filename in [
        ("targeted_vs_clean", "Label distribution shift: Targeted - Clean", "plot_delta_heatmap_targeted_vs_clean.png"),
        ("untargeted_vs_clean", "Label distribution shift: Untargeted - Clean", "plot_delta_heatmap_untargeted_vs_clean.png"),
    ]:
        columns = [f"delta_label_{label}_{suffix}" for label in range(int(num_classes))]
        matrix = df_comparison[columns].to_numpy().T
        fig, ax = plt.subplots(figsize=(20, 6))
        image = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
        ax.set_title(title)
        ax.set_xlabel("Client ID")
        ax.set_ylabel("Label")
        ax.set_xticks(np.arange(len(df_comparison)))
        ax.set_xticklabels(df_comparison["client_id"].astype(int), rotation=90)
        ax.set_yticks(np.arange(int(num_classes)))
        ax.set_yticklabels([f"label_{i}" for i in range(int(num_classes))])
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("Delta count")
        plt.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(20, 7))
    x = np.arange(len(df_comparison))
    width = 0.38
    ax.bar(x - width / 2, df_comparison["l1_shift_targeted_vs_clean"], width=width, label="Targeted vs Clean")
    ax.bar(x + width / 2, df_comparison["l1_shift_untargeted_vs_clean"], width=width, label="Untargeted vs Clean")
    ax.set_title("L1 shift of label distribution per poisoned client")
    ax.set_xlabel("Client ID")
    ax.set_ylabel("L1 shift")
    ax.set_xticks(x)
    ax.set_xticklabels(df_comparison["client_id"].astype(int), rotation=90)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "plot_l1_shift_per_client.png", dpi=180)
    plt.close(fig)


def analyze_poisoned_clients_compare(
    poisoned_list: str | Path,
    clean_dir: str | Path,
    targeted_dir: str | Path,
    untargeted_dir: str | Path,
    output_dir: str | Path,
    num_classes: int = 8,
    make_plots: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    poisoned_ids = read_poisoned_client_ids(poisoned_list)

    clean_df = analyze_dataset_for_clients("clean", clean_dir, poisoned_ids, num_classes)
    targeted_df = analyze_dataset_for_clients("targeted", targeted_dir, poisoned_ids, num_classes)
    untargeted_df = analyze_dataset_for_clients("untargeted", untargeted_dir, poisoned_ids, num_classes)

    all_counts_df = pd.concat([clean_df, targeted_df, untargeted_df], ignore_index=True)
    comparison_df = build_poisoned_clients_comparison(all_counts_df, poisoned_ids, num_classes)
    summary_df = build_poisoned_clients_summary(comparison_df, num_classes)

    all_counts_df.to_csv(output_dir / "poisoned_clients_counts_all_datasets.csv", index=False)
    comparison_df.to_csv(output_dir / "poisoned_clients_diff_vs_clean.csv", index=False)
    summary_df.to_csv(output_dir / "poisoned_clients_diff_summary.csv", index=False)

    if make_plots:
        _plot_outputs(all_counts_df, comparison_df, num_classes, output_dir)

    print(f"[OK] Poisoned clients loaded: {len(poisoned_ids)}")
    print(f"[OK] Saved outputs in: {output_dir}")
    return all_counts_df, comparison_df, summary_df

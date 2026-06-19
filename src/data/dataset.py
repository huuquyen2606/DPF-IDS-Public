"""Dataset loading and CSV/PT conversion utilities."""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


def _safe_torch_load(path, map_location="cpu"):
    """Load trusted local .pt files across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception as exc:
        if "weights_only" in str(exc) or "Unsupported global" in str(exc):
            return torch.load(path, map_location=map_location, weights_only=False)
        raise


def unpack_xy(obj):
    """Support common .pt structures: (X, y), {'X','y'}, {'data','labels'}, etc."""
    if isinstance(obj, (tuple, list)) and len(obj) >= 2:
        return obj[0], obj[1]
    if isinstance(obj, dict):
        key_pairs = [("X", "y"), ("x", "y"), ("data", "labels"), ("features", "labels"), ("X", "labels")]
        for x_key, y_key in key_pairs:
            if x_key in obj and y_key in obj:
                return obj[x_key], obj[y_key]
    raise ValueError(f"Unsupported .pt data format: {type(obj)}")


def _client_file_sort_key(path):
    """Natural sort for client files: client01.pt, client_01.pt, client_1.pt, client-1.pt."""
    base = os.path.basename(str(path))
    match = re.search(r"client[_\-]?(\d+)\.pt$", base)
    if match:
        return (0, int(match.group(1)), base)
    return (1, base)


def natural_client_sort_key(path: str | Path):
    name = Path(path).name
    parts = re.split(r"(\d+)", name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def list_client_files(data_dir: str, num_clients: int | None = None) -> Dict[int, str]:
    patterns = ["client_*.pt", "client*.pt", "client-*.pt"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(data_dir, pattern)))
    files = sorted(set(files), key=_client_file_sort_key)
    if not files:
        raise FileNotFoundError(f"No client .pt files found in: {data_dir}")
    if num_clients is not None:
        files = files[: int(num_clients)]
    return {cid: str(path) for cid, path in enumerate(files, start=1)}

def build_client_loaders(data_dir, num_clients, batch_size, num_workers=2, pin_memory=True):
    """Load client .pt files and build DataLoader for each client.

    Supported file names include:
        client_01.pt, client_1.pt, client01.pt, client1.pt, client-1.pt

    Returns:
        client_loaders: dict {client_id: DataLoader}
        client_info: list of dicts with client_id, num_samples, client_file
    """
    client_loaders = {}
    client_info = []

    patterns = ["client_*.pt", "client*.pt", "client-*.pt"]
    client_files = []
    for pat in patterns:
        client_files.extend(glob.glob(os.path.join(data_dir, pat)))
    client_files = sorted(set(client_files), key=_client_file_sort_key)

    print(f"Found {len(client_files)} client files.")
    if len(client_files) == 0:
        raise FileNotFoundError(f"No client .pt files found in: {data_dir}")

    selected_files = client_files[:num_clients]
    for cid, client_path in enumerate(
        tqdm(selected_files, desc="Loading client files", unit="client"), start=1
    ):
        loaded = _safe_torch_load(client_path, map_location="cpu")
        if isinstance(loaded, dict):
            X_tensor = loaded.get("X", loaded.get("x", loaded.get("features")))
            y_tensor = loaded.get("y", loaded.get("labels", loaded.get("target")))
            if X_tensor is None or y_tensor is None:
                raise ValueError(f"Unsupported dict format in {client_path}. Expected X/y or features/labels.")
        else:
            X_tensor, y_tensor = loaded

        y_tensor = y_tensor.long().view(-1)
        dataset = TensorDataset(X_tensor, y_tensor)
        client_loaders[cid] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        print(f"Loaded data for client {cid}. Samples: {len(dataset)} | File: {os.path.basename(client_path)}")

        client_info.append({
            "client_id": int(cid),
            "num_samples": int(len(dataset)),
            "client_file": os.path.basename(client_path),
            "client_path": client_path,
        })

    return client_loaders, client_info


def load_test_data(test_data_path, batch_size=64, num_workers=2, pin_memory=True):
    """Load test data from a .pt file and return (DataLoader, y_test)."""
    X_test, y_test = unpack_xy(_safe_torch_load(test_data_path, map_location="cpu"))
    y_test = y_test.long().view(-1)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return test_loader, y_test


def dataframe_to_tensors(df: pd.DataFrame, label_col: str = "Label"):
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if label_col in clean_df.columns:
        X = torch.tensor(clean_df.drop(label_col, axis=1).values, dtype=torch.float32)
        y = torch.tensor(clean_df[label_col].astype(int).values, dtype=torch.long)
        return X, y
    return torch.tensor(clean_df.values, dtype=torch.float32)


def csv_to_pt(input_path: str, output_path: str, label_col: str = "Label") -> str:
    df = pd.read_csv(input_path)
    data = dataframe_to_tensors(df, label_col=label_col)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(data, output_path)
    return output_path


def csv_folder_to_pt(input_dir: str, output_dir: str, label_col: str = "Label") -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    written = []
    csv_files = sorted([name for name in os.listdir(input_dir) if name.lower().endswith(".csv")])
    for csv_file in tqdm(csv_files, desc="CSV -> PT", unit="file"):
        input_path = os.path.join(input_dir, csv_file)
        output_path = os.path.join(output_dir, os.path.splitext(csv_file)[0] + ".pt")
        written.append(csv_to_pt(input_path, output_path, label_col=label_col))
    return written

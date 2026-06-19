"""Client partitioning helpers for IID and non-IID experiments."""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


def create_non_iid_distribution(n_samples: int, n_clients: int, concentration: float = 0.5, rng=None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    distribution = rng.dirichlet([concentration] * int(n_clients))
    distribution = (distribution * int(n_samples)).astype(int)
    diff = int(n_samples) - int(distribution.sum())
    for _ in range(abs(diff)):
        idx = int(np.argmin(distribution) if diff > 0 else np.argmax(distribution))
        distribution[idx] += 1 if diff > 0 else -1
    return distribution


def random_split_csv(input_path: str, output_dir: str, num_clients: int, seed: int = 42) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path).sample(frac=1, random_state=seed).reset_index(drop=True)
    chunk_size = len(df) // int(num_clients)
    written = []
    for i in tqdm(range(int(num_clients)), desc="Writing random client CSVs", unit="client"):
        start = i * chunk_size
        chunk = df.iloc[start:] if i == int(num_clients) - 1 else df.iloc[start:start + chunk_size]
        path = os.path.join(output_dir, f"client_{i + 1:02d}.csv")
        chunk.to_csv(path, index=False)
        written.append(path)
    return written


def non_iid_split(
    input_path: str,
    output_dir: str,
    n_clients: int = 50,
    concentration: float = 0.5,
    label_col: str = "Label",
    seed: int | None = 42,
) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    df = pd.read_csv(input_path)
    client_indices = defaultdict(list)

    for label in df[label_col].unique():
        indices = df[df[label_col] == label].index.to_numpy().copy()
        rng.shuffle(indices)
        dist = create_non_iid_distribution(len(indices), n_clients, concentration, rng=rng)
        start = 0
        for client_id in range(int(n_clients)):
            client_indices[client_id].extend(indices[start:start + dist[client_id]].tolist())
            start += int(dist[client_id])

    written = []
    for client_id in tqdm(range(int(n_clients)), desc="Writing non-IID client CSVs", unit="client"):
        client_df = df.loc[client_indices[client_id]]
        path = os.path.join(output_dir, f"client_{client_id + 1:02d}.csv")
        client_df.to_csv(path, index=False)
        written.append(path)
    return written


def chunked_train_test_split(
    input_path: str,
    train_path: str,
    test_path: str,
    test_size: float = 0.3,
    chunksize: int = 100_000,
    label_col: str = "Label",
    seed: int = 42,
) -> tuple[str, str]:
    os.makedirs(os.path.dirname(train_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(test_path) or ".", exist_ok=True)
    first = True
    for chunk in tqdm(pd.read_csv(input_path, chunksize=chunksize), desc="Train/test split", unit="chunk"):
        stratify = chunk[label_col] if label_col in chunk.columns else None
        train, test = train_test_split(chunk, test_size=test_size, random_state=seed, stratify=stratify)
        train.to_csv(train_path, index=False, mode="w" if first else "a", header=first)
        test.to_csv(test_path, index=False, mode="w" if first else "a", header=first)
        first = False
    return train_path, test_path

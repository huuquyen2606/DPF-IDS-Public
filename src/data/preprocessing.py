"""Preprocessing utilities for CIC-style CSV data."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm


def clean_data(input_path: str, output_path: str, chunksize: int = 1_000_000, label_col: str = "Label") -> str:
    """Remove rows with inf/NaN values and write a cleaned CSV in chunks."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    first = True
    for chunk in tqdm(pd.read_csv(input_path, chunksize=chunksize), desc="Cleaning CSV", unit="chunk"):
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
        if label_col in chunk.columns:
            chunk = chunk.dropna(subset=[label_col])
        chunk.to_csv(output_path, mode="w" if first else "a", index=False, header=first)
        first = False
    return output_path


def fit_minmax_scaler(train_csv_path: str, chunksize: int = 1_000_000, label_col: str = "Label") -> MinMaxScaler:
    scaler = MinMaxScaler()
    for chunk in tqdm(pd.read_csv(train_csv_path, chunksize=chunksize), desc="Fitting scaler", unit="chunk"):
        numeric = chunk.select_dtypes(include=[np.number])
        if label_col in numeric.columns:
            numeric = numeric.drop(columns=[label_col])
        scaler.partial_fit(numeric)
    return scaler


def transform_csv_with_scaler(
    input_path: str,
    output_path: str,
    scaler: MinMaxScaler,
    chunksize: int = 1_000_000,
    label_col: str = "Label",
) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    first = True
    for chunk in tqdm(pd.read_csv(input_path, chunksize=chunksize), desc="Scaling CSV", unit="chunk"):
        numeric = chunk.select_dtypes(include=[np.number])
        labels = chunk[label_col].values if label_col in chunk.columns else None
        if label_col in numeric.columns:
            numeric = numeric.drop(columns=[label_col])
        scaled = pd.DataFrame(scaler.transform(numeric), columns=numeric.columns)
        if labels is not None:
            scaled[label_col] = labels
        scaled.to_csv(output_path, mode="w" if first else "a", index=False, header=first)
        first = False
    return output_path


def check_nan_inf(csv_path: str, label_col: str = "Label") -> dict[str, int]:
    df = pd.read_csv(csv_path)
    features = df.drop(columns=[label_col]) if label_col in df.columns else df
    labels = df[label_col] if label_col in df.columns else None
    return {
        "feature_nan": int(features.isna().sum().sum()),
        "feature_inf": int(np.isinf(features.values).sum()),
        "label_nan": int(labels.isna().sum()) if labels is not None else 0,
        "label_inf": int(np.isinf(labels.values).sum()) if labels is not None else 0,
    }

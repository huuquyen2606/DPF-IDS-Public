"""Evaluation utilities for DPF-IDS client checkpoints."""

from __future__ import annotations

import gc
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .data.dataset import _safe_torch_load, list_client_files, load_test_data, unpack_xy
from .models.ffnn import FFNN
from .utils.logging import load_poisoned_clients_for_round
from .utils.metrics import (
    append_or_write_metrics,
    compute_average_metrics,
    load_existing_client_metrics,
    metrics_from_confusion_matrix,
    save_confusion_matrix_csv,
)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

G_X_TEST = None
G_Y_TEST = None
G_FIRST_BATCHES = None
G_CONFIG_DICT = None


@dataclass
class EvaluationConfig:
    input_size: int = 39
    num_classes: int = 8
    num_clients: int = 500
    learning_rate: float = 0.01
    preupdate_warmup_lr_scale: float = 0.5
    checkpoint_dir: str = "."
    checkpoint_algorithm: str = "FFNN"
    data_train: str = ""
    data_test: str = ""
    evaluation_rounds: List[int] = field(default_factory=lambda: list(range(1, 10)))
    parallel_device: str = "cpu"
    max_workers: int = field(default_factory=lambda: min(24, os.cpu_count() or 1))
    torch_threads_per_worker: int = 1
    test_batch_size: int = 65536
    preupdate_batch_size: int = 1024
    cache_client_first_batches: bool = True
    preupdate_shuffle: bool = False
    cache_test_in_ram: bool = True
    mp_start_method: str = field(default_factory=lambda: "fork" if os.name != "nt" else "spawn")
    eval_log_subdir: str = "eval_logs_parallel"
    resume_eval: bool = True
    flush_each_client: bool = True
    save_confusion_matrices: bool = True
    verbose_client_logs: bool = False
    seed: int = 42


def evaluate_model(model, test_loader, device):
    """Evaluate one model on test data and return metrics + confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", unit="batch", leave=False):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            _, outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
    }
    class_labels = np.unique(np.concatenate([all_labels, all_preds]))
    conf_mat = confusion_matrix(all_labels, all_preds, labels=class_labels)
    return metrics, all_preds, all_labels, conf_mat, class_labels


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_eval_progress(csv_path, tag):
    """Load existing per-client rows for a specific evaluation tag."""
    if not os.path.exists(csv_path):
        return {}, []
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}, []
    if df.empty:
        return {}, []
    if "tag" in df.columns:
        df = df[df["tag"].astype(str) == str(tag)]
    if "scope" in df.columns:
        df = df[df["scope"].astype(str) == "client"]

    loaded_metrics = {}
    loaded_rows = []
    for _, row in df.iterrows():
        try:
            client_id = int(row.get("client_id"))
        except (TypeError, ValueError):
            continue
        metrics = {
            "accuracy": _to_float(row.get("accuracy"), 0.0),
            "f1": _to_float(row.get("f1"), 0.0),
            "precision": _to_float(row.get("precision"), 0.0),
            "recall": _to_float(row.get("recall"), 0.0),
        }
        loaded_metrics[client_id] = metrics
        loaded_rows.append({"tag": str(tag), "scope": "client", "client_id": client_id, **metrics})
    return loaded_metrics, loaded_rows


def _confusion_matrix_file_name(tag, client_id):
    return f"confusion_matrix_{tag}_client_{int(client_id):04d}.csv"


def save_client_confusion_matrix(conf_mat, class_labels, client_id, tag, cm_dir):
    """Save one client's confusion matrix to CSV."""
    if cm_dir is None:
        return None
    os.makedirs(cm_dir, exist_ok=True)
    cm_path = os.path.join(cm_dir, _confusion_matrix_file_name(tag, client_id))
    cm_df = pd.DataFrame(conf_mat, index=class_labels, columns=class_labels)
    cm_df.index.name = "true_label"
    cm_df.columns.name = "pred_label"
    cm_df.to_csv(cm_path)
    return cm_path


def evaluate_all_clients_average(
    client_models,
    test_loader,
    device,
    csv_dir=None,
    tag="eval",
    resume=True,
    flush_each_client=True,
    cm_dir=None,
):
    """Evaluate all client models and return per-client + averaged metrics."""
    if not client_models:
        raise ValueError("client_models is empty")

    csv_path = None
    loaded_metrics = {}
    csv_rows = []

    if cm_dir is None and csv_dir is not None:
        cm_dir = os.path.join(csv_dir, "confusion_matrices")
    if cm_dir is not None:
        os.makedirs(cm_dir, exist_ok=True)
    if csv_dir is not None:
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f"metrics_{tag}.csv")
        if resume:
            loaded_metrics, csv_rows = _load_eval_progress(csv_path, tag)

    per_client_metrics = dict(loaded_metrics)
    for cid, model in sorted(client_models.items()):
        if resume and cid in loaded_metrics:
            continue
        metrics, _, _, conf_mat, class_labels = evaluate_model(model, test_loader, device)
        per_client_metrics[cid] = metrics
        cm_path = save_client_confusion_matrix(conf_mat, class_labels, cid, tag, cm_dir)
        if cm_path is not None:
            print(f"Saved confusion matrix for client {cid}: {cm_path}")
        csv_rows.append({"tag": tag, "scope": "client", "client_id": int(cid), **{k: float(v) for k, v in metrics.items()}})
        if csv_path is not None and flush_each_client:
            pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    if not per_client_metrics:
        raise RuntimeError("No client metrics available to compute average.")

    avg_metrics = {
        "accuracy": float(np.mean([m["accuracy"] for m in per_client_metrics.values()])),
        "f1": float(np.mean([m["f1"] for m in per_client_metrics.values()])),
        "precision": float(np.mean([m["precision"] for m in per_client_metrics.values()])),
        "recall": float(np.mean([m["recall"] for m in per_client_metrics.values()])),
    }
    final_rows = list(csv_rows)
    final_rows.append({"tag": tag, "scope": "average", "client_id": "all", **avg_metrics})
    if csv_path is not None:
        pd.DataFrame(final_rows).to_csv(csv_path, index=False)
    return avg_metrics, per_client_metrics


def load_test_tensors(test_data_path: str, cache_in_ram: bool = True):
    obj = _safe_torch_load(test_data_path, map_location="cpu")
    X_test, y_test = unpack_xy(obj)
    X_test = X_test.detach().contiguous().float().cpu()
    y_test = y_test.detach().contiguous().long().cpu()
    if cache_in_ram:
        try:
            X_test.share_memory_()
            y_test.share_memory_()
        except Exception as exc:
            print(f"Shared memory skipped: {exc}")
    print(f"Loaded test data: X={tuple(X_test.shape)}, y={tuple(y_test.shape)}")
    return X_test, y_test


def load_client_first_batches(client_paths: Dict[int, str], batch_size: int, shuffle: bool, seed: int):
    """Cache only the first pre-update batch for each client in RAM."""
    first_batches = {}
    for cid, path in tqdm(sorted(client_paths.items()), desc="Caching first pre-update batches", unit="client"):
        obj = _safe_torch_load(path, map_location="cpu")
        X, y = unpack_xy(obj)
        X = X.detach().contiguous().float().cpu()
        y = y.detach().contiguous().long().cpu()
        take = min(int(batch_size), int(X.shape[0]))
        if shuffle:
            g = torch.Generator()
            g.manual_seed(int(seed) + int(cid))
            idx = torch.randperm(X.shape[0], generator=g)[:take]
        else:
            idx = torch.arange(take)
        xb = X[idx].contiguous()
        yb = y[idx].contiguous()
        try:
            xb.share_memory_()
            yb.share_memory_()
        except Exception:
            pass
        first_batches[int(cid)] = (xb, yb)
    return first_batches


def checkpoint_path_by_round(checkpoint_dir: str, algorithm: str, target_round: int) -> str:
    return os.path.join(checkpoint_dir, f"{algorithm}_checkpoint_round{int(target_round)}.pt")


def load_checkpoint_cpu(checkpoint_dir: str, algorithm: str, target_round: int):
    path = checkpoint_path_by_round(checkpoint_dir, algorithm, target_round)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = _safe_torch_load(path, map_location="cpu")
    model_state = checkpoint.get("model_state_dict")
    if model_state is None:
        raise KeyError("checkpoint['model_state_dict'] is missing")
    extra_state = checkpoint.get("extra_state", {}) or {}
    loaded_round = checkpoint.get("round", target_round)
    return checkpoint, model_state, extra_state, loaded_round, path


def get_client_state_dict(model_state: Any, client_id: int):
    """Handle checkpoints where client keys are int or string."""
    if not isinstance(model_state, dict):
        raise ValueError("Expected checkpoint model_state_dict to be a dict of client states.")
    possible_keys = [client_id, str(client_id), f"client_{client_id}", f"client_{client_id}.pt"]
    for key in possible_keys:
        if key in model_state:
            return model_state[key]
    raise KeyError(f"Client {client_id} not found in model_state_dict. Example keys: {list(model_state.keys())[:5]}")


def move_state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().contiguous() if torch.is_tensor(value) else value for key, value in state_dict.items()}


def _init_parallel_worker(config_dict, x_test=None, y_test=None, first_batches=None):
    global G_CONFIG_DICT, G_X_TEST, G_Y_TEST, G_FIRST_BATCHES
    G_CONFIG_DICT = config_dict
    if x_test is not None:
        G_X_TEST = x_test
    if y_test is not None:
        G_Y_TEST = y_test
    if first_batches is not None:
        G_FIRST_BATCHES = first_batches
    torch.set_num_threads(int(config_dict.get("torch_threads_per_worker", 1)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def pre_update_with_global_prototype_one_batch(
    model: nn.Module,
    x_batch: torch.Tensor,
    lr: float,
    global_prototype: torch.Tensor,
    warmup_lr_scale: float,
    device: str = "cpu",
) -> Tuple[nn.Module, Optional[float]]:
    if global_prototype is None:
        return model, None
    model = model.to(device)
    model.train()
    x_batch = x_batch.to(device, non_blocking=False)
    global_proto_vec = global_prototype.detach().to(device).view(-1).float()
    optimizer = optim.SGD(model.parameters(), lr=float(lr) * float(warmup_lr_scale), momentum=0.9, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    hidden_output, _ = model(x_batch)
    local_batch_proto = hidden_output.mean(dim=0).view(-1).float()
    proto_align_loss = torch.norm(local_batch_proto - global_proto_vec, p=2)
    proto_align_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return model, float(proto_align_loss.item())


def get_first_batch_for_client(client_id: int, client_path: str, batch_size: int, shuffle: bool, seed: int):
    global G_FIRST_BATCHES
    if G_FIRST_BATCHES is not None and int(client_id) in G_FIRST_BATCHES:
        return G_FIRST_BATCHES[int(client_id)]
    obj = _safe_torch_load(client_path, map_location="cpu")
    X, y = unpack_xy(obj)
    X = X.detach().contiguous().float().cpu()
    y = y.detach().contiguous().long().cpu()
    take = min(int(batch_size), int(X.shape[0]))
    if shuffle:
        g = torch.Generator()
        g.manual_seed(int(seed) + int(client_id))
        idx = torch.randperm(X.shape[0], generator=g)[:take]
    else:
        idx = torch.arange(take)
    return X[idx].contiguous(), y[idx].contiguous()


def evaluate_model_fast_cpu(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor, batch_size: int, num_classes: int):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    n = int(y_test.shape[0])
    with torch.inference_mode():
        for start in range(0, n, int(batch_size)):
            end = min(start + int(batch_size), n)
            xb = X_test[start:end]
            yb = y_test[start:end]
            _, logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)
            labels = yb.cpu().numpy().astype(np.int64)
            valid = (labels >= 0) & (labels < num_classes) & (preds >= 0) & (preds < num_classes)
            if valid.any():
                idx = labels[valid] * num_classes + preds[valid]
                cm += np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    metrics = metrics_from_confusion_matrix(cm)
    return metrics, cm


def evaluate_one_client_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    global G_X_TEST, G_Y_TEST
    cfg = task["cfg"]
    cid = int(task["client_id"])
    model = FFNN(in_features=int(cfg["input_size"]), num_classes=int(cfg["num_classes"]))
    model.load_state_dict(task["state_dict"])
    model.to("cpu")
    x_first, _ = get_first_batch_for_client(
        client_id=cid,
        client_path=task["client_path"],
        batch_size=int(cfg["preupdate_batch_size"]),
        shuffle=bool(cfg["preupdate_shuffle"]),
        seed=int(cfg["seed"]),
    )
    model, align_loss = pre_update_with_global_prototype_one_batch(
        model=model,
        x_batch=x_first,
        lr=float(cfg["learning_rate"]),
        global_prototype=task["global_proto"],
        warmup_lr_scale=float(cfg["preupdate_warmup_lr_scale"]),
        device="cpu",
    )
    metrics, cm = evaluate_model_fast_cpu(
        model=model,
        X_test=G_X_TEST,
        y_test=G_Y_TEST,
        batch_size=int(cfg["test_batch_size"]),
        num_classes=int(cfg["num_classes"]),
    )
    cm_path = None
    if bool(cfg["save_confusion_matrices"]):
        cm_path = save_confusion_matrix_csv(cm, cid, task["tag"], task["cm_dir"], int(cfg["num_classes"]))
    return {"client_id": cid, "metrics": metrics, "align_loss": align_loss, "cm_path": cm_path}


def config_to_worker_dict(cfg: EvaluationConfig) -> Dict[str, Any]:
    keys = [
        "input_size",
        "num_classes",
        "learning_rate",
        "preupdate_warmup_lr_scale",
        "preupdate_batch_size",
        "preupdate_shuffle",
        "test_batch_size",
        "torch_threads_per_worker",
        "save_confusion_matrices",
        "seed",
    ]
    return {key: getattr(cfg, key) for key in keys}


def evaluate_clients_parallel_for_round(
    target_previous_round: int,
    client_paths: Dict[int, str],
    cfg: EvaluationConfig,
    eval_log_dir: str,
    cm_root_dir: str,
) -> Dict[str, Any]:
    current_round_for_eval = int(target_previous_round) + 1
    tag = f"round_{current_round_for_eval}_preupdate_detected_benign_prev_round_eval"
    csv_path = os.path.join(eval_log_dir, f"metrics_{tag}.csv")
    round_cm_dir = os.path.join(cm_root_dir, f"round_{int(target_previous_round):02d}")
    os.makedirs(round_cm_dir, exist_ok=True)

    checkpoint, model_state, extra_state, loaded_round, checkpoint_path = load_checkpoint_cpu(
        checkpoint_dir=cfg.checkpoint_dir,
        algorithm=cfg.checkpoint_algorithm,
        target_round=target_previous_round,
    )
    global_proto = extra_state.get("aggregated_server_proto", None)
    if global_proto is None:
        raise RuntimeError("aggregated_server_proto is missing in checkpoint extra_state.")
    global_proto = global_proto.detach().cpu().float() if torch.is_tensor(global_proto) else torch.tensor(global_proto, dtype=torch.float32)

    poisoned_ids = set(load_poisoned_clients_for_round(cfg.checkpoint_dir, target_previous_round))
    benign_ids = sorted([cid for cid in client_paths.keys() if cid not in poisoned_ids])
    loaded_metrics, csv_rows = load_existing_client_metrics(csv_path, tag) if cfg.resume_eval else ({}, [])
    pending_ids = [cid for cid in benign_ids if cid not in loaded_metrics]
    per_client_metrics = dict(loaded_metrics)

    if pending_ids:
        worker_cfg = config_to_worker_dict(cfg)
        tasks = []
        for cid in pending_ids:
            tasks.append(
                {
                    "client_id": int(cid),
                    "client_path": client_paths[int(cid)],
                    "state_dict": move_state_dict_to_cpu(get_client_state_dict(model_state, cid)),
                    "global_proto": global_proto,
                    "cfg": worker_cfg,
                    "tag": tag,
                    "cm_dir": round_cm_dir,
                }
            )
        del checkpoint
        gc.collect()

        max_workers = min(int(cfg.max_workers), len(tasks))
        mp_context = mp.get_context(cfg.mp_start_method)
        start_time = time.time()
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp_context,
            initializer=_init_parallel_worker,
            initargs=(worker_cfg, G_X_TEST, G_Y_TEST, G_FIRST_BATCHES),
        ) as executor:
            futures = [executor.submit(evaluate_one_client_worker, task) for task in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Round {target_previous_round} clients", unit="client"):
                result = fut.result()
                cid = int(result["client_id"])
                metrics = result["metrics"]
                per_client_metrics[cid] = metrics
                csv_rows.append(
                    {
                        "tag": tag,
                        "scope": "client",
                        "client_id": cid,
                        **{key: float(value) for key, value in metrics.items()},
                        "align_loss": result.get("align_loss"),
                        "cm_path": result.get("cm_path"),
                    }
                )
                if cfg.flush_each_client:
                    append_or_write_metrics(csv_path, csv_rows)
        print(f"Round {target_previous_round} pending clients completed in {time.time() - start_time:.2f} seconds.")

    avg_metrics = compute_average_metrics(per_client_metrics)
    final_rows = [row for row in csv_rows if str(row.get("scope", "client")) == "client"]
    final_rows.append({"tag": tag, "scope": "average", "client_id": "all", **avg_metrics, "align_loss": "", "cm_path": ""})
    append_or_write_metrics(csv_path, final_rows)

    return {
        "checkpoint_round": int(target_previous_round),
        "eval_round": int(current_round_for_eval),
        "eval_tag": tag,
        "checkpoint_path": checkpoint_path,
        "num_excluded_poisoned_clients": int(len(poisoned_ids)),
        "num_benign_clients": int(len(benign_ids)),
        **{key: float(value) for key, value in avg_metrics.items()},
    }


def run_parallel_evaluation(config: EvaluationConfig) -> list[dict[str, Any]]:
    global G_X_TEST, G_Y_TEST, G_FIRST_BATCHES
    if config.parallel_device.lower() != "cpu":
        raise ValueError("Parallel evaluator is CPU-oriented; set parallel_device='cpu'.")
    if os.name == "nt" and config.mp_start_method != "spawn":
        raise RuntimeError("On Windows, set mp_start_method='spawn'.")
    if not config.data_train:
        raise ValueError("config.data_train is required")
    if not config.data_test:
        raise ValueError("config.data_test is required")

    eval_log_dir = os.path.join(config.checkpoint_dir, config.eval_log_subdir)
    cm_root_dir = os.path.join(eval_log_dir, "confusion_matrices")
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(cm_root_dir, exist_ok=True)

    client_paths = list_client_files(config.data_train, config.num_clients)
    G_X_TEST, G_Y_TEST = load_test_tensors(config.data_test, cache_in_ram=config.cache_test_in_ram)
    G_FIRST_BATCHES = (
        load_client_first_batches(client_paths, config.preupdate_batch_size, config.preupdate_shuffle, config.seed)
        if config.cache_client_first_batches
        else None
    )

    results = []
    for target_previous_round in list(config.evaluation_rounds):
        results.append(
            evaluate_clients_parallel_for_round(
                target_previous_round=int(target_previous_round),
                client_paths=client_paths,
                cfg=config,
                eval_log_dir=eval_log_dir,
                cm_root_dir=cm_root_dir,
            )
        )

    if results:
        summary_path = os.path.join(
            eval_log_dir,
            f"metrics_round_{min(config.evaluation_rounds)}_{max(config.evaluation_rounds)}_summary.csv",
        )
        pd.DataFrame(results).to_csv(summary_path, index=False)
        print(f"Saved round summary CSV: {summary_path}")
    return results


def main(config: EvaluationConfig | None = None) -> list[dict[str, Any]]:
    return run_parallel_evaluation(config or EvaluationConfig())


if __name__ == "__main__":
    mp.freeze_support()
    main()

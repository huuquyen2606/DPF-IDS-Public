"""BoTPA targeted poisoning attack utilities."""

from __future__ import annotations

import copy
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm

from ..data.dataset import _safe_torch_load
from ..dpf_ids.alignment import pre_update_with_global_prototype
from .pga import clone_state_dict_cpu

def _forward_features_logits(model, inputs):
    """Return (features, logits) for models that output either logits or (features, logits)."""
    out = model(inputs)
    if isinstance(out, (tuple, list)):
        if len(out) >= 2:
            return out[0], out[-1]
        return None, out[0]
    return None, out


def soft_cross_entropy(logits, soft_targets, reduction="mean"):
    """Cross entropy that supports soft target vectors."""
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(soft_targets.to(logits.device).float() * log_probs).sum(dim=1)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def one_hot_label(label, num_classes, dtype=torch.float32):
    y = torch.zeros(int(num_classes), dtype=dtype)
    y[int(label)] = 1.0
    return y


def _extract_xy_from_dataset(dataset):
    """Extract X/y tensors from common dataset formats."""
    if isinstance(dataset, TensorDataset) and len(dataset.tensors) >= 2:
        X, y = dataset.tensors[:2]
        return X.detach().cpu(), y.detach().cpu().long().view(-1)

    X_list, y_list = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        X_list.append(x.detach().cpu() if torch.is_tensor(x) else torch.tensor(x))
        y_list.append(int(y.item()) if torch.is_tensor(y) else int(y))
    return torch.stack(X_list, dim=0), torch.tensor(y_list, dtype=torch.long)


def _sample_rows(X, max_samples=None, seed=42):
    """Return at most max_samples rows from X. If max_samples=None, return all rows."""
    if X is None or len(X) == 0:
        return X
    if max_samples is None or int(max_samples) >= len(X):
        return X
    g = torch.Generator()
    g.manual_seed(int(seed))
    idx = torch.randperm(len(X), generator=g)[:int(max_samples)]
    return X[idx]


def _append_with_cap(existing, new_x, cap=None, seed=42):
    """Append new_x to existing tensor list while respecting a per-class cap."""
    if new_x is None or len(new_x) == 0:
        return existing
    current_n = sum(len(t) for t in existing)
    if cap is not None:
        remaining = int(cap) - int(current_n)
        if remaining <= 0:
            return existing
        if len(new_x) > remaining:
            new_x = _sample_rows(new_x, remaining, seed=seed)
    existing.append(new_x.detach().cpu())
    return existing


def sample_malicious_data_by_class(
    client_loaders,
    malicious_client_ids,
    num_classes,
    max_samples_per_class=4000,
    seed=42,
    verbose=True,
):
    """Collect malicious-client data by original class label.

    max_samples_per_class controls only BoTPA preparation cost. The actual
    malicious local training still applies BoTPA to the full malicious client dataset.
    Set max_samples_per_class=None to use all available malicious data for preparation.
    """
    malicious_client_ids = set(int(x) for x in (malicious_client_ids or []))
    by_class_lists = {int(c): [] for c in range(int(num_classes))}

    for cid in sorted(malicious_client_ids):
        if cid not in client_loaders:
            continue
        X, y = _extract_xy_from_dataset(client_loaders[cid].dataset)
        for c in range(int(num_classes)):
            idx = (y == int(c)).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            X_c = X[idx]
            by_class_lists[c] = _append_with_cap(
                by_class_lists[c],
                X_c,
                cap=max_samples_per_class,
                seed=int(seed) + int(cid) * 1009 + int(c),
            )

    by_class = {}
    for c, chunks in by_class_lists.items():
        if chunks:
            by_class[c] = torch.cat(chunks, dim=0)
        else:
            by_class[c] = torch.empty((0,), dtype=torch.float32)

    if verbose:
        counts = {c: int(len(Xc)) for c, Xc in by_class.items()}
        print("\n=== BoTPA malicious preparation data by class ===")
        print(counts)

    return by_class


def make_botpa_surrogate_dataset(class_data_by_original_label, source_class, target_class):
    """Build surrogate dataset with source labels hard-flipped to target labels."""
    X_parts, y_parts = [], []
    for c, X_c in sorted(class_data_by_original_label.items()):
        if X_c is None or len(X_c) == 0:
            continue
        y_value = int(target_class) if int(c) == int(source_class) else int(c)
        X_parts.append(X_c.detach().cpu())
        y_parts.append(torch.full((len(X_c),), y_value, dtype=torch.long))

    if not X_parts:
        raise RuntimeError("BoTPA cannot train surrogate: no malicious preparation data was collected.")

    X = torch.cat(X_parts, dim=0)
    y = torch.cat(y_parts, dim=0)
    return TensorDataset(X, y)


def _load_state_dict_to_device(model, state_dict, device):
    model.load_state_dict({k: v.to(device) if torch.is_tensor(v) else v for k, v in state_dict.items()})
    return model


def train_botpa_surrogate(
    base_model,
    class_data_by_original_label,
    source_class,
    target_class,
    config_model,
    device,
):
    """Train the BoTPA surrogate and return model, w_mid, w_conv, and training info."""
    surrogate = copy.deepcopy(base_model).to(device)
    surrogate_dataset = make_botpa_surrogate_dataset(
        class_data_by_original_label=class_data_by_original_label,
        source_class=source_class,
        target_class=target_class,
    )

    batch_size = int(getattr(config_model, "botpa_surrogate_batch_size", getattr(config_model, "batch_size", 1024)))
    loader = DataLoader(
        surrogate_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(getattr(config_model, "botpa_num_workers", 0)),
        pin_memory=bool(getattr(config_model, "botpa_pin_memory", True)),
    )

    epochs = int(getattr(config_model, "botpa_surrogate_epochs", 5))
    mid_epoch = getattr(config_model, "botpa_surrogate_middle_epoch", None)
    if mid_epoch is None:
        mid_epoch = max(1, int(math.ceil(epochs / 2)))
    mid_epoch = int(max(1, min(int(mid_epoch), epochs)))

    opt_name = str(getattr(config_model, "botpa_surrogate_optimizer", "adam")).lower()
    lr = float(getattr(config_model, "botpa_surrogate_lr", getattr(config_model, "learning_rate", 0.001)))
    wd = float(getattr(config_model, "botpa_surrogate_weight_decay", 1e-4))
    if opt_name == "sgd":
        optimizer = optim.SGD(surrogate.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        optimizer = optim.Adam(surrogate.parameters(), lr=lr, weight_decay=wd)

    criterion = nn.CrossEntropyLoss()
    w_mid = None
    losses = []

    print("\n=== Training BoTPA surrogate model ===")
    print(f"source -> target: {source_class} -> {target_class} | epochs={epochs} | mid_epoch={mid_epoch}")

    for epoch in tqdm(range(1, epochs + 1), desc="BoTPA surrogate epochs", unit="epoch"):
        surrogate.train()
        total_loss, n_batches = 0.0, 0
        for X, y in tqdm(loader, desc=f"Surrogate epoch {epoch}/{epochs}", unit="batch", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            _, logits = _forward_features_logits(surrogate, X)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(surrogate.parameters(), max_norm=float(getattr(config_model, "botpa_grad_clip", 5.0)))
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        epoch_loss = total_loss / max(n_batches, 1)
        losses.append(epoch_loss)
        print(f"BoTPA surrogate epoch {epoch}/{epochs}, loss={epoch_loss:.6f}")
        if epoch == mid_epoch:
            w_mid = clone_state_dict_cpu(surrogate)

    if w_mid is None:
        w_mid = clone_state_dict_cpu(surrogate)
    w_conv = clone_state_dict_cpu(surrogate)

    info = {
        "surrogate_epochs": epochs,
        "surrogate_middle_epoch": mid_epoch,
        "surrogate_losses": losses,
        "surrogate_num_samples": int(len(surrogate_dataset)),
    }
    return surrogate, w_mid, w_conv, info


def _grad_vectors_for_class(
    model,
    X,
    label_for_grad,
    device,
    max_samples=24,
    seed=42,
):
    """Compute normalized per-sample gradient vectors of -log p(label_for_grad | x)."""
    if X is None or len(X) == 0:
        return None
    X = _sample_rows(X, max_samples=max_samples, seed=seed)
    params = [p for p in model.parameters() if p.requires_grad]
    vectors = []
    model.eval()

    for i in range(len(X)):
        xi = X[i:i + 1].to(device)
        yi = torch.tensor([int(label_for_grad)], dtype=torch.long, device=device)
        model.zero_grad(set_to_none=True)
        _, logits = _forward_features_logits(model, xi)
        loss = F.cross_entropy(logits, yi, reduction="sum")
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=True)
        flat_parts = []
        for g in grads:
            if g is not None:
                flat_parts.append(g.detach().flatten().float().cpu())
        if not flat_parts:
            continue
        flat = torch.cat(flat_parts, dim=0)
        flat = flat / (flat.norm(p=2) + 1e-12)
        vectors.append(flat)

    if not vectors:
        return None
    return torch.stack(vectors, dim=0)


def compute_contribution_similarity(
    base_model,
    w_mid,
    class_data_by_original_label,
    source_class,
    target_class,
    num_classes,
    device,
    grad_samples_per_class=24,
    seed=42,
):
    """Compute class-level contribution similarity between source and candidate classes."""
    model = copy.deepcopy(base_model).to(device)
    _load_state_dict_to_device(model, w_mid, device)

    source_X = class_data_by_original_label.get(int(source_class))
    source_grad_label = int(target_class)  # source samples are already hard-flipped in poisoned training
    G_src = _grad_vectors_for_class(
        model=model,
        X=source_X,
        label_for_grad=source_grad_label,
        device=device,
        max_samples=grad_samples_per_class,
        seed=seed + 17,
    )
    if G_src is None:
        raise RuntimeError(f"No source-class samples available for BoTPA source_class={source_class}.")

    contribution_scores = {}
    for c in range(int(num_classes)):
        if int(c) in {int(source_class), int(target_class)}:
            continue
        X_c = class_data_by_original_label.get(int(c))
        G_c = _grad_vectors_for_class(
            model=model,
            X=X_c,
            label_for_grad=int(c),
            device=device,
            max_samples=grad_samples_per_class,
            seed=seed + 1000 + int(c),
        )
        if G_c is None:
            contribution_scores[int(c)] = float("-inf")
            continue
        score = float((G_src @ G_c.T).mean().item())
        contribution_scores[int(c)] = score

    return contribution_scores


def select_intermediate_classes(contribution_scores, num_intermediate_classes):
    """Select top-N classes with the highest contribution similarity."""
    valid_items = [(int(c), float(s)) for c, s in contribution_scores.items() if np.isfinite(float(s))]
    valid_items = sorted(valid_items, key=lambda kv: kv[1], reverse=True)
    return [c for c, _ in valid_items[:int(num_intermediate_classes)]]


def _representations_for_class(
    model,
    X,
    device,
    max_samples=512,
    seed=42,
    feature_layer="logits",
):
    if X is None or len(X) == 0:
        return None
    X = _sample_rows(X, max_samples=max_samples, seed=seed)
    batch_size = 4096
    reps = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size].to(device)
            features, logits = _forward_features_logits(model, xb)
            if str(feature_layer).lower() == "features" and features is not None:
                rep = features
            else:
                rep = logits
            rep = rep.detach().float().cpu()
            rep = F.normalize(rep, p=2, dim=1, eps=1e-12)
            reps.append(rep)
    if not reps:
        return None
    return torch.cat(reps, dim=0)


def compute_feature_similarity(
    base_model,
    w_conv,
    class_data_by_original_label,
    source_class,
    intermediate_classes,
    device,
    feature_samples_per_class=512,
    seed=42,
    feature_layer="logits",
):
    """Compute class-level latent/logits similarity between source and intermediate classes."""
    model = copy.deepcopy(base_model).to(device)
    _load_state_dict_to_device(model, w_conv, device)

    R_src = _representations_for_class(
        model=model,
        X=class_data_by_original_label.get(int(source_class)),
        device=device,
        max_samples=feature_samples_per_class,
        seed=seed + 31,
        feature_layer=feature_layer,
    )
    if R_src is None:
        raise RuntimeError(f"No source-class samples available for feature similarity, source_class={source_class}.")

    feature_scores = {}
    for c in intermediate_classes:
        R_c = _representations_for_class(
            model=model,
            X=class_data_by_original_label.get(int(c)),
            device=device,
            max_samples=feature_samples_per_class,
            seed=seed + 2000 + int(c),
            feature_layer=feature_layer,
        )
        if R_c is None:
            feature_scores[int(c)] = float("-inf")
            continue
        feature_scores[int(c)] = float((R_src @ R_c.T).mean().item())

    return feature_scores


def build_botpa_soft_labels(feature_scores, intermediate_classes, num_classes, target_class):
    """Build soft labels for each selected intermediate class."""
    soft_labels = {}
    alpha_by_class = {}
    for c in intermediate_classes:
        raw_alpha = float(feature_scores.get(int(c), 0.0))
        alpha = float(np.clip(raw_alpha, 0.0, 1.0)) if raw_alpha > 0 else 0.0
        y_soft = one_hot_label(int(c), int(num_classes))
        if alpha > 0:
            y_soft = alpha * one_hot_label(int(target_class), int(num_classes)) + (1.0 - alpha) * one_hot_label(int(c), int(num_classes))
        soft_labels[int(c)] = y_soft.float().cpu()
        alpha_by_class[int(c)] = alpha
    return soft_labels, alpha_by_class


class BoTPADataset(Dataset):
    """Dataset wrapper that applies BoTPA labels only at training time."""
    def __init__(self, base_dataset, source_class, target_class, intermediate_soft_labels, num_classes):
        self.base_dataset = base_dataset
        self.source_class = int(source_class)
        self.target_class = int(target_class)
        self.num_classes = int(num_classes)
        self.intermediate_soft_labels = {
            int(k): v.detach().cpu().float()
            for k, v in (intermediate_soft_labels or {}).items()
        }

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        y_int = int(y.item()) if torch.is_tensor(y) else int(y)
        if y_int == self.source_class:
            y_soft = one_hot_label(self.target_class, self.num_classes)
        elif y_int in self.intermediate_soft_labels:
            y_soft = self.intermediate_soft_labels[y_int]
        else:
            y_soft = one_hot_label(y_int, self.num_classes)
        return x, y_soft


def train_one_epoch_botpa(
    model,
    dataloader,
    epochs,
    lr,
    device,
    botpa_metadata,
    global_prototype=None,
    warmup_lr_scale=0.5,
    grad_clip=5.0,
    num_workers=0,
    pin_memory=True,
):
    """Local training for a malicious client under BoTPA."""
    if botpa_metadata is None:
        raise ValueError("botpa_metadata must be prepared before malicious local training.")

    local_model, align_loss_value = pre_update_with_global_prototype(
        model=model,
        dataloader=dataloader,
        lr=lr,
        device=device,
        global_prototype=global_prototype,
        warmup_lr_scale=warmup_lr_scale,
    )
    if align_loss_value is not None:
        print(f"[BoTPA] Prototype pre-update done. Align loss: {align_loss_value:.4f}")

    botpa_dataset = BoTPADataset(
        base_dataset=dataloader.dataset,
        source_class=botpa_metadata["source_class"],
        target_class=botpa_metadata["target_class"],
        intermediate_soft_labels=botpa_metadata["soft_labels"],
        num_classes=botpa_metadata["num_classes"],
    )
    botpa_loader = DataLoader(
        botpa_dataset,
        batch_size=getattr(dataloader, "batch_size", 1024),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )

    optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    local_model.train()
    epoch_losses = []

    for epoch in tqdm(range(int(epochs)), desc="BoTPA local epochs", unit="epoch", leave=False):
        total_loss, num_batches = 0.0, 0
        for inputs, soft_targets in tqdm(botpa_loader, desc=f"BoTPA epoch {epoch + 1}/{epochs}", unit="batch", leave=False):
            inputs = inputs.to(device)
            soft_targets = soft_targets.to(device).float()
            optimizer.zero_grad()
            _, logits = _forward_features_logits(local_model, inputs)
            loss = soft_cross_entropy(logits, soft_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=float(grad_clip))
            optimizer.step()
            total_loss += float(loss.item())
            num_batches += 1
        epoch_losses.append(total_loss / max(num_batches, 1))

    mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

    # Prototype extraction uses the original local inputs. Labels are irrelevant here.
    all_hidden = []
    local_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting BoTPA prototype", unit="batch", leave=False):
            inputs, _ = batch
            inputs = inputs.to(device)
            hidden_output, _ = _forward_features_logits(local_model, inputs)
            if hidden_output is None:
                _, logits = _forward_features_logits(local_model, inputs)
                hidden_output = logits
            all_hidden.append(hidden_output.detach())

    if not all_hidden:
        raise RuntimeError("Cannot extract BoTPA prototype from an empty client dataloader.")

    mean_prototype = torch.mean(torch.cat(all_hidden, dim=0), dim=0)
    local_state_dict = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}

    botpa_info = {
        "source_class": int(botpa_metadata["source_class"]),
        "target_class": int(botpa_metadata["target_class"]),
        "intermediate_classes": list(map(int, botpa_metadata["intermediate_classes"])),
        "soft_label_alpha": {int(k): float(v) for k, v in botpa_metadata.get("soft_label_alpha", {}).items()},
        "mean_soft_ce_loss": float(mean_loss),
    }
    return mean_prototype, mean_loss, local_state_dict, botpa_info


def prepare_botpa_metadata(
    client_loaders,
    malicious_client_ids,
    base_model,
    config_model,
    checkpoint_dir=None,
    force_recompute=False,
):
    """Run the BoTPA pre-training stage once and save botpa_metadata.pt."""
    checkpoint_dir = checkpoint_dir or getattr(config_model, "checkpoint_dir", ".")
    os.makedirs(checkpoint_dir, exist_ok=True)
    metadata_path = getattr(config_model, "botpa_metadata_path", None) or os.path.join(checkpoint_dir, "botpa_metadata.pt")

    if os.path.exists(metadata_path) and not force_recompute:
        print(f"Loading existing BoTPA metadata: {metadata_path}")
        metadata = _safe_torch_load(metadata_path, map_location="cpu")
        print(f"Loaded BoTPA intermediate classes: {metadata.get('intermediate_classes')}")
        return metadata

    source_class = int(getattr(config_model, "botpa_source_class"))
    target_class = int(getattr(config_model, "botpa_target_class"))
    num_classes = int(getattr(config_model, "num_classes"))
    num_intermediate = int(getattr(config_model, "botpa_num_intermediate_classes", 2))
    seed = int(getattr(config_model, "botpa_seed", 42))

    print("\n================ BoTPA PREPARATION ================")
    print(f"Loaded poisoned clients: {sorted(map(int, malicious_client_ids))[:50]}{' ...' if len(malicious_client_ids) > 50 else ''}")
    print(f"BoTPA source -> target: {source_class} -> {target_class}")

    class_data = sample_malicious_data_by_class(
        client_loaders=client_loaders,
        malicious_client_ids=malicious_client_ids,
        num_classes=num_classes,
        max_samples_per_class=getattr(config_model, "botpa_surrogate_max_samples_per_class", 4000),
        seed=seed,
        verbose=True,
    )

    surrogate, w_mid, w_conv, surrogate_info = train_botpa_surrogate(
        base_model=base_model,
        class_data_by_original_label=class_data,
        source_class=source_class,
        target_class=target_class,
        config_model=config_model,
        device=getattr(config_model, "device"),
    )

    contribution_scores = compute_contribution_similarity(
        base_model=base_model,
        w_mid=w_mid,
        class_data_by_original_label=class_data,
        source_class=source_class,
        target_class=target_class,
        num_classes=num_classes,
        device=getattr(config_model, "device"),
        grad_samples_per_class=getattr(config_model, "botpa_grad_samples_per_class", 24),
        seed=seed,
    )
    intermediate_classes = select_intermediate_classes(contribution_scores, num_intermediate)

    feature_scores = compute_feature_similarity(
        base_model=base_model,
        w_conv=w_conv,
        class_data_by_original_label=class_data,
        source_class=source_class,
        intermediate_classes=intermediate_classes,
        device=getattr(config_model, "device"),
        feature_samples_per_class=getattr(config_model, "botpa_feature_samples_per_class", 512),
        seed=seed,
        feature_layer=getattr(config_model, "botpa_feature_layer", "logits"),
    )
    soft_labels, alpha_by_class = build_botpa_soft_labels(
        feature_scores=feature_scores,
        intermediate_classes=intermediate_classes,
        num_classes=num_classes,
        target_class=target_class,
    )

    metadata = {
        "attack": "BoTPA",
        "source_class": source_class,
        "target_class": target_class,
        "num_classes": num_classes,
        "intermediate_classes": list(map(int, intermediate_classes)),
        "contribution_scores": {int(k): float(v) for k, v in contribution_scores.items()},
        "feature_scores": {int(k): float(v) for k, v in feature_scores.items()},
        "soft_labels": {int(k): v.detach().cpu() for k, v in soft_labels.items()},
        "soft_label_alpha": {int(k): float(v) for k, v in alpha_by_class.items()},
        "malicious_client_ids": sorted(map(int, malicious_client_ids)),
        "sample_config": {
            "surrogate_max_samples_per_class": getattr(config_model, "botpa_surrogate_max_samples_per_class", 4000),
            "grad_samples_per_class": getattr(config_model, "botpa_grad_samples_per_class", 24),
            "feature_samples_per_class": getattr(config_model, "botpa_feature_samples_per_class", 512),
            "feature_layer": getattr(config_model, "botpa_feature_layer", "logits"),
        },
        "surrogate_info": surrogate_info,
    }

    torch.save(metadata, metadata_path)
    print("\n=== BoTPA metadata saved ===")
    print(f"Path: {metadata_path}")
    print(f"Selected intermediate classes: {metadata['intermediate_classes']}")
    print(f"Contribution scores: {metadata['contribution_scores']}")
    print(f"Feature scores / soft-label alpha: {metadata['soft_label_alpha']}")
    print("====================================================\n")
    return metadata


def compute_targeted_asr(model, test_loader, source_class, target_class, device):
    """Compute ASR = P(model predicts target_class | true label is source_class)."""
    model.eval()
    source_total, source_to_target = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long().view(-1)
            mask = labels == int(source_class)
            if mask.sum().item() == 0:
                continue
            _, logits = _forward_features_logits(model, inputs)
            preds = logits.argmax(dim=1)
            source_total += int(mask.sum().item())
            source_to_target += int(((preds == int(target_class)) & mask).sum().item())
    return float(source_to_target / max(source_total, 1)), int(source_to_target), int(source_total)


def evaluate_botpa_asr_all_clients(client_models, test_loader, source_class, target_class, device, output_csv=None, round_num=None):
    """Evaluate source->target ASR for every client model and optionally save a CSV."""
    rows = []
    for cid, model in sorted(client_models.items()):
        asr, n_success, n_source = compute_targeted_asr(model, test_loader, source_class, target_class, device)
        rows.append({
            "round": None if round_num is None else int(round_num) + 1,
            "client_id": int(cid),
            "source_class": int(source_class),
            "target_class": int(target_class),
            "ASR": float(asr),
            "source_to_target": int(n_success),
            "num_source_samples": int(n_source),
        })
    df = pd.DataFrame(rows)
    if output_csv is not None:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
            df.to_csv(output_csv, mode="a", header=False, index=False)
        else:
            df.to_csv(output_csv, mode="w", header=True, index=False)
        print(f"Saved BoTPA ASR CSV: {output_csv}")
    if not df.empty:
        print(f"Mean BoTPA ASR {source_class}->{target_class}: {df['ASR'].mean():.6f}")
    return df

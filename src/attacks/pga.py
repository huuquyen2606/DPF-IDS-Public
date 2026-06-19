"""Projected Gradient Ascent model-poisoning attack utilities."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.optim as optim
from tqdm.auto import tqdm

from ..dpf_ids.alignment import pre_update_with_global_prototype
from ..dpf_ids.prototype import extract_prototype_limited

def clone_state_dict_cpu(model):
    """Return a detached CPU copy of a model state_dict."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def state_delta(after_state, before_state):
    """Compute after_state - before_state for floating-point tensors."""
    delta = {}
    for name, before_tensor in before_state.items():
        after_tensor = after_state[name].detach().cpu()
        if torch.is_floating_point(before_tensor):
            delta[name] = after_tensor - before_tensor
        else:
            # Non-floating buffers are copied from the after state when applied.
            delta[name] = after_tensor.clone()
    return delta


def delta_l2_norm(delta):
    """Compute L2 norm of a state-dict delta over floating-point tensors."""
    sq_sum = 0.0
    for tensor in delta.values():
        if torch.is_floating_point(tensor):
            sq_sum += float(torch.sum(tensor.float() ** 2).item())
    return float(sq_sum ** 0.5)


def scale_delta(delta, scale):
    """Scale only floating-point tensors of a state-dict delta."""
    scaled = {}
    for name, tensor in delta.items():
        if torch.is_floating_point(tensor):
            scaled[name] = tensor * float(scale)
        else:
            scaled[name] = tensor.clone()
    return scaled


def apply_delta_to_model(model, before_state, delta, device):
    """
    Load before_state + delta into model.
    Non-floating buffers are taken from before_state unless overwritten.
    """
    new_state = {}
    for name, before_tensor in before_state.items():
        if name in delta and torch.is_floating_point(before_tensor):
            new_state[name] = (before_tensor + delta[name]).to(device)
        else:
            new_state[name] = before_tensor.to(device)

    model.load_state_dict(new_state)
    return model


def estimate_self_benign_update_norm(
    model,
    dataloader,
    lr,
    criterion,
    device,
    epochs=1,
    max_batches=None,
    grad_clip=5.0,
):
    """
    Estimate tau from the compromised client's own benign update.
    This approximates the paper's tau when benign updates from other clients
    are not available and avoids requiring model updates to be sent to server.
    """
    benign_model = copy.deepcopy(model).to(device)
    before_state = clone_state_dict_cpu(benign_model)

    optimizer = optim.SGD(
        benign_model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
    )
    benign_model.train()

    seen_batches = 0
    for _ in range(int(epochs)):
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            _, logits = benign_model(inputs)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(benign_model.parameters(), max_norm=float(grad_clip))
            optimizer.step()

            seen_batches += 1
            if max_batches is not None and seen_batches >= int(max_batches):
                break
        if max_batches is not None and seen_batches >= int(max_batches):
            break

    after_state = clone_state_dict_cpu(benign_model)
    benign_delta = state_delta(after_state, before_state)
    tau = delta_l2_norm(benign_delta)

    return float(tau)


def pga_project_update(
    raw_delta,
    tau,
    gamma=1e4,
    eps=1e-12,
    verbose=False,
):
    """
    Fixed-gamma projection function for the controlled PGA experiment.

    Paper Algorithm 2 searches gamma over a range to maximize poisoned
    aggregate deviation. In this experiment, we intentionally use ONE
    fixed amplification factor:
        gamma = 1e4

    Projection rule:
        1) compute raw malicious update Delta from gradient ascent;
        2) scale Delta to have norm tau;
        3) amplify once by fixed gamma.

    No server-side loss, gradient, or model parameter is required for
    detection. This function is executed only inside the compromised
    client simulation.
    """
    raw_norm = delta_l2_norm(raw_delta)

    if raw_norm < eps:
        return raw_delta, {
            "raw_update_norm": float(raw_norm),
            "tau": float(tau),
            "base_scale": 1.0,
            "gamma": float(gamma),
            "final_update_norm": float(raw_norm),
        }

    tau = float(max(tau, eps))
    gamma = float(gamma)

    base_scale = tau / (raw_norm + eps)
    final_scale = base_scale * gamma
    projected_delta = scale_delta(raw_delta, final_scale)

    if verbose:
        print(
            f"PGA fixed projection: raw_norm={raw_norm:.6f}, tau={tau:.6f}, "
            f"base_scale={base_scale:.6f}, gamma={gamma:.4e}, "
            f"final_norm={delta_l2_norm(projected_delta):.6f}"
        )

    return projected_delta, {
        "raw_update_norm": float(raw_norm),
        "tau": float(tau),
        "base_scale": float(base_scale),
        "gamma": float(gamma),
        "final_update_norm": float(delta_l2_norm(projected_delta)),
    }


def train_one_epoch_pga(
    model,
    dataloader,
    epochs,
    lr,
    criterion,
    device,
    global_prototype=None,
    previous_prototype=None,
    warmup_lr_scale=0.5,
    pga_ascent_epochs=1,
    pga_lr_multiplier=1.0,
    pga_tau_mode="self_benign",
    pga_projection_radius=10.0,
    pga_gamma=1e4,
    pga_projection_batches=1,
    pga_tau_batches=None,
    grad_clip=5.0,
    verbose=False,
):
    """
    Local compromised-client PGA training.

    Benign local training:
        minimize CE loss.

    PGA local training:
        maximize CE loss by minimizing -CE loss.
        Then project/scale the resulting model update.

    Returns the same interface as train_one_epoch:
        mean_prototype, mean_loss, local_state_dict

    This is for controlled experiments. The server still receives only
    the final prototype, not model params, loss, or gradients.
    """
    # Start from the same pre-update step as benign clients.
    local_model, align_loss_value = pre_update_with_global_prototype(
        model=model,
        dataloader=dataloader,
        lr=lr,
        device=device,
        global_prototype=global_prototype,
        warmup_lr_scale=warmup_lr_scale,
    )

    if align_loss_value is not None and verbose:
        print(f"[PGA] Prototype pre-update done. Align loss: {align_loss_value:.4f}")

    before_state = clone_state_dict_cpu(local_model)

    # Step A: stochastic gradient ascent.
    poisoned_model = copy.deepcopy(local_model).to(device)
    optimizer = optim.SGD(
        poisoned_model.parameters(),
        lr=lr * float(pga_lr_multiplier),
        momentum=0.9,
        weight_decay=1e-4,
    )
    poisoned_model.train()

    ce_losses = []
    for epoch in tqdm(range(int(pga_ascent_epochs)), desc="PGA ascent epochs", unit="epoch", leave=False):
        for batch in tqdm(dataloader, desc=f"PGA epoch {epoch + 1}/{pga_ascent_epochs}", unit="batch", leave=False):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            _, logits = poisoned_model(inputs)
            ce_loss = criterion(logits, labels)

            # Minimize negative CE = maximize CE.
            ascent_loss = -ce_loss
            ascent_loss.backward()

            torch.nn.utils.clip_grad_norm_(poisoned_model.parameters(), max_norm=float(grad_clip))
            optimizer.step()

            ce_losses.append(float(ce_loss.item()))

    after_ascent_state = clone_state_dict_cpu(poisoned_model)
    raw_delta = state_delta(after_ascent_state, before_state)

    # Step B: compute tau.
    if str(pga_tau_mode).lower() == "self_benign":
        tau = estimate_self_benign_update_norm(
            model=local_model,
            dataloader=dataloader,
            lr=lr,
            criterion=criterion,
            device=device,
            epochs=1,
            max_batches=pga_tau_batches,
            grad_clip=grad_clip,
        )
        if tau <= 1e-12:
            tau = float(pga_projection_radius)
    elif str(pga_tau_mode).lower() == "fixed":
        tau = float(pga_projection_radius)
    else:
        raise ValueError("pga_tau_mode must be either 'self_benign' or 'fixed'.")

    # Step C: fixed-gamma PGA projection.
    projected_delta, pga_info = pga_project_update(
        raw_delta=raw_delta,
        tau=tau,
        gamma=pga_gamma,
        verbose=verbose,
    )

    # Step D: apply projected malicious update.
    final_model = copy.deepcopy(local_model).to(device)
    final_model = apply_delta_to_model(final_model, before_state, projected_delta, device)

    mean_prototype = extract_prototype_limited(
        final_model,
        dataloader,
        device=device,
        max_batches=None,
    )

    local_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in final_model.state_dict().items()
    }

    mean_loss = float(np.mean(ce_losses)) if ce_losses else 0.0
    if verbose:
        print(f"[PGA] mean CE before/ascent tracking: {mean_loss:.4f}")
        print(f"[PGA] info: {pga_info}")

    return mean_prototype, mean_loss, local_state_dict, pga_info

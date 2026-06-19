"""Prototype alignment utilities used before local client training."""

from __future__ import annotations

import copy

import torch
import torch.optim as optim

def pre_update_with_global_prototype(model, dataloader, lr, device, global_prototype=None, warmup_lr_scale=0.5):
    updated_model = copy.deepcopy(model).to(device)

    if global_prototype is None:
        return updated_model, None

    proto_optimizer = optim.SGD(
        updated_model.parameters(),
        lr=lr * float(warmup_lr_scale),
        momentum=0.9,
        weight_decay=1e-4,
    )
    updated_model.train()
    global_proto_vec = global_prototype.detach().to(device).view(-1).float()

    align_loss_value = None
    for batch in dataloader:
        inputs, _ = batch
        inputs = inputs.to(device)

        proto_optimizer.zero_grad()
        hidden_output, _ = updated_model(inputs)
        local_batch_proto = hidden_output.mean(dim=0).view(-1).float()
        proto_align_loss = torch.norm(local_batch_proto - global_proto_vec, p=2)
        proto_align_loss.backward()
        torch.nn.utils.clip_grad_norm_(updated_model.parameters(), max_norm=5.0)
        proto_optimizer.step()

        align_loss_value = float(proto_align_loss.item())
        break

    return updated_model, align_loss_value

"""Local client training loops for DPF-IDS."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.optim as optim
from tqdm.auto import tqdm

from .attacks.botpa import train_one_epoch_botpa
from .attacks.pga import (
    apply_delta_to_model,
    clone_state_dict_cpu,
    delta_l2_norm,
    scale_delta,
    state_delta,
    train_one_epoch_pga,
)
from .dpf_ids.alignment import pre_update_with_global_prototype
from .dpf_ids.prototype import extract_prototype_limited
from .utils.io import save_full_checkpoint

def train_one_epoch(model, dataloader, epochs, lr, criterion, device, global_prototype=None, warmup_lr_scale=0.5):
    # Pre-update first, then local training (classification only).
    local_model, align_loss_value = pre_update_with_global_prototype(
        model=model,
        dataloader=dataloader,
        lr=lr,
        device=device,
        global_prototype=global_prototype,
        warmup_lr_scale=warmup_lr_scale,
    )
    if align_loss_value is not None:
        print(f"Prototype pre-update done. Align loss: {align_loss_value:.4f}")

    optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4) 
    local_model.train()
    epoch_losses = []

    for epoch in tqdm(range(epochs), desc="Local epochs", unit="epoch", leave=False):
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", leave=False):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            hidden_output, output = local_model(inputs)

            # Local training uses only supervised classification loss.
            loss = criterion(output, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        agg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_losses.append(agg_loss)

    mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
    print(f"Epoch {epochs}/{epochs}, Loss: {mean_loss:.4f}")

    # Take prototype (mean hidden representation over local dataset).
    all_hidden = []
    local_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting prototype", unit="batch", leave=False):
            inputs, _ = batch
            inputs = inputs.to(device)
            hidden_output, _ = local_model(inputs)
            all_hidden.append(hidden_output)

    mean_prototype = torch.mean(torch.cat(all_hidden, dim=0), dim=0)
    print(f"Mean prototype shape: {mean_prototype.shape}")

    local_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in local_model.state_dict().items()
    }

    return mean_prototype, mean_loss, local_state_dict


def run_local_training_round(
    client_loaders,
    base_model,
    config_model,
    round_idx=0,
    global_prototype=None,
    client_models=None,
    resume_state=None,
    checkpoint_dir=None,
    checkpoint_algorithm="dfl_prototype",
    prev_clusters=None,
    prev_client_prototypes=None,
    poison_client_ids=None,
    enable_pga_attack=False,
    enable_botpa_attack=False,
    botpa_metadata=None,
):
    """
    Train local models for one round and keep per-client model states across rounds.

    Benign clients:
        use train_one_epoch(...), i.e., normal SGD minimizing CE loss.

    Poisoned clients:
        - if enable_botpa_attack=True: use BoTPA targeted poisoning labels;
        - else if enable_pga_attack=True: use the old PGA local attack;
        - else: train benignly.

    A client flagged as poisoned by detection in a previous round is NOT
    permanently removed. Attack participation is controlled only by
    poison_client_ids, the experimental ground-truth compromised clients.
    """
    resume_state = resume_state or {}
    client_prototypes = resume_state.get("client_prototypes", {}) or {}
    client_losses = resume_state.get("client_losses", {}) or {}
    client_sample_counts = resume_state.get("client_sample_counts", {}) or {}
    client_attack_info = resume_state.get("client_attack_info", {}) or {}

    poison_client_ids = set(int(x) for x in (poison_client_ids or []))

    if client_models is None:
        client_models = {
            cid: copy.deepcopy(base_model).to(config_model.device)
            for cid in client_loaders.keys()
        }

    ordered_client_ids = sorted(client_loaders.keys())
    start_pos = int(resume_state.get("next_client_pos", 0) or 0)
    start_pos = max(0, min(start_pos, len(ordered_client_ids)))

    attack_name = "BoTPA" if enable_botpa_attack else ("PGA" if enable_pga_attack else "none")
    num_poison_selected = len([cid for cid in ordered_client_ids if cid in poison_client_ids])
    print(f"=== Round {round_idx + 1}: Local train + all-to-all broadcast ===")
    print(f"Attack enabled: {attack_name} | poison clients in this experiment: {num_poison_selected}/{len(ordered_client_ids)}")
    if enable_botpa_attack and botpa_metadata is not None:
        print(
            f"BoTPA source -> target: {botpa_metadata['source_class']} -> {botpa_metadata['target_class']} | "
            f"intermediate={botpa_metadata.get('intermediate_classes', [])}"
        )
    if global_prototype is not None:
        print("Using received global prototype for pre-update before local training.")
    if start_pos > 0:
        print(f"Resuming round {round_idx + 1} from client index {start_pos}.")

    for pos in tqdm(range(start_pos, len(ordered_client_ids)), desc="Training clients", unit="client"):
        cid = ordered_client_ids[pos]
        dataloader = client_loaders[cid]
        is_ground_truth_poison_client = cid in poison_client_ids
        is_botpa_client = bool(enable_botpa_attack and is_ground_truth_poison_client)
        is_pga_client = bool((not enable_botpa_attack) and enable_pga_attack and is_ground_truth_poison_client)

        if is_botpa_client:
            # ---------------------------------------------------------
            # Save the round-start model state before BoTPA local training.
            # This state is the anchor for optional malicious update scaling:
            #     w_scaled = w_start + gamma * (w_local - w_start)
            # ---------------------------------------------------------
            round_start_state = clone_state_dict_cpu(client_models[cid])

            local_proto, local_loss, local_state_dict, botpa_info = train_one_epoch_botpa(
                model=client_models[cid],
                dataloader=dataloader,
                epochs=config_model.num_epochs,
                lr=config_model.learning_rate,
                device=config_model.device,
                botpa_metadata=botpa_metadata,
                global_prototype=global_prototype,
                warmup_lr_scale=0.5,
                grad_clip=getattr(config_model, "botpa_grad_clip", 5.0),
                num_workers=getattr(config_model, "botpa_num_workers", 0),
                pin_memory=getattr(config_model, "botpa_pin_memory", True),
            )

            # ---------------------------------------------------------
            # Optional BoTPA model-update amplification.
            # gamma = 1.0 keeps pure BoTPA.
            # gamma > 1.0 turns the attack into BoTPA + update scaling.
            # ---------------------------------------------------------
            gamma = float(getattr(config_model, "botpa_gamma", 1.0))

            if gamma != 1.0:
                raw_delta = state_delta(
                    after_state=local_state_dict,
                    before_state=round_start_state,
                )
                raw_update_norm = delta_l2_norm(raw_delta)

                scaled_delta = scale_delta(
                    delta=raw_delta,
                    scale=gamma,
                )
                scaled_update_norm = delta_l2_norm(scaled_delta)

                client_models[cid] = apply_delta_to_model(
                    model=client_models[cid],
                    before_state=round_start_state,
                    delta=scaled_delta,
                    device=config_model.device,
                )

                local_state_dict = clone_state_dict_cpu(client_models[cid])

                # Recompute the prototype from the scaled model.
                # Without this step, the server would aggregate a scaled model
                # but the DFL-IDS detection scores would still use the
                # pre-scaling prototype, which makes TP/FP/FN/TN less faithful.
                local_proto = extract_prototype_limited(
                    model=client_models[cid],
                    dataloader=dataloader,
                    device=config_model.device,
                    max_batches=getattr(config_model, "botpa_recompute_proto_max_batches", None),
                )

                botpa_info["botpa_gamma"] = float(gamma)
                botpa_info["update_scaled"] = True
                botpa_info["raw_update_norm"] = float(raw_update_norm)
                botpa_info["scaled_update_norm"] = float(scaled_update_norm)
                botpa_info["prototype_recomputed_after_scaling"] = True

                print(
                    f"[BoTPA Scaling] Client {cid}: "
                    f"gamma={gamma}, "
                    f"raw_norm={raw_update_norm:.6f}, "
                    f"scaled_norm={scaled_update_norm:.6f}, "
                    f"prototype_recomputed=True"
                )
            else:
                botpa_info["botpa_gamma"] = float(gamma)
                botpa_info["update_scaled"] = False
                botpa_info["raw_update_norm"] = None
                botpa_info["scaled_update_norm"] = None
                botpa_info["prototype_recomputed_after_scaling"] = False

            client_attack_info[cid] = {
                "round": int(round_idx + 1),
                "attack": "BoTPA",
                **botpa_info,
            }

        elif is_pga_client:
            previous_proto_for_client = None
            if prev_client_prototypes is not None and cid in prev_client_prototypes:
                previous_proto_for_client = prev_client_prototypes[cid]

            local_proto, local_loss, local_state_dict, pga_info = train_one_epoch_pga(
                model=client_models[cid],
                dataloader=dataloader,
                epochs=config_model.num_epochs,
                lr=config_model.learning_rate,
                criterion=config_model.criterion,
                device=config_model.device,
                global_prototype=global_prototype,
                previous_prototype=previous_proto_for_client,
                warmup_lr_scale=0.5,
                pga_ascent_epochs=config_model.pga_ascent_epochs,
                pga_lr_multiplier=config_model.pga_lr_multiplier,
                pga_tau_mode=config_model.pga_tau_mode,
                pga_projection_radius=config_model.pga_projection_radius,
                pga_gamma=config_model.pga_gamma,
                pga_projection_batches=config_model.pga_projection_batches,
                pga_tau_batches=config_model.pga_tau_batches,
                grad_clip=config_model.pga_grad_clip,
                verbose=config_model.pga_verbose,
            )
            client_attack_info[cid] = {
                "round": int(round_idx + 1),
                "attack": "PGA",
                **pga_info,
            }

        else:
            local_proto, local_loss, local_state_dict = train_one_epoch(
                model=client_models[cid],
                dataloader=dataloader,
                epochs=config_model.num_epochs,
                lr=config_model.learning_rate,
                criterion=config_model.criterion,
                device=config_model.device,
                global_prototype=global_prototype,
            )
            client_attack_info[cid] = {
                "round": int(round_idx + 1),
                "attack": "benign",
            }

        client_models[cid].load_state_dict(local_state_dict)
        client_prototypes[cid] = local_proto.detach().cpu()
        client_losses[cid] = float(local_loss)
        client_sample_counts[cid] = int(len(dataloader.dataset))

        if checkpoint_dir:
            save_full_checkpoint(
                model=client_models,
                optimizer=None,
                round_num=round_idx,
                checkpoint_dir=checkpoint_dir,
                algorithm=checkpoint_algorithm,
                loss=None,
                metrics=None,
                scheduler=None,
                extra_state={
                    "phase": "in_round",
                    "next_client_pos": pos + 1,
                    "client_prototypes": client_prototypes,
                    "client_losses": client_losses,
                    "client_sample_counts": client_sample_counts,
                    "client_attack_info": client_attack_info,
                    "poison_client_ids": sorted(poison_client_ids),
                    "attack_name": attack_name,
                    "botpa_metadata": botpa_metadata,
                    "aggregated_server_proto": global_prototype,
                    "prev_clusters": prev_clusters,
                    "prev_client_prototypes": prev_client_prototypes,
                },
            )

    print("\nLocal training done for all clients.")
    print("Per-client loss:", {cid: round(loss, 4) for cid, loss in client_losses.items()})
    print("Each client model is kept for next round training.")

    return client_prototypes, client_losses, client_sample_counts, client_models, client_attack_info

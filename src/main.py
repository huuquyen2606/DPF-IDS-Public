"""End-to-end DPF-IDS training pipeline."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .attacks.botpa import evaluate_botpa_asr_all_clients, prepare_botpa_metadata
from .data.dataset import build_client_loaders, load_test_data
from .dpf_ids.aggregation import stage2_global_aggregation
from .dpf_ids.alignment import pre_update_with_global_prototype
from .dpf_ids.detection import build_benign_cluster_prototypes, detect_poisoned_clients_prototype_only, save_detection_report
from .dpf_ids.prototype import compute_lambda_g_robust, coordinate_wise_median, select_closest_prototype
from .dpf_ids.topology import optimize_topology
from .models.ffnn import FFNN
from .train import run_local_training_round
from .utils.io import load_full_checkpoint, save_full_checkpoint
from .utils.logging import (
    load_poisoned_clients_for_round,
    log_poisoned_clients,
    resolve_true_poison_client_ids,
    save_poison_detection_txt_mapping,
    write_client_id_txt,
)
from .utils.seed import set_seed

DATA_MODE_OPTIONS = ("benign", "data_poison")


@dataclass
class FrameworkConfig:
    data_mode: str = "data_poison"
    batch_size: int = 1024
    learning_rate: float = 0.01
    num_epochs: int = 5
    input_size: int = 39
    hidden_size: int = 128
    num_classes: int = 8
    num_clients: int = 500
    checkpoint_dir: str = "."
    checkpoint_algorithm: str = "FFNN"
    data_train: str = ""
    data_test: str = ""
    true_poison_index_path: str = ""
    true_poison_index_base: str = "auto"
    seed: int = 42
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    criterion: nn.Module = field(default_factory=nn.CrossEntropyLoss)

    topology_metric: str = "cosine"
    lambda_g_tau: float = 2.5
    reuse_prev_clusters: bool = False
    detect_tau_E_start: float = 1.5
    detect_tau_C_start: float = 1.5
    detect_tau_D_start: float = 2.0
    detect_tau_N_start: float = 1.5
    detect_tau_E_min: float = 0.7
    detect_tau_C_min: float = 0.7
    detect_tau_D_min: float = 1.0
    detect_tau_N_min: float = 0.7
    lambda_decay_beta: float = 2.5
    lambda_min_ratio: float = 0.20
    detect_vote_threshold: int = 1
    min_cluster_size_for_local_threshold: int = 4
    cluster_contamination_ratio: float = 0.3
    enable_cluster_centroid_filter: bool = False

    enable_pga_attack: bool = False
    pga_ascent_epochs: int = 1
    pga_lr_multiplier: float = 1.0
    pga_tau_mode: str = "fixed"
    pga_projection_radius: float = 10.0
    pga_gamma: float = 1e4
    pga_projection_batches: int = 1
    pga_tau_batches: int = 1
    pga_grad_clip: float = 5.0
    pga_verbose: bool = False

    enable_botpa_attack: bool = False
    botpa_source_class: int = 2
    botpa_target_class: int = 0
    botpa_num_intermediate_classes: int = 3
    botpa_gamma: float = 1.0
    botpa_recompute_proto_max_batches: int | None = None
    botpa_surrogate_max_samples_per_class: int = 4000
    botpa_grad_samples_per_class: int = 24
    botpa_feature_samples_per_class: int = 512
    botpa_surrogate_epochs: int = 5
    botpa_surrogate_middle_epoch: int | None = None
    botpa_surrogate_batch_size: int = 1024
    botpa_surrogate_optimizer: str = "adam"
    botpa_surrogate_lr: float = 1e-3
    botpa_surrogate_weight_decay: float = 1e-4
    botpa_feature_layer: str = "logits"
    botpa_grad_clip: float = 5.0
    botpa_num_workers: int = 0
    botpa_pin_memory: bool = True
    botpa_seed: int = 42
    botpa_force_recompute_metadata: bool = False
    botpa_metadata_path: str | None = None
    evaluate_botpa_asr_each_round: bool = False

    def __post_init__(self) -> None:
        if self.data_mode not in DATA_MODE_OPTIONS:
            raise ValueError(f"data_mode must be one of {DATA_MODE_OPTIONS}, got {self.data_mode}")
        if self.botpa_metadata_path is None:
            self.botpa_metadata_path = os.path.join(self.checkpoint_dir, "botpa_metadata.pt")


def build_base_model(config: FrameworkConfig) -> FFNN:
    return FFNN(in_features=config.input_size, num_classes=config.num_classes).to(config.device)


def _load_true_poison_ids(config: FrameworkConfig, client_data_info, mapping_log_dir: str):
    os.makedirs(mapping_log_dir, exist_ok=True)
    if config.data_mode == "data_poison" and config.true_poison_index_path:
        true_ids, mapping_info = resolve_true_poison_client_ids(
            poison_index_path=config.true_poison_index_path,
            client_data_info=client_data_info,
            num_clients=config.num_clients,
            index_base=config.true_poison_index_base,
            verbose=True,
        )
    else:
        true_ids = []
        mapping_info = {
            "data_mode": config.data_mode,
            "poison_index_path": config.true_poison_index_path or None,
            "raw_count": 0,
            "mapped_count": 0,
            "chosen_index_base": None,
            "note": "No ground-truth poison .txt file loaded.",
        }

    write_client_id_txt(true_ids, os.path.join(mapping_log_dir, "ground_truth_poison_client_ids_mapped.txt"))
    with open(os.path.join(mapping_log_dir, "ground_truth_mapping_info.txt"), "w", encoding="utf-8") as f:
        for key, value in mapping_info.items():
            f.write(f"{key}: {value}\n")
    return true_ids, mapping_info


def _resume_state(config: FrameworkConfig, base_model, client_loaders):
    initial_models = {cid: copy.deepcopy(base_model).to(config.device) for cid in client_loaders.keys()}
    loaded_round, _, _, extra_state = load_full_checkpoint(
        model=initial_models,
        checkpoint_dir=config.checkpoint_dir,
        algorithm=config.checkpoint_algorithm,
        optimizer=None,
        scheduler=None,
        map_location=config.device,
        return_extra_state=True,
        strict_rng_restore=True,
    )

    state = {
        "start_round": 0,
        "client_models": None,
        "aggregated_server_proto": None,
        "prev_clusters": None,
        "prev_client_prototypes": None,
        "threshold_state": {},
        "resume_round_state": {},
    }
    if loaded_round <= 0 and not extra_state:
        return state

    state["client_models"] = initial_models
    state["aggregated_server_proto"] = extra_state.get("aggregated_server_proto")
    state["prev_clusters"] = extra_state.get("prev_clusters")
    state["prev_client_prototypes"] = extra_state.get("prev_client_prototypes")
    state["threshold_state"] = extra_state.get("threshold_state", {}) or {}

    if extra_state.get("phase") == "in_round":
        state["start_round"] = max(int(loaded_round) - 1, 0)
        state["resume_round_state"] = {
            "next_client_pos": int(extra_state.get("next_client_pos", 0) or 0),
            "client_prototypes": extra_state.get("client_prototypes", {}) or {},
            "client_losses": extra_state.get("client_losses", {}) or {},
            "client_sample_counts": extra_state.get("client_sample_counts", {}) or {},
            "client_attack_info": extra_state.get("client_attack_info", {}) or {},
        }
    else:
        state["start_round"] = int(loaded_round)
    return state


def run_training_pipeline(
    config: FrameworkConfig | None = None,
    num_rounds: int = 11,
    start_round: int | None = None,
    perform_evaluation: bool = False,
) -> dict[str, Any]:
    config = config or FrameworkConfig()
    if not config.data_train:
        raise ValueError("config.data_train must point to client .pt files")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    set_seed(config.seed)

    base_model = build_base_model(config)
    client_loaders, client_data_info = build_client_loaders(
        data_dir=config.data_train,
        num_clients=config.num_clients,
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = None
    if perform_evaluation or (config.enable_botpa_attack and config.evaluate_botpa_asr_each_round):
        if not config.data_test:
            raise ValueError("config.data_test is required for evaluation")
        test_loader, _ = load_test_data(config.data_test, batch_size=50000)

    resumed = _resume_state(config, base_model, client_loaders)
    if start_round is not None:
        resumed["start_round"] = int(start_round)

    eval_log_dir = os.path.join(config.checkpoint_dir, "eval_logs")
    detection_log_dir = os.path.join(config.checkpoint_dir, "detection_logs")
    mapping_log_dir = os.path.join(config.checkpoint_dir, "poison_index_mapping_logs")
    lambda_g_path = os.path.join(config.checkpoint_dir, "lambda_g.txt")

    true_poison_client_ids, true_poison_mapping_info = _load_true_poison_ids(config, client_data_info, mapping_log_dir)
    all_client_ids_for_mapping = sorted(client_loaders.keys())

    attack_client_ids = true_poison_client_ids if (config.enable_pga_attack or config.enable_botpa_attack) else []
    botpa_metadata = None
    if config.enable_botpa_attack:
        botpa_metadata = prepare_botpa_metadata(
            client_loaders=client_loaders,
            malicious_client_ids=attack_client_ids,
            base_model=base_model,
            config_model=config,
            checkpoint_dir=config.checkpoint_dir,
            force_recompute=config.botpa_force_recompute_metadata,
        )

    client_models = resumed["client_models"]
    aggregated_server_proto = resumed["aggregated_server_proto"]
    prev_clusters = resumed["prev_clusters"]
    prev_client_prototypes = resumed["prev_client_prototypes"]
    threshold_state = resumed["threshold_state"]
    resume_round_state = resumed["resume_round_state"]
    last_detection_result = None

    for round_num in range(int(resumed["start_round"]), int(num_rounds)):
        print(f"\n=== Round {round_num + 1} ===")
        preupdated_models = None

        if client_models is not None and aggregated_server_proto is not None:
            preupdated_models = {}
            for cid, model in client_models.items():
                updated_model, align_loss_value = pre_update_with_global_prototype(
                    model=model,
                    dataloader=client_loaders[cid],
                    lr=config.learning_rate,
                    device=config.device,
                    global_prototype=aggregated_server_proto,
                    warmup_lr_scale=0.5,
                )
                preupdated_models[cid] = updated_model
                if align_loss_value is not None:
                    print(f"Round {round_num + 1} - Client {cid} pre-update align loss: {align_loss_value:.4f}")

            if perform_evaluation and test_loader is not None:
                from .evaluate import evaluate_all_clients_average

                previous_poisoned_client_ids = load_poisoned_clients_for_round(config.checkpoint_dir, round_num) if round_num > 0 else []
                benign_client_ids = sorted(set(client_loaders.keys()) - set(previous_poisoned_client_ids))
                preupdated_benign_models = {cid: preupdated_models[cid] for cid in benign_client_ids if cid in preupdated_models}
                if preupdated_benign_models:
                    evaluate_all_clients_average(
                        client_models=preupdated_benign_models,
                        test_loader=test_loader,
                        device=config.device,
                        csv_dir=eval_log_dir,
                        tag=f"round_{round_num + 1}_preupdate_detected_benign_prev_round_eval",
                        resume=True,
                        flush_each_client=True,
                    )

        if round_num == int(num_rounds) - 1:
            break

        current_resume_state = (
            resume_round_state
            if (round_num == int(resumed["start_round"]) and bool(resume_round_state.get("next_client_pos")))
            else None
        )
        models_for_training = preupdated_models if preupdated_models is not None else client_models
        client_prototypes, client_losses, _, client_models, client_attack_info = run_local_training_round(
            client_loaders=client_loaders,
            base_model=base_model,
            config_model=config,
            round_idx=round_num,
            global_prototype=aggregated_server_proto,
            client_models=models_for_training,
            resume_state=current_resume_state,
            checkpoint_dir=config.checkpoint_dir,
            checkpoint_algorithm=config.checkpoint_algorithm,
            prev_clusters=prev_clusters,
            prev_client_prototypes=prev_client_prototypes,
            poison_client_ids=attack_client_ids,
            enable_pga_attack=config.enable_pga_attack,
            enable_botpa_attack=config.enable_botpa_attack,
            botpa_metadata=botpa_metadata,
        )
        resume_round_state = {}

        round_reference_proto = coordinate_wise_median(client_prototypes)
        best_client_id = select_closest_prototype(client_prototypes, round_reference_proto, metric=config.topology_metric)
        lambda_g, lambda_g_info = compute_lambda_g_robust(
            client_prototypes=client_prototypes,
            metric=config.topology_metric,
            tau=config.lambda_g_tau,
            verbose=True,
        )
        with open(lambda_g_path, "w", encoding="utf-8") as f:
            f.write(str(lambda_g))

        topology_result = optimize_topology(
            client_prototypes=client_prototypes,
            best_client_id=best_client_id,
            clusters=prev_clusters if config.reuse_prev_clusters else None,
            lambda_g=lambda_g,
            metric=config.topology_metric,
            max_iters=20,
            stable_rounds=2,
            verbose=True,
        )

        detection_result = detect_poisoned_clients_prototype_only(
            client_prototypes=client_prototypes,
            clusters=topology_result["clusters"],
            prev_client_prototypes=prev_client_prototypes,
            round_idx=round_num,
            total_rounds=num_rounds,
            threshold_state=threshold_state,
            tau_E_start=config.detect_tau_E_start,
            tau_C_start=config.detect_tau_C_start,
            tau_D_start=config.detect_tau_D_start,
            tau_N_start=config.detect_tau_N_start,
            tau_E_min=config.detect_tau_E_min,
            tau_C_min=config.detect_tau_C_min,
            tau_D_min=config.detect_tau_D_min,
            tau_N_min=config.detect_tau_N_min,
            lambda_decay_beta=config.lambda_decay_beta,
            lambda_min_ratio=config.lambda_min_ratio,
            vote_threshold=config.detect_vote_threshold,
            min_cluster_size_for_local_threshold=config.min_cluster_size_for_local_threshold,
            cluster_contamination_ratio=config.cluster_contamination_ratio,
            enable_cluster_centroid_filter=config.enable_cluster_centroid_filter,
            verbose=True,
        )
        threshold_state = detection_result.get("threshold_state", threshold_state) or threshold_state
        poisoned_client_ids = detection_result["poisoned_client_ids"]
        benign_client_ids = detection_result["benign_client_ids"]
        save_detection_report(detection_result, round_num, detection_log_dir)

        poisoned_log_file = os.path.join(config.checkpoint_dir, f"poisoned_clients_log_round_{round_num + 1}.csv")
        if poisoned_client_ids:
            log_poisoned_clients(poisoned_client_ids, round_num, poisoned_log_file)

        detection_mapping_summary = save_poison_detection_txt_mapping(
            detected_poison_client_ids=poisoned_client_ids,
            true_poison_client_ids=true_poison_client_ids,
            all_client_ids=all_client_ids_for_mapping,
            round_num=round_num,
            output_dir=mapping_log_dir,
            prefix="botpa" if config.enable_botpa_attack else ("pga" if config.enable_pga_attack else config.data_mode),
        )

        benign_cluster_protos, benign_safe_counts, _ = build_benign_cluster_prototypes(
            clusters=topology_result["clusters"],
            client_prototypes=client_prototypes,
            poisoned_client_ids=poisoned_client_ids,
        )
        if benign_cluster_protos:
            aggregated_server_proto = stage2_global_aggregation(benign_cluster_protos, safe_counts=benign_safe_counts)
        elif aggregated_server_proto is None:
            aggregated_server_proto = coordinate_wise_median(client_prototypes)

        prev_client_prototypes = {cid: proto.detach().cpu().clone() for cid, proto in client_prototypes.items()}
        prev_clusters = topology_result["clusters"]

        botpa_asr_mean = None
        if config.enable_botpa_attack and config.evaluate_botpa_asr_each_round and test_loader is not None:
            asr_df = evaluate_botpa_asr_all_clients(
                client_models=client_models,
                test_loader=test_loader,
                source_class=config.botpa_source_class,
                target_class=config.botpa_target_class,
                device=config.device,
                output_csv=os.path.join(eval_log_dir, "botpa_asr_by_client.csv"),
                round_num=round_num,
            )
            botpa_asr_mean = float(asr_df["ASR"].mean()) if not asr_df.empty else None

        save_full_checkpoint(
            model=client_models,
            optimizer=None,
            round_num=round_num,
            checkpoint_dir=config.checkpoint_dir,
            algorithm=config.checkpoint_algorithm,
            loss=float(np.mean(list(client_losses.values()))) if client_losses else None,
            metrics={
                "num_detected_poisoned": len(poisoned_client_ids),
                "num_detected_benign": len(benign_client_ids),
                "lambda_g": float(lambda_g),
                "botpa_asr_mean": botpa_asr_mean,
                "detection_TP": int(detection_mapping_summary["TP"]),
                "detection_FP": int(detection_mapping_summary["FP"]),
                "detection_FN": int(detection_mapping_summary["FN"]),
                "detection_TN": int(detection_mapping_summary["TN"]),
                "detection_precision": float(detection_mapping_summary["precision"]),
                "detection_recall": float(detection_mapping_summary["recall"]),
                "detection_f1": float(detection_mapping_summary["f1"]),
            },
            scheduler=None,
            extra_state={
                "phase": "round_done",
                "aggregated_server_proto": aggregated_server_proto,
                "prev_clusters": prev_clusters,
                "prev_client_prototypes": prev_client_prototypes,
                "threshold_state": threshold_state,
                "last_detection_result": detection_result,
                "client_attack_info": client_attack_info,
                "data_mode": config.data_mode,
                "true_poison_client_ids": true_poison_client_ids,
                "true_poison_mapping_info": true_poison_mapping_info,
                "detection_mapping_summary": detection_mapping_summary,
                "lambda_g_info": lambda_g_info,
                "botpa_metadata": botpa_metadata,
            },
        )
        last_detection_result = detection_result

    return {
        "client_models": client_models,
        "aggregated_server_proto": aggregated_server_proto,
        "prev_clusters": prev_clusters,
        "prev_client_prototypes": prev_client_prototypes,
        "threshold_state": threshold_state,
        "last_detection_result": last_detection_result,
        "true_poison_client_ids": true_poison_client_ids,
    }


def main(config: FrameworkConfig | None = None) -> dict[str, Any]:
    return run_training_pipeline(config or FrameworkConfig())


if __name__ == "__main__":
    main()

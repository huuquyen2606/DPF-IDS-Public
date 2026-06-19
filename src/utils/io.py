"""Checkpoint and lightweight file IO utilities."""

from __future__ import annotations

import glob
import os
import random
import warnings

import numpy as np
import torch

def save_full_checkpoint(
    model,
    optimizer,
    round_num,
    checkpoint_dir,
    algorithm,
    loss=None,
    metrics=None,
    scheduler=None,
    extra_state=None,
):
    """
    Lưu checkpoint đầy đủ để resume training.

    Hỗ trợ:
    - model là 1 nn.Module hoặc dict {client_id: nn.Module}
    - optimizer/scheduler là object đơn hoặc dict cùng key với model dict
    - extra_state để lưu thêm trạng thái của bài toán (clusters, prototype, ...)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir, f"{algorithm}_checkpoint_round{round_num + 1}.pt"
    )

    def _pack_model_state(m):
        if isinstance(m, dict):
            packed = {}
            for k, v in m.items():
                if hasattr(v, "state_dict"):
                    packed[k] = v.state_dict()
                else:
                    packed[k] = v
            return packed
        if hasattr(m, "state_dict"):
            return m.state_dict()
        return m

    def _pack_opt_state(o):
        if o is None:
            return None
        if isinstance(o, dict):
            packed = {}
            for k, v in o.items():
                packed[k] = v.state_dict() if hasattr(v, "state_dict") else v
            return packed
        return o.state_dict() if hasattr(o, "state_dict") else o

    checkpoint = {
        "model_state_dict": _pack_model_state(model),
        "optimizer_state_dict": _pack_opt_state(optimizer),
        "round": int(round_num) + 1,
        "loss": loss,
        "metrics": metrics,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "scheduler_state_dict": _pack_opt_state(scheduler),
        "extra_state": extra_state if extra_state is not None else {},
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_full_checkpoint(
    model,
    checkpoint_dir,
    algorithm,
    optimizer=None,
    scheduler=None,
    map_location=None,
    return_extra_state=False,
    strict_rng_restore=True,
):
    """
    Load checkpoint mới nhất và resume training.

    Hỗ trợ:
    - model là 1 nn.Module hoặc dict {client_id: nn.Module}
    - optimizer/scheduler là object đơn hoặc dict cùng key

    strict_rng_restore=True:
    - bắt buộc khôi phục RNG đầy đủ để đảm bảo resume không lệch
    - nếu RNG state lỗi kiểu dữ liệu sẽ raise ngay

    Returns:
        mặc định: round, loss, metrics
        nếu return_extra_state=True: round, loss, metrics, extra_state
    """
    if not os.path.exists(checkpoint_dir):
        print("No checkpoint directory found. Starting from round 0.")
        if return_extra_state:
            return 0, None, None, {}
        return 0, None, None

    checkpoint_files = glob.glob(
        os.path.join(checkpoint_dir, f"{algorithm}_checkpoint_round*.pt")
    )
    if not checkpoint_files:
        print(f"No checkpoint found for algorithm '{algorithm}'. Starting from round 0.")
        if return_extra_state:
            return 0, None, None, {}
        return 0, None, None

    round_nums = []
    for file in checkpoint_files:
        base = os.path.basename(file)
        try:
            round_num = int(base.split("round")[-1].replace(".pt", ""))
            round_nums.append((round_num, file))
        except ValueError:
            continue

    if not round_nums:
        print("No valid checkpoint found. Starting from round 0.")
        if return_extra_state:
            return 0, None, None, {}
        return 0, None, None

    latest_round, latest_file = max(round_nums, key=lambda x: x[0])
    checkpoint = torch.load(latest_file, map_location=map_location, weights_only=False)

    model_state = checkpoint.get("model_state_dict")

    # Load model
    if isinstance(model, dict):
        if not isinstance(model_state, dict):
            raise ValueError("Checkpoint model_state_dict is not dict but current model is dict.")
        for k, m in model.items():
            if k in model_state and hasattr(m, "load_state_dict"):
                m.load_state_dict(model_state[k])
    else:
        if hasattr(model, "load_state_dict"):
            model.load_state_dict(model_state)

    # Load optimizer
    opt_state = checkpoint.get("optimizer_state_dict")
    if optimizer is not None and opt_state is not None:
        if isinstance(optimizer, dict):
            if isinstance(opt_state, dict):
                for k, o in optimizer.items():
                    if k in opt_state and hasattr(o, "load_state_dict"):
                        o.load_state_dict(opt_state[k])
            print("Optimizer states loaded (dict mode).")
        else:
            if hasattr(optimizer, "load_state_dict"):
                optimizer.load_state_dict(opt_state)
            print("Optimizer state loaded.")
    elif optimizer is not None:
        print("No optimizer state found. Optimizer initialized fresh.")

    # Load scheduler
    sch_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and sch_state is not None:
        if isinstance(scheduler, dict):
            if isinstance(sch_state, dict):
                for k, s in scheduler.items():
                    if k in sch_state and hasattr(s, "load_state_dict"):
                        s.load_state_dict(sch_state[k])
            print("Scheduler states loaded (dict mode).")
        else:
            if hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(sch_state)
            print("Scheduler state loaded.")

    def _as_uint8_tensor(state_obj):
        if state_obj is None:
            return None
        if isinstance(state_obj, torch.Tensor):
            return state_obj.detach().to(device="cpu", dtype=torch.uint8)
        try:
            return torch.tensor(state_obj, dtype=torch.uint8, device="cpu")
        except Exception as exc:
            raise TypeError(f"Cannot convert torch RNG state to uint8 tensor: {type(state_obj)}") from exc

    # Load RNG states
    rng_state = checkpoint.get("rng_state")
    if rng_state:
        try:
            if rng_state.get("torch") is not None:
                torch.set_rng_state(_as_uint8_tensor(rng_state["torch"]))

            if rng_state.get("cuda") is not None and torch.cuda.is_available():
                cuda_states = rng_state["cuda"]
                if isinstance(cuda_states, (list, tuple)):
                    restored_cuda_states = [_as_uint8_tensor(s) for s in cuda_states]
                else:
                    restored_cuda_states = [_as_uint8_tensor(cuda_states)]
                torch.cuda.set_rng_state_all(restored_cuda_states)

            if rng_state.get("numpy") is not None:
                np.random.set_state(rng_state["numpy"])
            if rng_state.get("python") is not None:
                random.setstate(rng_state["python"])
        except Exception as exc:
            msg = (
                "Failed to restore RNG states from checkpoint. "
                "Resume may drift from the original run."
            )
            if strict_rng_restore:
                raise RuntimeError(msg) from exc
            warnings.warn(msg)

    loaded_round = checkpoint.get("round", latest_round)
    loaded_loss = checkpoint.get("loss")
    loaded_metrics = checkpoint.get("metrics")
    extra_state = checkpoint.get("extra_state", {})

    print(f"Loaded checkpoint from round {loaded_round}: {latest_file}")
    if return_extra_state:
        return loaded_round, loaded_loss, loaded_metrics, extra_state
    return loaded_round, loaded_loss, loaded_metrics


def load_lambda_g_from_txt(lambda_g_path, current_round, min_round=2):
    """
    Load lambda_g from lambda_g.txt if:
    - current_round >= min_round
    - file exists
    - file contains a valid float value

    Supports both formats:
        0.12345
    or:
        round=2, lambda_g=0.12345
    """
    if current_round < min_round:
        return None

    if not os.path.exists(lambda_g_path):
        return None

    with open(lambda_g_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return None

    # Case 1: file only contains a number
    try:
        return float(content)
    except ValueError:
        pass

    # Case 2: file contains something like "round=2, lambda_g=0.12345"
    if "lambda_g" in content:
        try:
            value_str = content.split("lambda_g")[-1]
            value_str = value_str.replace("=", "").replace(":", "").replace(",", "").strip()
            return float(value_str)
        except Exception:
            return None

    return None

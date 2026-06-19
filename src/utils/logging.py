"""Poison-client logging and mapping helpers."""

from __future__ import annotations

import os
import re
from typing import List

import pandas as pd

def load_poisoned_clients_for_round(log_dir: str, target_round: int) -> List[int]:
    """
    Loads poisoned client IDs from a log file for a specific round.
    The log files are named 'poisoned_clients_log_round_{round_num_1_indexed}.csv'
    where round_num_1_indexed is target_round.
    """
    if target_round <= 0:
        return [] # No previous rounds to load from

    log_file_path = os.path.join(log_dir, f"poisoned_clients_log_round_{target_round}.csv")

    if not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0:
        return []
    try:
        df = pd.read_csv(log_file_path)
        # Filter for the specific round (1-indexed) and get unique client IDs
        # The 'round' column in the CSV is 1-indexed.
        poisoned_in_round = df[df['round'] == target_round]['client_id'].unique().tolist()
        return poisoned_in_round
    except pd.errors.EmptyDataError:
        return []
    except Exception as e:
        print(f"Error loading poisoned clients from {log_file_path} for round {target_round}: {e}")
        return []


def log_poisoned_clients(poisoned_client_ids: List[int], round_num: int, log_file_path: str):
    """
    Logs poisoned client IDs to a CSV file, preventing duplicates for the same round.

    Args:
        poisoned_client_ids: A list of client IDs identified as poisoned.
        round_num: The current round number (0-indexed).
        log_file_path: The path to the CSV file where logs will be stored.
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Convert round_num to 1-based for logging consistency
    current_log_round = round_num + 1

    # Try to read existing log file
    existing_df = pd.DataFrame(columns=['round', 'client_id'])
    file_exists_and_not_empty = False
    if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0:
        try:
            existing_df = pd.read_csv(log_file_path)
            file_exists_and_not_empty = True
        except pd.errors.EmptyDataError:
            print(f"Log file '{log_file_path}' exists but is empty.")
        except Exception as e:
            print(f"Error reading existing poisoned clients log file '{log_file_path}': {e}")
            existing_df = pd.DataFrame(columns=['round', 'client_id']) # Fallback to empty DataFrame

    # Determine which clients are already logged for this round
    logged_in_current_round = set()
    if file_exists_and_not_empty:
        current_round_logs = existing_df[existing_df['round'] == current_log_round]
        logged_in_current_round = set(current_round_logs['client_id'].unique())

    # Filter out client IDs that are already logged for this round
    new_poisoned_client_ids = [
        cid for cid in poisoned_client_ids if cid not in logged_in_current_round
    ]

    if new_poisoned_client_ids:
        # Create DataFrame for new entries
        new_entries_df = pd.DataFrame({
            'round': [current_log_round] * len(new_poisoned_client_ids),
            'client_id': new_poisoned_client_ids
        })

        # Append new entries to the CSV file
        # If file_exists_and_not_empty is True, append without header.
        # Otherwise (new file or empty existing file), write with header.
        new_entries_df.to_csv(
            log_file_path,
            mode='a' if file_exists_and_not_empty else 'w',
            header=not file_exists_and_not_empty,
            index=False
        )
        print(f"Logged {len(new_poisoned_client_ids)} NEW poisoned clients for round {current_log_round} to {log_file_path}")
    else:
        print(f"No new poisoned clients to log for round {current_log_round}.")


def _read_ints_from_txt(path):
    """Read integer indexes from a .txt file with whitespace/comma/newline separators."""
    if path is None or str(path).strip() == "":
        return []

    if not os.path.exists(path):
        print(f"[WARN] True poison index file not found: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    values = []
    for token in re.split(r"[\s,;]+", text.strip()):
        if token == "":
            continue
        try:
            values.append(int(token))
        except ValueError:
            # Ignore non-integer tokens such as headers.
            continue

    return values


def _parse_client_file_number(filename):
    """
    Extract numeric id from names such as:
        client_0.pt, client_001.pt, client_500.pt
    """
    base = os.path.basename(str(filename))
    m = re.search(r"client[_\-]?(\d+)\.pt$", base)
    if m:
        return int(m.group(1))
    return None


def resolve_true_poison_client_ids(
    poison_index_path,
    client_data_info,
    num_clients=None,
    index_base="auto",
    verbose=True,
):
    """
    Map the original poison index .txt file to notebook client IDs.

    Notebook client IDs are created in build_client_loaders(...) as:
        enumerate(selected_files, start=1)
    so they are usually 1..N in sorted file order.

    This function supports common index conventions and returns:
        true_poison_client_ids, mapping_info
    """
    raw_indices = _read_ints_from_txt(poison_index_path)
    raw_set = set(raw_indices)

    if num_clients is None:
        num_clients = len(client_data_info)

    client_ids = set(int(info["client_id"]) for info in client_data_info)
    file_number_to_client_id = {}
    for info in client_data_info:
        file_num = _parse_client_file_number(info.get("client_file", ""))
        if file_num is not None:
            file_number_to_client_id[int(file_num)] = int(info["client_id"])

    def valid(mapped):
        mapped_set = set(mapped)
        return bool(mapped_set) and mapped_set.issubset(client_ids)

    candidates = {}

    # Candidate 1: values are already notebook client IDs.
    direct = [idx for idx in raw_set if idx in client_ids]
    candidates["client_id"] = sorted(direct)

    # Candidate 2: values are zero-based notebook positions.
    zero_based = [idx + 1 for idx in raw_set if (idx + 1) in client_ids]
    candidates["zero_based_client_id"] = sorted(zero_based)

    # Candidate 3: values are the numeric suffix in client_*.pt.
    filename_direct = [
        file_number_to_client_id[idx]
        for idx in raw_set
        if idx in file_number_to_client_id
    ]
    candidates["filename_number"] = sorted(set(filename_direct))

    # Candidate 4: values are zero-based numeric suffixes for client_*.pt.
    filename_zero_based = [
        file_number_to_client_id[idx + 1]
        for idx in raw_set
        if (idx + 1) in file_number_to_client_id
    ]
    candidates["zero_based_filename_number"] = sorted(set(filename_zero_based))

    chosen_mode = str(index_base or "auto")

    if chosen_mode != "auto":
        if chosen_mode not in candidates:
            raise ValueError(
                f"Unknown index_base={index_base}. "
                f"Choose from {list(candidates.keys()) + ['auto']}."
            )
        mapped_ids = candidates[chosen_mode]
    else:
        # Auto-inference preference:
        # 1) If filename-number mapping captures all raw values, prefer it because
        #    the original poison file often stores file indexes.
        # 2) Otherwise, if direct client_id captures all, use direct.
        # 3) Otherwise, use the candidate with the most mapped IDs.
        if len(candidates["filename_number"]) == len(raw_set) and len(raw_set) > 0:
            chosen_mode = "filename_number"
        elif len(candidates["zero_based_filename_number"]) == len(raw_set) and len(raw_set) > 0:
            chosen_mode = "zero_based_filename_number"
        elif len(candidates["client_id"]) == len(raw_set) and len(raw_set) > 0:
            chosen_mode = "client_id"
        elif len(candidates["zero_based_client_id"]) == len(raw_set) and len(raw_set) > 0:
            chosen_mode = "zero_based_client_id"
        else:
            chosen_mode = max(candidates.keys(), key=lambda k: len(candidates[k]))

        mapped_ids = candidates[chosen_mode]

    mapped_set = set(mapped_ids)
    unmapped_raw_indices = sorted([
        idx for idx in raw_set
        if (
            idx not in client_ids
            and (idx + 1) not in client_ids
            and idx not in file_number_to_client_id
            and (idx + 1) not in file_number_to_client_id
        )
    ])

    mapping_info = {
        "poison_index_path": poison_index_path,
        "raw_count": len(raw_set),
        "mapped_count": len(mapped_set),
        "chosen_index_base": chosen_mode,
        "raw_indices_preview": sorted(list(raw_set))[:20],
        "mapped_client_ids_preview": sorted(list(mapped_set))[:20],
        "unmapped_raw_indices": unmapped_raw_indices,
        "candidate_counts": {k: len(set(v)) for k, v in candidates.items()},
    }

    if verbose:
        print("\n=== True poison-index mapping ===")
        print(f"path: {poison_index_path}")
        print(f"chosen mode: {chosen_mode}")
        print(f"raw indices: {len(raw_set)} | mapped notebook client IDs: {len(mapped_set)}")
        print("candidate counts:", mapping_info["candidate_counts"])
        if unmapped_raw_indices:
            print(f"[WARN] unmapped raw indices: {unmapped_raw_indices[:20]}")

    return sorted(mapped_set), mapping_info


def write_client_id_txt(client_ids, path):
    """Write one client id per line."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for cid in sorted(set(int(x) for x in client_ids)):
            f.write(f"{cid}\n")


def save_poison_detection_txt_mapping(
    detected_poison_client_ids,
    true_poison_client_ids,
    all_client_ids,
    round_num,
    output_dir,
    prefix="pga",
):
    """
    Save .txt files and a CSV summary for detection correctness.

    Files per round:
        round_XXX_detected_poison.txt
        round_XXX_true_poison.txt
        round_XXX_true_positive.txt
        round_XXX_false_positive.txt
        round_XXX_false_negative.txt
        round_XXX_true_negative.txt

    Summary:
        poison_detection_summary.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    detected_set = set(int(x) for x in detected_poison_client_ids)
    true_set = set(int(x) for x in true_poison_client_ids)
    all_set = set(int(x) for x in all_client_ids)

    tp = sorted(detected_set & true_set)
    fp = sorted(detected_set - true_set)
    fn = sorted(true_set - detected_set)
    tn = sorted(all_set - detected_set - true_set)

    r = int(round_num) + 1
    round_tag = f"round_{r:03d}"

    write_client_id_txt(sorted(detected_set), os.path.join(output_dir, f"{round_tag}_detected_poison.txt"))
    write_client_id_txt(sorted(true_set), os.path.join(output_dir, f"{round_tag}_true_poison.txt"))
    write_client_id_txt(tp, os.path.join(output_dir, f"{round_tag}_true_positive.txt"))
    write_client_id_txt(fp, os.path.join(output_dir, f"{round_tag}_false_positive.txt"))
    write_client_id_txt(fn, os.path.join(output_dir, f"{round_tag}_false_negative.txt"))
    write_client_id_txt(tn, os.path.join(output_dir, f"{round_tag}_true_negative.txt"))

    precision = len(tp) / max(len(tp) + len(fp), 1)
    recall = len(tp) / max(len(tp) + len(fn), 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    false_positive_rate = len(fp) / max(len(fp) + len(tn), 1)

    summary_row = {
        "round": r,
        "prefix": prefix,
        "num_all_clients": len(all_set),
        "num_detected_poison": len(detected_set),
        "num_true_poison": len(true_set),
        "TP": len(tp),
        "FP": len(fp),
        "FN": len(fn),
        "TN": len(tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
    }

    summary_path = os.path.join(output_dir, f"{prefix}_poison_detection_summary.csv")
    summary_df = pd.DataFrame([summary_row])
    if os.path.exists(summary_path) and os.path.getsize(summary_path) > 0:
        summary_df.to_csv(summary_path, mode="a", header=False, index=False)
    else:
        summary_df.to_csv(summary_path, mode="w", header=True, index=False)

    readable_path = os.path.join(output_dir, f"{round_tag}_mapping_report.txt")
    with open(readable_path, "w", encoding="utf-8") as f:
        f.write(f"Round: {r}\n")
        f.write(f"Detected poison: {len(detected_set)}\n")
        f.write(f"True poison: {len(true_set)}\n")
        f.write(f"TP: {len(tp)}\n")
        f.write(f"FP: {len(fp)}\n")
        f.write(f"FN: {len(fn)}\n")
        f.write(f"TN: {len(tn)}\n")
        f.write(f"Precision: {precision:.6f}\n")
        f.write(f"Recall: {recall:.6f}\n")
        f.write(f"F1: {f1:.6f}\n")
        f.write(f"False Positive Rate: {false_positive_rate:.6f}\n\n")
        f.write(f"TP client IDs: {tp}\n")
        f.write(f"FP client IDs: {fp}\n")
        f.write(f"FN client IDs: {fn}\n")

    print(
        f"[Round {r}] Detection mapping | "
        f"TP={len(tp)} FP={len(fp)} FN={len(fn)} TN={len(tn)} | "
        f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}"
    )

    return summary_row

"""
Common utilities for fair fixed-schedule FL comparison.

Place this file in the same folder as config.py, models.py, and feature_extractor.py.
This file intentionally does NOT save per-client local model weights.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import cfg
from models import FL_CNN

StateDict = Dict[str, torch.Tensor]


@dataclass(frozen=True)
class ScheduleRow:
    round_num: int
    client_id: str
    is_potential_attacker: bool
    is_actual_attacker: bool
    attack_type: str
    attacker_percentage: float
    alpha: float


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic mode improves reproducibility. If an op is unsupported on your GPU,
    # change warn_only=True or remove this block.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def stable_seed(*parts: object, modulo: int = 2**31 - 1) -> int:
    text = "::".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_csv_header(path: os.PathLike[str] | str, header: Sequence[str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        csv.writer(f).writerow(list(header))


def append_csv_row(path: os.PathLike[str] | str, row: Sequence[object]) -> None:
    with Path(path).open("a", newline="") as f:
        csv.writer(f).writerow(list(row))


def append_csv_rows(path: os.PathLike[str] | str, rows: Iterable[Sequence[object]]) -> None:
    with Path(path).open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def state_to_cpu(state: Mapping[str, torch.Tensor]) -> StateDict:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def state_to_device(state: Mapping[str, torch.Tensor], device: torch.device) -> StateDict:
    device_str = str(device)
    return {k: v.detach().to(device_str).clone() for k, v in state.items()}


def zeros_like_state(reference: Mapping[str, torch.Tensor]) -> StateDict:
    return {k: torch.zeros_like(v, dtype=torch.float32, device=v.device) for k, v in reference.items()}


def load_state_to_model(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    model.load_state_dict({k: v.to(cfg.DEVICE) for k, v in state.items()})


def make_transform():
    return transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def make_target_transform_numeric(class_names: Sequence[str]) -> Callable[[int], int]:
    # ImageFolder uses local class indices. This maps them back to the true numeric label
    # stored as the folder name, e.g., folder "17" -> label 17.
    def _target_transform(local_idx: int) -> int:
        return int(class_names[int(local_idx)])
    return _target_transform


def list_client_folders() -> List[str]:
    root = Path(cfg.DATA_ROOT_PATH)
    if not root.exists():
        raise FileNotFoundError(f"DATA_ROOT_PATH does not exist: {root}")
    clients = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not clients:
        raise RuntimeError(f"No client folders found under DATA_ROOT_PATH: {root}")
    return clients


def make_test_loader(batch_size: int = 128) -> DataLoader:
    transform = make_transform()
    test_dataset = datasets.ImageFolder(root=cfg.TEST_DATA_PATH, transform=transform)
    test_dataset.target_transform = make_target_transform_numeric(test_dataset.classes)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def make_client_loader(client_id: str, seed: int, batch_size: Optional[int] = None) -> Tuple[Optional[DataLoader], int, int]:
    client_train_path = Path(cfg.DATA_ROOT_PATH) / client_id / "train"
    if not client_train_path.exists() or not any(client_train_path.iterdir()):
        return None, 0, 0

    dataset = datasets.ImageFolder(root=str(client_train_path), transform=make_transform())
    dataset.target_transform = make_target_transform_numeric(dataset.classes)
    n_k = len(dataset)
    if n_k == 0:
        return None, 0, len(dataset.classes)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size or cfg.CNN_BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    return loader, n_k, len(dataset.classes)


@torch.no_grad()
def evaluate_global_model(model: torch.nn.Module, test_loader: DataLoader) -> Tuple[float, float, float, float]:
    model.eval()
    correct, total = 0, 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for images, labels in test_loader:
        images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    accuracy = (correct / total) * 100 if total > 0 else 0.0
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    return float(accuracy), float(precision), float(recall), float(f1)


def train_one_client_from_global(
    global_state_cpu: Mapping[str, torch.Tensor],
    client_id: str,
    round_num: int,
    seed: int,
) -> Tuple[Optional[StateDict], int, float, float]:
    """Train one benign local model and return its final state, n_k, final acc, final loss."""
    loader_seed = stable_seed(seed, "loader", round_num, client_id)
    local_loader, n_k, _num_classes = make_client_loader(client_id, loader_seed)
    if local_loader is None or n_k <= 0:
        return None, 0, 0.0, 0.0

    local_model = FL_CNN().to(cfg.DEVICE)
    load_state_to_model(local_model, global_state_cpu)
    optimizer = optim.SGD(local_model.parameters(), lr=cfg.CNN_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    final_epoch_loss = 0.0
    final_epoch_acc = 0.0
    local_model.train()
    for _epoch in range(cfg.LOCAL_CNN_EPOCHS):
        epoch_loss, correct, total = 0.0, 0, 0
        for images, labels in local_loader:
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            outputs = local_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        final_epoch_loss = epoch_loss / max(1, len(local_loader))
        final_epoch_acc = (correct / total) * 100 if total > 0 else 0.0

    result_state = state_to_cpu(local_model.state_dict())
    del local_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result_state, n_k, float(final_epoch_acc), float(final_epoch_loss)


def randomize_state_dict_like(reference_state: Mapping[str, torch.Tensor], seed: int) -> StateDict:
    """Create standard-normal random parameters with the same tensor shapes as reference_state."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    randomized: StateDict = {}
    for k, v in reference_state.items():
        noise = torch.randn(v.shape, dtype=v.dtype, generator=generator, device="cpu")
        randomized[k] = noise
    return randomized


def flatten_update(local_state: Mapping[str, torch.Tensor], prev_state: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([(local_state[k].float() - prev_state[k].float()).reshape(-1) for k in prev_state.keys()])


def add_update_to_state(prev_state: Mapping[str, torch.Tensor], update_state: Mapping[str, torch.Tensor]) -> StateDict:
    return {k: (prev_state[k].float() + update_state[k].float()).cpu() for k in prev_state.keys()}


def weighted_average_states(
    local_states: Mapping[str, Mapping[str, torch.Tensor]],
    weights: Mapping[str, float],
    prev_state: Optional[Mapping[str, torch.Tensor]] = None,
) -> StateDict:
    """Weighted average of full submitted model states. Weights should sum to 1."""
    if not local_states:
        if prev_state is None:
            raise ValueError("Cannot aggregate empty local_states without prev_state.")
        return state_to_cpu(prev_state)

    first_state = next(iter(local_states.values()))
    avg = {k: torch.zeros_like(v, dtype=torch.float32, device="cpu") for k, v in first_state.items()}
    for cid, state in local_states.items():
        w = float(weights.get(cid, 0.0))
        if w <= 0:
            continue
        for k in avg.keys():
            avg[k] += state[k].float().cpu() * w
    return avg


def fedavg_aggregate(
    local_states: Mapping[str, Mapping[str, torch.Tensor]],
    sample_sizes: Mapping[str, int],
    prev_state: Mapping[str, torch.Tensor],
) -> StateDict:
    total_samples = sum(max(0, int(sample_sizes.get(cid, 0))) for cid in local_states.keys())
    if total_samples <= 0:
        return state_to_cpu(prev_state)
    weights = {cid: max(0, int(sample_sizes.get(cid, 0))) / total_samples for cid in local_states.keys()}
    return weighted_average_states(local_states, weights, prev_state)


def weighted_update_aggregate(
    local_states: Mapping[str, Mapping[str, torch.Tensor]],
    prev_state: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
) -> StateDict:
    """prev_state + weighted average of client updates. Weights should sum to 1."""
    if not local_states:
        return state_to_cpu(prev_state)
    update_sum = {k: torch.zeros_like(v, dtype=torch.float32, device="cpu") for k, v in prev_state.items()}
    for cid, local_state in local_states.items():
        w = float(weights.get(cid, 0.0))
        if w <= 0:
            continue
        for k in update_sum.keys():
            update_sum[k] += (local_state[k].float().cpu() - prev_state[k].float().cpu()) * w
    return add_update_to_state(prev_state, update_sum)


def krum_select_index(update_vectors: torch.Tensor, f: int) -> Tuple[int, List[float]]:
    """Return selected index and Krum scores for flattened update vectors [n, d]."""
    n = update_vectors.size(0)
    if 2 * f + 2 >= n:
        raise ValueError(f"Krum requires 2f + 2 < n. Got n={n}, f={f}.")
    neighbor_count = n - f - 2
    if neighbor_count <= 0:
        raise ValueError(f"Invalid Krum neighbor count: n-f-2={neighbor_count}.")

    distances = torch.cdist(update_vectors.float(), update_vectors.float(), p=2) ** 2
    scores: List[float] = []
    for i in range(n):
        row = distances[i].clone()
        row[i] = float("inf")
        closest, _ = torch.topk(row, k=neighbor_count, largest=False)
        scores.append(float(closest.sum().item()))
    selected_idx = int(np.argmin(scores))
    return selected_idx, scores


def krum_aggregate(
    local_states: Mapping[str, Mapping[str, torch.Tensor]],
    prev_state: Mapping[str, torch.Tensor],
    f: int,
) -> Tuple[StateDict, str, List[Tuple[str, float]]]:
    cids = list(local_states.keys())
    if not cids:
        return state_to_cpu(prev_state), "", []
    update_vectors = torch.stack([flatten_update(local_states[cid], prev_state) for cid in cids], dim=0)
    selected_idx, scores = krum_select_index(update_vectors, f)
    selected_cid = cids[selected_idx]
    selected_update = {k: local_states[selected_cid][k].float().cpu() - prev_state[k].float().cpu() for k in prev_state.keys()}
    score_rows = list(zip(cids, scores))
    return add_update_to_state(prev_state, selected_update), selected_cid, score_rows


def multikrum_aggregate(
    local_states: Mapping[str, Mapping[str, torch.Tensor]],
    prev_state: Mapping[str, torch.Tensor],
    f: int,
    m: Optional[int] = None,
) -> Tuple[StateDict, List[str], List[Tuple[str, float]]]:
    cids = list(local_states.keys())
    if not cids:
        return state_to_cpu(prev_state), [], []
    n = len(cids)
    m_select = int(m if m is not None else n - f)
    m_select = max(1, min(m_select, n))

    update_vectors = torch.stack([flatten_update(local_states[cid], prev_state) for cid in cids], dim=0)
    _selected_idx, scores = krum_select_index(update_vectors, f)
    order = np.argsort(scores)[:m_select].tolist()
    selected_cids = [cids[i] for i in order]
    weights = {cid: (1.0 / len(selected_cids) if cid in selected_cids else 0.0) for cid in cids}
    score_rows = list(zip(cids, scores))
    return weighted_update_aggregate(local_states, prev_state, weights), selected_cids, score_rows


def coordinatewise_median_aggregate(
    local_states: Mapping[str, Mapping[str, torch.Tensor]],
    prev_state: Mapping[str, torch.Tensor],
) -> StateDict:
    cids = list(local_states.keys())
    if not cids:
        return state_to_cpu(prev_state)
    median_update: StateDict = {}
    n = len(cids)
    for k in prev_state.keys():
        stacked = torch.stack([local_states[cid][k].float().cpu() - prev_state[k].float().cpu() for cid in cids], dim=0)
        sorted_vals, _ = torch.sort(stacked, dim=0)
        if n % 2 == 1:
            median_update[k] = sorted_vals[n // 2]
        else:
            # Usual one-dimensional median for an even number of values: average of the two middle values.
            median_update[k] = 0.5 * (sorted_vals[(n // 2) - 1] + sorted_vals[n // 2])
    return add_update_to_state(prev_state, median_update)


def coordinatewise_trimmed_mean_aggregate(
    local_states: Mapping[str, Mapping[str, torch.Tensor]],
    prev_state: Mapping[str, torch.Tensor],
    f_trim: int,
) -> StateDict:
    cids = list(local_states.keys())
    if not cids:
        return state_to_cpu(prev_state)
    n = len(cids)
    if f_trim < 0:
        raise ValueError("f_trim must be non-negative.")
    if 2 * f_trim >= n:
        raise ValueError(f"Trimmed mean requires 2*f_trim < n. Got n={n}, f_trim={f_trim}.")

    trimmed_update: StateDict = {}
    for k in prev_state.keys():
        stacked = torch.stack([local_states[cid][k].float().cpu() - prev_state[k].float().cpu() for cid in cids], dim=0)
        if f_trim == 0:
            trimmed_update[k] = stacked.mean(dim=0)
        else:
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed_update[k] = sorted_vals[f_trim:n - f_trim].mean(dim=0)
    return add_update_to_state(prev_state, trimmed_update)


def detection_counts(y_true_actual: Mapping[str, bool], y_pred_attacker: Mapping[str, bool]) -> Tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for cid, actual in y_true_actual.items():
        pred = bool(y_pred_attacker.get(cid, False))
        if actual and pred:
            tp += 1
        elif (not actual) and pred:
            fp += 1
        elif (not actual) and (not pred):
            tn += 1
        elif actual and (not pred):
            fn += 1
    return tp, fp, tn, fn


def detection_metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    dr = recall
    fdr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # false detection rate / FPR
    return {
        "DR": dr,
        "FDR": fdr,
        "Detection Precision": precision,
        "Detection Recall": recall,
        "Detection F1": f1,
    }


def schedule_default_path(attack_type: str, attacker_percentage: float, alpha: float, seed: int, live_rounds: int) -> Path:
    safe_pct = str(attacker_percentage).replace(".", "p")
    safe_alpha = str(alpha).replace(".", "p")
    return Path(cfg.LOG_DIR_BASE) / "fixed_schedules" / f"{attack_type}_atk{safe_pct}_alpha{safe_alpha}_seed{seed}_rounds{live_rounds}.csv"


def generate_fixed_schedule(
    attack_type: str,
    attacker_percentage: float,
    alpha: float,
    live_rounds: int,
    seed: int,
    out_csv: os.PathLike[str] | str,
) -> Path:
    """Generate one fixed schedule for one condition and save it."""
    if attack_type not in {"signflip", "freeride"}:
        raise ValueError("attack_type must be 'signflip' or 'freeride'.")
    all_clients = list_client_folders()
    rng = random.Random(seed)

    total_attackers = int(len(all_clients) * (float(attacker_percentage) / 100.0))
    if total_attackers < 0 or total_attackers > len(all_clients):
        raise ValueError(f"Invalid attacker_percentage={attacker_percentage}")

    attacker_pool = set(rng.sample(all_clients, total_attackers)) if total_attackers > 0 else set()
    benign_pool = sorted(set(all_clients) - attacker_pool)
    attacker_pool_sorted = sorted(attacker_pool)

    potential_per_round = int(cfg.CLIENTS_PER_ROUND * (float(attacker_percentage) / 100.0))
    benign_per_round = cfg.CLIENTS_PER_ROUND - potential_per_round
    if potential_per_round > len(attacker_pool_sorted):
        raise ValueError("Not enough attackers in attacker pool for per-round sampling.")
    if benign_per_round > len(benign_pool):
        raise ValueError("Not enough benign clients in benign pool for per-round sampling.")

    out_csv = Path(out_csv)
    ensure_dir(out_csv.parent)
    header = ["Round", "ClientID", "Is_Potential_Attacker", "Is_Actual_Attacker", "Attack_Type", "Attacker_Percentage", "Alpha"]
    rows: List[List[object]] = []

    for round_num in range(1, live_rounds + 1):
        selected_benign = rng.sample(benign_pool, benign_per_round)
        selected_attackers = rng.sample(attacker_pool_sorted, potential_per_round) if potential_per_round > 0 else []
        selected_clients = selected_benign + selected_attackers
        rng.shuffle(selected_clients)
        for cid in selected_clients:
            is_potential = cid in attacker_pool
            is_actual = bool(is_potential and (rng.random() < float(alpha)))
            rows.append([round_num, cid, int(is_potential), int(is_actual), attack_type, attacker_percentage, alpha])

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    meta = {
        "attack_type": attack_type,
        "attacker_percentage": attacker_percentage,
        "alpha": alpha,
        "seed": seed,
        "live_rounds": live_rounds,
        "clients_per_round": cfg.CLIENTS_PER_ROUND,
        "potential_attackers_per_round": potential_per_round,
        "benign_per_round": benign_per_round,
        "total_clients": len(all_clients),
        "total_potential_attackers_in_pool": len(attacker_pool_sorted),
        "attacker_pool": attacker_pool_sorted,
    }
    with out_csv.with_suffix(".metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)
    return out_csv


def load_schedule(schedule_csv: os.PathLike[str] | str) -> Dict[int, List[ScheduleRow]]:
    schedule_by_round: Dict[int, List[ScheduleRow]] = {}
    with Path(schedule_csv).open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            srow = ScheduleRow(
                round_num=int(row["Round"]),
                client_id=str(row["ClientID"]),
                is_potential_attacker=bool(int(row["Is_Potential_Attacker"])),
                is_actual_attacker=bool(int(row["Is_Actual_Attacker"])),
                attack_type=str(row["Attack_Type"]),
                attacker_percentage=float(row["Attacker_Percentage"]),
                alpha=float(row["Alpha"]),
            )
            schedule_by_round.setdefault(srow.round_num, []).append(srow)
    return schedule_by_round


def find_warmup_global_path() -> Path:
    preferred = Path(cfg.LOG_DIR_BASE) / "global_model_warmup.pth"
    old_name = Path(cfg.LOG_DIR_BASE) / "global_model_stage1.pth"
    if preferred.exists():
        return preferred
    if old_name.exists():
        return old_name
    raise FileNotFoundError(f"Missing warm-up global model. Expected {preferred} or {old_name}.")

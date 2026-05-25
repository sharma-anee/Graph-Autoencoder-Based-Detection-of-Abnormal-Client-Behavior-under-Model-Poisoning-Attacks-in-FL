"""
02_run_fair_comparison_fixed_schedule.py

Purpose
-------
Run fair fixed-schedule comparison from the common warm-up global model.

The live comparison rounds are numbered 1..N, even though they start from the
warm-up global model produced by 01_warmup_train_detector_no_local_saves.py.

Methods implemented
-------------------
1. fedavg_no_attack
2. fedavg_attack_nodefense
3. krum
4. multikrum
5. coord_median
6. trimmed_mean
7. proposed_thresholding
8. proposed_credit_scoring

Fairness rule
-------------
For each condition, this script generates or reads ONE fixed schedule containing:
- attacker pool
- selected clients in every live round
- actual attacker decisions in every live round

Every method reads that same schedule. The methods start from the same warm-up
global model, but each method evolves its own global model after aggregation.

No per-client local model weights are saved.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from config import cfg
from feature_extractor import convert_update_to_graph
from fl_fair_common import (
    ScheduleRow,
    append_csv_row,
    append_csv_rows,
    coordinatewise_median_aggregate,
    coordinatewise_trimmed_mean_aggregate,
    detection_counts,
    detection_metrics_from_counts,
    ensure_dir,
    evaluate_global_model,
    fedavg_aggregate,
    find_warmup_global_path,
    generate_fixed_schedule,
    krum_aggregate,
    load_schedule,
    load_state_to_model,
    make_test_loader,
    multikrum_aggregate,
    randomize_state_dict_like,
    schedule_default_path,
    set_all_seeds,
    stable_seed,
    state_to_cpu,
    train_one_client_from_global,
    weighted_average_states,
    weighted_update_aggregate,
    write_csv_header,
)
from models import FL_CNN, GraphAutoencoder

StateDict = Dict[str, torch.Tensor]

ALL_METHODS = [
    "fedavg_no_attack",
    "fedavg_attack_nodefense",
    "krum",
    "multikrum",
    "coord_median",
    "trimmed_mean",
    "proposed_thresholding",
    "proposed_credit_scoring",
]


def compute_alignment_feature(conv2_update: torch.Tensor, benign_ref_dirs: torch.Tensor) -> torch.Tensor:
    u = conv2_update.view(32, -1)
    r = benign_ref_dirs.view(32, -1)
    u = u / (u.norm(dim=1, keepdim=True).clamp_min(1e-12))
    r = r / (r.norm(dim=1, keepdim=True).clamp_min(1e-12))
    return (u * r).sum(dim=1)


def get_graph_tensors(graph: Data, context: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return graph.x and graph.edge_index as non-optional tensors for PyLance and runtime safety."""
    x = graph.x
    edge_index = graph.edge_index
    if x is None:
        raise RuntimeError(f"{context}: graph.x is None.")
    if edge_index is None:
        raise RuntimeError(f"{context}: graph.edge_index is None.")
    return cast(torch.Tensor, x), cast(torch.Tensor, edge_index)


def load_detector_and_reference() -> Tuple[GraphAutoencoder, torch.Tensor]:
    detector_path = Path(cfg.LOG_DIR_BASE) / "detector_model.pth"
    ref_path = Path(cfg.LOG_DIR_BASE) / "benign_ref_dirs.pth"
    if not detector_path.exists():
        raise FileNotFoundError(f"Missing detector model: {detector_path}")
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing benign reference directions: {ref_path}")

    detector = GraphAutoencoder(
        in_channels=7,
        hidden_channels=cfg.GAE_HIDDEN_CHANNELS,
        out_channels=cfg.GAE_EMBEDDING_SIZE,
    ).to(cfg.DEVICE)
    detector.load_state_dict(torch.load(detector_path, map_location=cfg.DEVICE))
    detector.eval()
    benign_ref_dirs = torch.load(ref_path, map_location=cfg.DEVICE)
    return detector, benign_ref_dirs


def produce_local_submissions_for_round(
    global_state_cpu: Mapping[str, torch.Tensor],
    round_rows: Sequence[ScheduleRow],
    round_num: int,
    seed: int,
    attack_type: str,
    no_attack: bool,
) -> Tuple[Dict[str, StateDict], Dict[str, int], Dict[str, bool], Dict[str, bool], float, float]:
    """
    Produce local submitted model states for one round.

    If no_attack=True, all scheduled clients train benignly even if the schedule marks them as attackers.
    If no_attack=False, actual attackers follow attack_type from the fixed schedule.
    """
    local_states: Dict[str, StateDict] = {}
    sample_sizes: Dict[str, int] = {}
    actual_attacker_flags: Dict[str, bool] = {}
    potential_attacker_flags: Dict[str, bool] = {}
    train_accs: List[float] = []
    train_losses: List[float] = []

    for row in tqdm(round_rows, desc=f"Round {round_num} local submissions", leave=False):
        cid = row.client_id
        scheduled_actual = bool(row.is_actual_attacker)
        is_actual = False if no_attack else scheduled_actual
        actual_attacker_flags[cid] = is_actual
        potential_attacker_flags[cid] = bool(row.is_potential_attacker)

        if attack_type == "freeride" and is_actual:
            # Free-rider does not train. It still has sample size for weighting, matching the older setup.
            # We read the client loader only to obtain n_k; no local training is done.
            loader_seed = stable_seed(seed, "freerider_loader_for_nk", round_num, cid)
            from fl_fair_common import make_client_loader  # local import avoids unused import in other paths
            _loader, n_k, _num_classes = make_client_loader(cid, loader_seed)
            if n_k <= 0:
                continue
            rand_seed = stable_seed(seed, "freeride_random_state", round_num, cid)
            local_state = randomize_state_dict_like(global_state_cpu, rand_seed)
            local_states[cid] = local_state
            sample_sizes[cid] = n_k
            # Free-riders are excluded from train accuracy/loss, because they did not train.
            continue

        # Benign client or sign-flipping attacker: train locally first.
        local_state, n_k, train_acc, train_loss = train_one_client_from_global(
            global_state_cpu,
            client_id=cid,
            round_num=round_num,
            seed=seed,
        )
        if local_state is None or n_k <= 0:
            continue

        if attack_type == "signflip" and is_actual:
            # Sign flipping after local training: flip the submitted model parameters.
            local_state = {k: (-1.0 * v.float()).cpu() for k, v in local_state.items()}

        local_states[cid] = local_state
        sample_sizes[cid] = n_k
        train_accs.append(train_acc)
        train_losses.append(train_loss)

    avg_train_acc = float(np.mean(train_accs)) if train_accs else 0.0
    avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
    return local_states, sample_sizes, actual_attacker_flags, potential_attacker_flags, avg_train_acc, avg_train_loss


def proposed_detection_and_weights(
    local_states: Mapping[str, Mapping[str, torch.Tensor]],
    sample_sizes: Mapping[str, int],
    actual_attacker_flags: Mapping[str, bool],
    prev_global_cpu: Mapping[str, torch.Tensor],
    detector: GraphAutoencoder,
    benign_ref_dirs: torch.Tensor,
    defense_method: str,
    threshold_renormalize: bool,
) -> Tuple[StateDict, Dict[str, float], Dict[str, float], Dict[str, bool], Dict[str, float], Tuple[int, int, int, int]]:
    """
    Run proposed GAE detection and return new global state + logs.

    Detection decision is score > mean(score) for BOTH proposed variants.
    - thresholding: predicted attackers are dropped.
    - credit_scoring: all clients receive positive score-based weights, but the same detection
      decision is logged for DR/FDR/precision/recall/F1.
    """
    recon_errors: Dict[str, float] = {}
    anomaly_scores: Dict[str, float] = {}

    bce_loss_fn = torch.nn.BCELoss()
    mse_loss_fn = torch.nn.MSELoss()

    for cid, local_state in local_states.items():
        conv2_update = (local_state["conv2.weight"].float() - prev_global_cpu["conv2.weight"].float()).to(cfg.DEVICE)
        alignment_feature = compute_alignment_feature(conv2_update, benign_ref_dirs.to(cfg.DEVICE))
        graph = convert_update_to_graph(conv2_update, alignment_feature).to(str(cfg.DEVICE))
        graph_x, graph_edge_index = get_graph_tensors(graph, f"Detection graph for client {cid}")
        if graph_x.size(1) != 7:
            raise RuntimeError(f"Expected 7 node features, got {graph_x.size(1)}")

        with torch.no_grad():
            z = detector.encode(graph_x, graph_edge_index)
            recon_adj, recon_feat = detector.decode(z)
            num_nodes = int(graph_x.size(0))
            true_adj = torch.zeros((num_nodes, num_nodes), device=cfg.DEVICE)
            true_adj[graph_edge_index[0], graph_edge_index[1]] = 1.0
            loss_a = bce_loss_fn(recon_adj, true_adj)
            loss_x = mse_loss_fn(recon_feat, graph_x)
            err = cfg.LAMBDA_ADJACENCY * loss_a + cfg.LAMBDA_FEATURES * loss_x
            recon_errors[cid] = float(err.item())

    if recon_errors:
        min_err = min(recon_errors.values())
        for cid, err in recon_errors.items():
            anomaly_scores[cid] = float((1.0 + err) / (1.0 + min_err))

    threshold = float(np.mean(list(anomaly_scores.values()))) if anomaly_scores else 0.0
    predicted_attacker = {cid: (score > threshold) for cid, score in anomaly_scores.items()}

    aggregation_weights: Dict[str, float] = {}
    if defense_method == "thresholding":
        approved = [cid for cid in local_states.keys() if not predicted_attacker.get(cid, False)]
        if threshold_renormalize:
            approved_samples = sum(max(0, int(sample_sizes.get(cid, 0))) for cid in approved)
            for cid in local_states.keys():
                aggregation_weights[cid] = (sample_sizes.get(cid, 0) / approved_samples) if (cid in approved and approved_samples > 0) else 0.0
        else:
            # Compatibility with the old script: denominator is all submitted clients.
            total_samples = sum(max(0, int(sample_sizes.get(cid, 0))) for cid in local_states.keys())
            for cid in local_states.keys():
                aggregation_weights[cid] = (sample_sizes.get(cid, 0) / total_samples) if (cid in approved and total_samples > 0) else 0.0

        if sum(aggregation_weights.values()) > 0:
            new_global = weighted_average_states(local_states, aggregation_weights, prev_global_cpu)
        else:
            new_global = state_to_cpu(prev_global_cpu)

    elif defense_method == "credit_scoring":
        denominator = sum(
            max(0, int(sample_sizes.get(cid, 0))) * (float(anomaly_scores.get(cid, 1.0)) ** (-cfg.CREDIT_SCORE_L))
            for cid in local_states.keys()
        )
        if (not math.isfinite(denominator)) or denominator <= 0:
            total_samples = sum(max(0, int(sample_sizes.get(cid, 0))) for cid in local_states.keys())
            for cid in local_states.keys():
                aggregation_weights[cid] = (sample_sizes.get(cid, 0) / total_samples) if total_samples > 0 else 0.0
        else:
            for cid in local_states.keys():
                n_k = max(0, int(sample_sizes.get(cid, 0)))
                score = float(anomaly_scores.get(cid, 1.0))
                aggregation_weights[cid] = (n_k * (score ** (-cfg.CREDIT_SCORE_L))) / denominator

        # Credit scoring keeps all clients but reduces suspicious influence.
        new_global = weighted_average_states(local_states, aggregation_weights, prev_global_cpu)
    else:
        raise ValueError(f"Unknown proposed defense method: {defense_method}")

    counts = detection_counts(actual_attacker_flags, predicted_attacker)
    return new_global, recon_errors, anomaly_scores, predicted_attacker, aggregation_weights, counts


def initialize_method_logs(method_dir: Path, is_proposed: bool) -> Dict[str, Path]:
    metrics_csv = method_dir / "round_metrics.csv"
    write_csv_header(
        metrics_csv,
        [
            "Round",
            "Test Accuracy",
            "Test Precision",
            "Test Recall",
            "Test F1 Score",
            "Avg Train Accuracy",
            "Avg Train Loss",
            "Num Submitted Clients",
            "Num Actual Attackers",
            "Total Samples",
        ],
    )
    paths = {"round_metrics": metrics_csv}

    if is_proposed:
        detection_round_csv = method_dir / "detection_round_metrics.csv"
        detection_client_csv = method_dir / "detection_client_log.csv"
        overall_detection_csv = method_dir / "overall_detection_metrics.csv"
        write_csv_header(
            detection_round_csv,
            [
                "Round",
                "TP",
                "FP",
                "TN",
                "FN",
                "DR",
                "FDR",
                "Detection Precision",
                "Detection Recall",
                "Detection F1",
                "Err Mean",
                "Err p50",
                "Err p90",
                "Score Mean",
                "Score p50",
                "Score p90",
            ],
        )
        write_csv_header(
            detection_client_csv,
            [
                "Round",
                "ClientID",
                "Is_Potential_Attacker",
                "Is_Actual_Attacker",
                "Recon_Error",
                "Anomaly_Score",
                "Predicted_Attacker",
                "Aggregation_Weight",
            ],
        )
        write_csv_header(
            overall_detection_csv,
            ["TP", "FP", "TN", "FN", "DR", "FDR", "Detection Precision", "Detection Recall", "Detection F1"],
        )
        paths.update(
            {
                "detection_round": detection_round_csv,
                "detection_client": detection_client_csv,
                "overall_detection": overall_detection_csv,
            }
        )
    return paths


def run_one_method(
    method: str,
    schedule_by_round: Mapping[int, Sequence[ScheduleRow]],
    output_root: Path,
    attack_type: str,
    attacker_percentage: float,
    alpha: float,
    seed: int,
    live_rounds: int,
    threshold_renormalize: bool,
    save_final_model: bool,
) -> Dict[str, float]:
    print(f"\n==============================")
    print(f"Running method: {method}")
    print(f"==============================")

    is_proposed = method in {"proposed_thresholding", "proposed_credit_scoring"}
    method_dir = ensure_dir(output_root / method)
    log_paths = initialize_method_logs(method_dir, is_proposed)
    if method == "krum":
        write_csv_header(method_dir / "krum_scores.csv", ["Round", "ClientID", "Krum Score", "Selected"] )
    if method == "multikrum":
        write_csv_header(method_dir / "multikrum_scores.csv", ["Round", "ClientID", "MultiKrum Score", "Selected"] )

    set_all_seeds(seed)
    test_loader = make_test_loader()

    warmup_global_path = find_warmup_global_path()
    global_model = FL_CNN().to(cfg.DEVICE)
    global_model.load_state_dict(torch.load(warmup_global_path, map_location=cfg.DEVICE))
    global_model.eval()

    detector: Optional[GraphAutoencoder] = None
    benign_ref_dirs: Optional[torch.Tensor] = None
    if is_proposed:
        detector, benign_ref_dirs = load_detector_and_reference()

    # Robust aggregators use the known upper bound f = potential attackers per round.
    f_byzantine_bound = int(cfg.CLIENTS_PER_ROUND * (float(attacker_percentage) / 100.0))
    f_trim = f_byzantine_bound

    overall_tp = overall_fp = overall_tn = overall_fn = 0
    final_metrics: Dict[str, float] = {}

    for round_num in range(1, live_rounds + 1):
        round_rows = list(schedule_by_round.get(round_num, []))
        if not round_rows:
            raise RuntimeError(f"Schedule has no clients for round {round_num}.")

        prev_global_cpu = state_to_cpu(global_model.state_dict())
        no_attack = method == "fedavg_no_attack"
        local_states, sample_sizes, actual_flags, potential_flags, avg_train_acc, avg_train_loss = produce_local_submissions_for_round(
            prev_global_cpu,
            round_rows=round_rows,
            round_num=round_num,
            seed=seed,
            attack_type=attack_type,
            no_attack=no_attack,
        )

        if not local_states:
            print(f"Round {round_num}: no valid local submissions. Global unchanged.")
            new_global_cpu = prev_global_cpu
        elif method in {"fedavg_no_attack", "fedavg_attack_nodefense"}:
            new_global_cpu = fedavg_aggregate(local_states, sample_sizes, prev_global_cpu)
        elif method == "krum":
            new_global_cpu, selected_cid, score_rows = krum_aggregate(local_states, prev_global_cpu, f_byzantine_bound)
            append_csv_rows(method_dir / "krum_scores.csv", [[round_num, cid, score, int(cid == selected_cid)] for cid, score in score_rows])
        elif method == "multikrum":
            new_global_cpu, selected_cids, score_rows = multikrum_aggregate(local_states, prev_global_cpu, f_byzantine_bound, m=None)
            selected_set = set(selected_cids)
            append_csv_rows(method_dir / "multikrum_scores.csv", [[round_num, cid, score, int(cid in selected_set)] for cid, score in score_rows])
        elif method == "coord_median":
            new_global_cpu = coordinatewise_median_aggregate(local_states, prev_global_cpu)
        elif method == "trimmed_mean":
            new_global_cpu = coordinatewise_trimmed_mean_aggregate(local_states, prev_global_cpu, f_trim=f_trim)
        elif method == "proposed_thresholding":
            assert detector is not None and benign_ref_dirs is not None
            new_global_cpu, recon_errors, anomaly_scores, pred_attack, agg_weights, counts = proposed_detection_and_weights(
                local_states,
                sample_sizes,
                actual_flags,
                prev_global_cpu,
                detector,
                benign_ref_dirs,
                defense_method="thresholding",
                threshold_renormalize=threshold_renormalize,
            )
            tp, fp, tn, fn = counts
            overall_tp += tp; overall_fp += fp; overall_tn += tn; overall_fn += fn
            log_detection_round_and_clients(
                log_paths, round_num, recon_errors, anomaly_scores, pred_attack, agg_weights,
                potential_flags, actual_flags, tp, fp, tn, fn
            )
        elif method == "proposed_credit_scoring":
            assert detector is not None and benign_ref_dirs is not None
            new_global_cpu, recon_errors, anomaly_scores, pred_attack, agg_weights, counts = proposed_detection_and_weights(
                local_states,
                sample_sizes,
                actual_flags,
                prev_global_cpu,
                detector,
                benign_ref_dirs,
                defense_method="credit_scoring",
                threshold_renormalize=threshold_renormalize,
            )
            tp, fp, tn, fn = counts
            overall_tp += tp; overall_fp += fp; overall_tn += tn; overall_fn += fn
            log_detection_round_and_clients(
                log_paths, round_num, recon_errors, anomaly_scores, pred_attack, agg_weights,
                potential_flags, actual_flags, tp, fp, tn, fn
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        load_state_to_model(global_model, new_global_cpu)
        test_acc, test_prec, test_rec, test_f1 = evaluate_global_model(global_model, test_loader)
        num_actual = int(sum(1 for v in actual_flags.values() if v))
        total_samples = int(sum(sample_sizes.values()))

        append_csv_row(
            log_paths["round_metrics"],
            [
                round_num,
                test_acc,
                test_prec,
                test_rec,
                test_f1,
                avg_train_acc,
                avg_train_loss,
                len(local_states),
                num_actual,
                total_samples,
            ],
        )

        print(
            f"{method} | Round {round_num}/{live_rounds}: "
            f"TestAcc={test_acc:.4f}, Prec={test_prec:.4f}, Rec={test_rec:.4f}, F1={test_f1:.4f}, "
            f"TrainAcc={avg_train_acc:.4f}, TrainLoss={avg_train_loss:.6f}, ActualAtk={num_actual}"
        )

        final_metrics = {
            "Final Test Accuracy": test_acc,
            "Final Test Precision": test_prec,
            "Final Test Recall": test_rec,
            "Final Test F1 Score": test_f1,
            "Final Avg Train Accuracy": avg_train_acc,
            "Final Avg Train Loss": avg_train_loss,
        }

    if is_proposed:
        det = detection_metrics_from_counts(overall_tp, overall_fp, overall_tn, overall_fn)
        append_csv_row(
            log_paths["overall_detection"],
            [
                overall_tp,
                overall_fp,
                overall_tn,
                overall_fn,
                det["DR"],
                det["FDR"],
                det["Detection Precision"],
                det["Detection Recall"],
                det["Detection F1"],
            ],
        )
        final_metrics.update({
            "Overall TP": overall_tp,
            "Overall FP": overall_fp,
            "Overall TN": overall_tn,
            "Overall FN": overall_fn,
            "Overall DR": det["DR"],
            "Overall FDR": det["FDR"],
            "Overall Detection Precision": det["Detection Precision"],
            "Overall Detection Recall": det["Detection Recall"],
            "Overall Detection F1": det["Detection F1"],
        })

    if save_final_model:
        torch.save(global_model.state_dict(), method_dir / "final_global_model.pth")

    return final_metrics


def log_detection_round_and_clients(
    log_paths: Mapping[str, Path],
    round_num: int,
    recon_errors: Mapping[str, float],
    anomaly_scores: Mapping[str, float],
    pred_attack: Mapping[str, bool],
    agg_weights: Mapping[str, float],
    potential_flags: Mapping[str, bool],
    actual_flags: Mapping[str, bool],
    tp: int,
    fp: int,
    tn: int,
    fn: int,
) -> None:
    metrics = detection_metrics_from_counts(tp, fp, tn, fn)
    err_values = np.array(list(recon_errors.values()), dtype=float) if recon_errors else np.array([], dtype=float)
    score_values = np.array(list(anomaly_scores.values()), dtype=float) if anomaly_scores else np.array([], dtype=float)

    err_mean = float(np.mean(err_values)) if err_values.size else 0.0
    err_p50 = float(np.percentile(err_values, 50)) if err_values.size else 0.0
    err_p90 = float(np.percentile(err_values, 90)) if err_values.size else 0.0
    score_mean = float(np.mean(score_values)) if score_values.size else 0.0
    score_p50 = float(np.percentile(score_values, 50)) if score_values.size else 0.0
    score_p90 = float(np.percentile(score_values, 90)) if score_values.size else 0.0

    append_csv_row(
        log_paths["detection_round"],
        [
            round_num,
            tp,
            fp,
            tn,
            fn,
            metrics["DR"],
            metrics["FDR"],
            metrics["Detection Precision"],
            metrics["Detection Recall"],
            metrics["Detection F1"],
            err_mean,
            err_p50,
            err_p90,
            score_mean,
            score_p50,
            score_p90,
        ],
    )

    client_rows = []
    for cid in recon_errors.keys():
        client_rows.append([
            round_num,
            cid,
            int(bool(potential_flags.get(cid, False))),
            int(bool(actual_flags.get(cid, False))),
            float(recon_errors.get(cid, 0.0)),
            float(anomaly_scores.get(cid, 0.0)),
            int(bool(pred_attack.get(cid, False))),
            float(agg_weights.get(cid, 0.0)),
        ])
    append_csv_rows(log_paths["detection_client"], client_rows)


def parse_methods(raw_methods: Optional[Sequence[str]]) -> List[str]:
    if not raw_methods:
        return list(ALL_METHODS)
    methods: List[str] = []
    for item in raw_methods:
        if item == "all":
            return list(ALL_METHODS)
        if item not in ALL_METHODS:
            raise ValueError(f"Unknown method '{item}'. Valid methods: {ALL_METHODS}")
        methods.append(item)
    return methods



def validate_schedule(
    schedule_by_round: Mapping[int, Sequence[ScheduleRow]],
    attack_type: str,
    attacker_pct: float,
    alpha: float,
    live_rounds: int,
) -> None:
    """
    Validate that a loaded fixed schedule belongs to the requested condition.

    This prevents silent mistakes such as running signflip/10%/alpha=0.5 while
    accidentally passing a schedule generated for freeride/30%/alpha=1.0.
    """
    if len(schedule_by_round) != live_rounds:
        raise RuntimeError(
            f"Schedule has {len(schedule_by_round)} rounds, expected {live_rounds}."
        )

    expected_rounds = set(range(1, live_rounds + 1))
    found_rounds = set(schedule_by_round.keys())
    if found_rounds != expected_rounds:
        missing = sorted(expected_rounds - found_rounds)
        extra = sorted(found_rounds - expected_rounds)
        raise RuntimeError(
            "Schedule round IDs do not match the requested live rounds. "
            f"Missing rounds: {missing[:10]}{'...' if len(missing) > 10 else ''}; "
            f"extra rounds: {extra[:10]}{'...' if len(extra) > 10 else ''}."
        )

    expected_potential_per_round = int(cfg.CLIENTS_PER_ROUND * (float(attacker_pct) / 100.0))

    for r in range(1, live_rounds + 1):
        rows = list(schedule_by_round.get(r, []))
        if len(rows) != cfg.CLIENTS_PER_ROUND:
            raise RuntimeError(
                f"Round {r} has {len(rows)} clients, expected {cfg.CLIENTS_PER_ROUND}."
            )

        client_ids = [row.client_id for row in rows]
        if len(set(client_ids)) != len(client_ids):
            raise RuntimeError(f"Round {r} contains duplicate client IDs in the schedule.")

        potential_count = sum(1 for row in rows if row.is_potential_attacker)
        if potential_count != expected_potential_per_round:
            raise RuntimeError(
                f"Round {r} has {potential_count} potential attackers, "
                f"expected {expected_potential_per_round} for attacker_pct={attacker_pct}."
            )

        for row in rows:
            if row.attack_type != attack_type:
                raise RuntimeError(
                    f"Schedule attack type mismatch in round {r}: "
                    f"found '{row.attack_type}', expected '{attack_type}'."
                )
            if abs(float(row.attacker_percentage) - float(attacker_pct)) > 1e-9:
                raise RuntimeError(
                    f"Schedule attacker percentage mismatch in round {r}: "
                    f"found {row.attacker_percentage}, expected {attacker_pct}."
                )
            if abs(float(row.alpha) - float(alpha)) > 1e-9:
                raise RuntimeError(
                    f"Schedule alpha mismatch in round {r}: found {row.alpha}, expected {alpha}."
                )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-type", choices=["signflip", "freeride"], default="signflip")
    parser.add_argument("--attacker-pct", type=float, default=float(cfg.PERCENTAGE_ATTACKERS))
    parser.add_argument("--alpha", type=float, default=float(cfg.ATTACK_PROBABILITY))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--live-rounds", type=int, default=int(cfg.T_TOTAL_ROUNDS))
    parser.add_argument("--methods", nargs="*", default=["all"], help=f"Subset of methods or 'all'. Valid: {ALL_METHODS}")
    parser.add_argument("--schedule-path", type=str, default=None)
    parser.add_argument("--force-regenerate-schedule", action="store_true")
    parser.add_argument("--threshold-renormalize", action="store_true", default=True,
                        help="For proposed thresholding, renormalize FedAvg weights over approved clients. This is the mathematically correct drop-and-aggregate behavior.")
    parser.add_argument("--old-threshold-denominator", action="store_true",
                        help="Use old script behavior for thresholding: approved clients use n_k / total selected samples, without renormalization.")
    parser.add_argument("--save-final-model", action="store_true", help="Save final global model for each method.")
    args = parser.parse_args()

    set_all_seeds(args.seed)

    schedule_path = Path(args.schedule_path) if args.schedule_path else schedule_default_path(
        args.attack_type, args.attacker_pct, args.alpha, args.seed, args.live_rounds
    )

    if args.force_regenerate_schedule or not schedule_path.exists():
        print(f"Generating fixed schedule: {schedule_path}")
        generate_fixed_schedule(
            attack_type=args.attack_type,
            attacker_percentage=args.attacker_pct,
            alpha=args.alpha,
            live_rounds=args.live_rounds,
            seed=args.seed,
            out_csv=schedule_path,
        )
    else:
        print(f"Using existing fixed schedule: {schedule_path}")

    schedule_by_round = load_schedule(schedule_path)
    validate_schedule(
        schedule_by_round=schedule_by_round,
        attack_type=args.attack_type,
        attacker_pct=args.attacker_pct,
        alpha=args.alpha,
        live_rounds=args.live_rounds,
    )
    methods = parse_methods(args.methods)

    # Safety checks for robust aggregators.
    f_bound = int(cfg.CLIENTS_PER_ROUND * (float(args.attacker_pct) / 100.0))
    if any(m in methods for m in ["krum", "multikrum"]) and (2 * f_bound + 2 >= cfg.CLIENTS_PER_ROUND):
        raise ValueError(
            f"Krum/Multi-Krum are invalid for CLIENTS_PER_ROUND={cfg.CLIENTS_PER_ROUND}, f={f_bound}. "
            "The paper condition is 2f + 2 < n. Remove krum/multikrum from --methods or lower attacker percentage."
        )
    if "trimmed_mean" in methods and (2 * f_bound >= cfg.CLIENTS_PER_ROUND):
        raise ValueError(
            f"Coordinate-wise trimmed mean is invalid for CLIENTS_PER_ROUND={cfg.CLIENTS_PER_ROUND}, f_trim={f_bound}; "
            "it requires 2*f_trim < n. Remove trimmed_mean from --methods or lower attacker percentage."
        )

    safe_pct = str(args.attacker_pct).replace(".", "p")
    safe_alpha = str(args.alpha).replace(".", "p")
    output_root = ensure_dir(
        Path(cfg.LOG_DIR_BASE) / "fair_comparison_fixed_schedule" / f"{args.attack_type}_atk{safe_pct}_alpha{safe_alpha}_seed{args.seed}_rounds{args.live_rounds}"
    )

    # Copy metadata about the run.
    with (output_root / "run_config.json").open("w") as f:
        json.dump(
            {
                "attack_type": args.attack_type,
                "attacker_pct": args.attacker_pct,
                "alpha": args.alpha,
                "seed": args.seed,
                "live_rounds": args.live_rounds,
                "methods": methods,
                "schedule_path": str(schedule_path),
                "warmup_global_path": str(find_warmup_global_path()),
                "clients_per_round": cfg.CLIENTS_PER_ROUND,
                "local_epochs": cfg.LOCAL_CNN_EPOCHS,
                "cnn_batch_size": cfg.CNN_BATCH_SIZE,
                "cnn_learning_rate": cfg.CNN_LEARNING_RATE,
                "threshold_renormalize": bool(not args.old_threshold_denominator),
            },
            f,
            indent=2,
        )

    summary_csv = output_root / "overall_method_summary.csv"
    write_csv_header(
        summary_csv,
        [
            "Method",
            "Final Test Accuracy",
            "Final Test Precision",
            "Final Test Recall",
            "Final Test F1 Score",
            "Final Avg Train Accuracy",
            "Final Avg Train Loss",
            "Overall TP",
            "Overall FP",
            "Overall TN",
            "Overall FN",
            "Overall DR",
            "Overall FDR",
            "Overall Detection Precision",
            "Overall Detection Recall",
            "Overall Detection F1",
        ],
    )

    for method in methods:
        metrics = run_one_method(
            method=method,
            schedule_by_round=schedule_by_round,
            output_root=output_root,
            attack_type=args.attack_type,
            attacker_percentage=args.attacker_pct,
            alpha=args.alpha,
            seed=args.seed,
            live_rounds=args.live_rounds,
            threshold_renormalize=bool(not args.old_threshold_denominator),
            save_final_model=bool(args.save_final_model),
        )
        append_csv_row(
            summary_csv,
            [
                method,
                metrics.get("Final Test Accuracy", 0.0),
                metrics.get("Final Test Precision", 0.0),
                metrics.get("Final Test Recall", 0.0),
                metrics.get("Final Test F1 Score", 0.0),
                metrics.get("Final Avg Train Accuracy", 0.0),
                metrics.get("Final Avg Train Loss", 0.0),
                metrics.get("Overall TP", ""),
                metrics.get("Overall FP", ""),
                metrics.get("Overall TN", ""),
                metrics.get("Overall FN", ""),
                metrics.get("Overall DR", ""),
                metrics.get("Overall FDR", ""),
                metrics.get("Overall Detection Precision", ""),
                metrics.get("Overall Detection Recall", ""),
                metrics.get("Overall Detection F1", ""),
            ],
        )

    print(f"\nAll requested methods completed. Summary: {summary_csv}")


if __name__ == "__main__":
    main()

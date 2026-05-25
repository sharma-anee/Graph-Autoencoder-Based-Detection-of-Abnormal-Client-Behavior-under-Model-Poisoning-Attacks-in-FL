"""
01_warmup_train_detector_no_local_saves.py

Purpose
-------
Run the common benign warm-up phase and train the GAE detector without saving
per-client local model HDF5 files.

Outputs under cfg.LOG_DIR_BASE
------------------------------
- global_model_warmup.pth         : common warm-up global model for all methods
- global_model_stage1.pth         : compatibility copy with your older scripts
- benign_ref_dirs.pth             : mean benign Conv2 update direction
- detector_model.pth              : trained GAE detector
- warmup/warmup_global_metrics.csv
- warmup/warmup_train_metrics.csv
- warmup/warmup_schedule.csv
- warmup/gae_training_loss.csv

This warm-up should be run once. All comparison methods should then start from
exactly this same global_model_warmup.pth.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple, cast

from torch_geometric.data import Data

import numpy as np
import torch
from torch_geometric.loader import DataLoader as GraphDataLoader
from tqdm import tqdm

from config import cfg
from feature_extractor import convert_update_to_graph
from fl_fair_common import (
    append_csv_row,
    append_csv_rows,
    ensure_dir,
    evaluate_global_model,
    fedavg_aggregate,
    list_client_folders,
    load_state_to_model,
    make_test_loader,
    set_all_seeds,
    stable_seed,
    state_to_cpu,
    train_one_client_from_global,
    write_csv_header,
)
from models import FL_CNN, GraphAutoencoder


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


def train_gae_from_graphs(graph_dataset: List[Data], out_dir: Path) -> None:
    if not graph_dataset:
        raise RuntimeError("No benign graphs available to train the GAE detector.")

    graph_loader = GraphDataLoader(graph_dataset, batch_size=cfg.GAE_BATCH_SIZE, shuffle=True)
    first_x, _first_edge_index = get_graph_tensors(graph_dataset[0], "First benign graph")
    num_node_features = int(first_x.size(1))

    gae_model = GraphAutoencoder(
        in_channels=num_node_features,
        hidden_channels=cfg.GAE_HIDDEN_CHANNELS,
        out_channels=cfg.GAE_EMBEDDING_SIZE,
    ).to(cfg.DEVICE)

    optimizer = torch.optim.Adam(gae_model.parameters(), lr=cfg.GAE_LEARNING_RATE)
    bce_loss_fn = torch.nn.BCELoss()
    mse_loss_fn = torch.nn.MSELoss()

    loss_csv = out_dir / "gae_training_loss.csv"
    write_csv_header(loss_csv, ["Epoch", "Combined Loss"])

    gae_model.train()
    for epoch in range(1, cfg.GAE_EPOCHS + 1):
        total_loss = 0.0
        for batch in tqdm(graph_loader, desc=f"GAE Epoch {epoch}/{cfg.GAE_EPOCHS}", leave=False):
            batch = batch.to(cfg.DEVICE)
            optimizer.zero_grad(set_to_none=True)

            batch_x, batch_edge_index = get_graph_tensors(batch, "GAE training batch")
            z = gae_model.encode(batch_x, batch_edge_index)
            recon_adj, recon_feat = gae_model.decode(z)

            num_nodes = int(batch_x.size(0))
            true_adj = torch.zeros((num_nodes, num_nodes), device=cfg.DEVICE)
            true_adj[batch_edge_index[0], batch_edge_index[1]] = 1.0

            loss_a = bce_loss_fn(recon_adj, true_adj)
            loss_x = mse_loss_fn(recon_feat, batch_x)
            loss = cfg.LAMBDA_ADJACENCY * loss_a + cfg.LAMBDA_FEATURES * loss_x
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(graph_loader))
        print(f"GAE Epoch {epoch}/{cfg.GAE_EPOCHS}, Combined Loss: {avg_loss:.6f}")
        append_csv_row(loss_csv, [epoch, avg_loss])

    torch.save(gae_model.state_dict(), Path(cfg.LOG_DIR_BASE) / "detector_model.pth")
    print(f"Saved detector model: {Path(cfg.LOG_DIR_BASE) / 'detector_model.pth'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-rounds", type=int, default=cfg.R0_PRETRAIN_ROUNDS)
    args = parser.parse_args()

    set_all_seeds(args.seed)
    base_dir = ensure_dir(cfg.LOG_DIR_BASE)
    warmup_dir = ensure_dir(base_dir / "warmup")

    global_metrics_csv = warmup_dir / "warmup_global_metrics.csv"
    train_metrics_csv = warmup_dir / "warmup_train_metrics.csv"
    schedule_csv = warmup_dir / "warmup_schedule.csv"

    write_csv_header(global_metrics_csv, ["Round", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score"])
    write_csv_header(train_metrics_csv, ["Round", "Avg Train Accuracy", "Avg Train Loss", "Num Clients Trained", "Total Samples"])
    write_csv_header(schedule_csv, ["Round", "ClientID"])

    all_clients = list_client_folders()
    test_loader = make_test_loader()
    global_model = FL_CNN().to(cfg.DEVICE)

    # Store only Conv2 benign updates for detector training, not full local models.
    benign_conv2_updates: List[torch.Tensor] = []

    rng = np.random.default_rng(args.seed)
    print(f"Warm-up rounds: 1..{args.warmup_rounds}")
    print(f"Clients per round: {cfg.CLIENTS_PER_ROUND}")
    print("No per-client local model weights will be saved.")

    for round_num in range(1, args.warmup_rounds + 1):
        print(f"\n=== Warm-up Round {round_num}/{args.warmup_rounds} ===")
        prev_global_cpu = state_to_cpu(global_model.state_dict())

        # Deterministic selected clients for warm-up.
        selected_clients = list(rng.choice(all_clients, size=cfg.CLIENTS_PER_ROUND, replace=False))
        append_csv_rows(schedule_csv, [[round_num, cid] for cid in selected_clients])

        local_states = {}
        sample_sizes = {}
        local_accs: List[float] = []
        local_losses: List[float] = []

        for client_id in tqdm(selected_clients, desc="Benign local training"):
            local_state, n_k, train_acc, train_loss = train_one_client_from_global(
                prev_global_cpu,
                client_id=client_id,
                round_num=round_num,
                seed=args.seed,
            )
            if local_state is None or n_k <= 0:
                continue

            local_states[client_id] = local_state
            sample_sizes[client_id] = n_k
            local_accs.append(train_acc)
            local_losses.append(train_loss)

            conv2_update = local_state["conv2.weight"].float() - prev_global_cpu["conv2.weight"].float()
            benign_conv2_updates.append(conv2_update.cpu())

        if local_states:
            new_global_cpu = fedavg_aggregate(local_states, sample_sizes, prev_global_cpu)
            load_state_to_model(global_model, new_global_cpu)
        else:
            print("No valid client updates in this warm-up round; global model unchanged.")

        test_acc, test_prec, test_rec, test_f1 = evaluate_global_model(global_model, test_loader)
        avg_train_acc = float(np.mean(local_accs)) if local_accs else 0.0
        avg_train_loss = float(np.mean(local_losses)) if local_losses else 0.0
        total_samples = int(sum(sample_sizes.values()))

        append_csv_row(global_metrics_csv, [round_num, test_acc, test_prec, test_rec, test_f1])
        append_csv_row(train_metrics_csv, [round_num, avg_train_acc, avg_train_loss, len(local_states), total_samples])

        print(
            f"Round {round_num}: Test Acc={test_acc:.4f}, Precision={test_prec:.4f}, "
            f"Recall={test_rec:.4f}, F1={test_f1:.4f}, AvgTrainAcc={avg_train_acc:.4f}, "
            f"AvgTrainLoss={avg_train_loss:.6f}"
        )

    # Save the common warm-up global model.
    warmup_global_path = base_dir / "global_model_warmup.pth"
    compatibility_path = base_dir / "global_model_stage1.pth"
    torch.save(global_model.state_dict(), warmup_global_path)
    torch.save(global_model.state_dict(), compatibility_path)
    print(f"Saved common warm-up global model: {warmup_global_path}")
    print(f"Saved compatibility copy: {compatibility_path}")

    if not benign_conv2_updates:
        raise RuntimeError("No benign Conv2 updates collected; cannot train detector.")

    benign_updates_tensor = torch.stack(benign_conv2_updates, dim=0)
    benign_ref_dirs = benign_updates_tensor.mean(dim=0)
    benign_ref_path = base_dir / "benign_ref_dirs.pth"
    torch.save(benign_ref_dirs, benign_ref_path)
    print(f"Saved benign reference directions: {benign_ref_path}")

    print("Creating benign graphs in memory for GAE training...")
    graph_dataset: List[Data] = []
    for conv2_update in tqdm(benign_conv2_updates, desc="Benign graph construction"):
        alignment_feature = compute_alignment_feature(conv2_update, benign_ref_dirs)
        graph = convert_update_to_graph(conv2_update, alignment_feature)
        graph_x, _graph_edge_index = get_graph_tensors(graph, "Benign graph construction")
        if graph_x.size(1) != 7:
            raise RuntimeError(f"Expected 7 node features, got {graph_x.size(1)}")
        graph_dataset.append(graph.cpu())

    train_gae_from_graphs(graph_dataset, warmup_dir)
    print("Warm-up and detector training completed successfully.")


if __name__ == "__main__":
    main()

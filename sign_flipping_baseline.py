import os
import csv
import math
import h5py
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import precision_score, recall_score, f1_score

# --- Project imports ---
from config import cfg
from models import FL_CNN

# Repro
random.seed(42); np.random.seed(42); torch.manual_seed(42)

def evaluate_global_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = (correct / total) * 100 if total > 0 else 0.0
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    rec  = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    return acc, prec, rec, f1

def make_target_transform_numeric(class_names):
    # Map ImageFolder local class index -> true numeric label (folder name)
    return lambda local_idx: int(class_names[local_idx])

def main():
    """Plain FedAvg baseline (NO detection/defense) under SIGN-FLIPPING attack."""
    print(f"--- 🟢 BASELINE: FedAvg (No Defense) | Sign Flipping | Rounds {cfg.R0_PRETRAIN_ROUNDS + 1}..{cfg.T_TOTAL_ROUNDS} ---")

    # 1) Load Stage-1 global model (starting point)
    stage1_global_path = os.path.join(cfg.LOG_DIR_BASE, "global_model_stage1.pth")
    if not os.path.exists(stage1_global_path):
        raise FileNotFoundError(f"Missing Stage-1 global model: {stage1_global_path}")
    global_model = FL_CNN().to(cfg.DEVICE)
    global_model.load_state_dict(torch.load(stage1_global_path, map_location=cfg.DEVICE))
    print("✅ Loaded Stage-1 global model.")

    # 2) Logging / output dirs
    log_subdir = f"{cfg.PERCENTAGE_ATTACKERS}p_alpha_{cfg.ATTACK_PROBABILITY}_signflip_NODEFENSE"
    run_log_dir       = os.path.join(cfg.LOG_DIR_BASE, log_subdir)
    local_weights_dir = os.path.join(cfg.LOG_DIR_BASE, "stage_2_local_models",  log_subdir)
    preflip_dir       = os.path.join(cfg.LOG_DIR_BASE, "stage_2_local_models_preflip", log_subdir)
    global_weights_dir= os.path.join(cfg.LOG_DIR_BASE, "stage_2_global_models", log_subdir)

    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(local_weights_dir, exist_ok=True)
    os.makedirs(preflip_dir, exist_ok=True)
    os.makedirs(global_weights_dir, exist_ok=True)

    global_perf_path   = os.path.join(run_log_dir, 'global_performance.csv')
    local_perf_path    = os.path.join(run_log_dir, 'local_performance.csv')
    participation_path = os.path.join(run_log_dir, 'participation_log.csv')  # who attacked, n_k, etc.

    with open(global_perf_path, "w", newline="") as f:
        csv.writer(f).writerow(["Round", "Test Accuracy", "Precision", "Recall", "F1 Score"])
    with open(local_perf_path, "w", newline="") as f:
        csv.writer(f).writerow(["Round", "Avg Train Accuracy", "Avg Train Loss"])
    with open(participation_path, "w", newline="") as f:
        csv.writer(f).writerow(["Round", "ClientID", "Is_Potential_Attacker", "Is_Actual_Attacker", "Samples"])

    print(f"📝 Logging to: {run_log_dir}")

    # 3) Clients & attacker pool
    all_client_folders = [d for d in os.listdir(cfg.DATA_ROOT_PATH)
                          if os.path.isdir(os.path.join(cfg.DATA_ROOT_PATH, d))]
    num_total_attackers = int(len(all_client_folders) * (cfg.PERCENTAGE_ATTACKERS / 100))
    attacker_pool = set(random.sample(all_client_folders, num_total_attackers))
    benign_pool   = set(all_client_folders) - attacker_pool
    print(f"Total clients: {len(all_client_folders)} | Potential attackers: {len(attacker_pool)}")

    # 4) Data pipeline (match Stage-1/2 label handling)
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Test set with true numeric labels
    test_dataset = datasets.ImageFolder(root=cfg.TEST_DATA_PATH, transform=transform)
    test_dataset.target_transform = make_target_transform_numeric(test_dataset.classes)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 5) Live FL rounds (No detection; plain FedAvg)
    for round_num in range(cfg.R0_PRETRAIN_ROUNDS + 1, cfg.T_TOTAL_ROUNDS + 1):
        print(f"\n--- Round {round_num}/{cfg.T_TOTAL_ROUNDS} ---")

        # a) Sample clients (some from attacker pool)
        num_attackers_this_round = int(cfg.CLIENTS_PER_ROUND * (cfg.PERCENTAGE_ATTACKERS / 100))
        selected_benign    = random.sample(list(benign_pool),   cfg.CLIENTS_PER_ROUND - num_attackers_this_round)
        selected_attackers = random.sample(list(attacker_pool), num_attackers_this_round)
        selected_clients   = selected_benign + selected_attackers
        random.shuffle(selected_clients)

        print(f"👥 Selected {len(selected_clients)}: benign={len(selected_benign)} | potential_attackers={len(selected_attackers)}")

        local_models_info   = []  # (state_dict, n_k)
        local_metrics_round = []
        # optional per-round participation logging
        part_rows = []

        # b) Local training + (if actual attacker) sign flipping AFTER training
        for client_id in tqdm(selected_clients, desc="Local Client Training"):
            is_potential_attacker = client_id in attacker_pool
            is_actual_attacker = is_potential_attacker and (random.random() < cfg.ATTACK_PROBABILITY)

            # Load client dataset
            client_train_path = os.path.join(cfg.DATA_ROOT_PATH, client_id, 'train')
            if not os.path.exists(client_train_path) or not os.listdir(client_train_path):
                continue

            client_dataset = datasets.ImageFolder(root=client_train_path, transform=transform)
            client_dataset.target_transform = make_target_transform_numeric(client_dataset.classes)
            local_loader = DataLoader(client_dataset, batch_size=cfg.CNN_BATCH_SIZE, shuffle=True)

            n_k = len(client_dataset)
            num_classes = len(client_dataset.classes)
            print(f"\n🔄 Client {client_id} | Type: {'ATTACKER' if is_actual_attacker else ('POTENTIAL' if is_potential_attacker else 'Benign')} | n_k: {n_k} | classes: {num_classes}")

            # Init local model from current global
            local_model = FL_CNN().to(cfg.DEVICE)
            local_model.load_state_dict(global_model.state_dict())

            optimizer = optim.SGD(local_model.parameters(), lr=cfg.CNN_LEARNING_RATE)
            criterion = nn.CrossEntropyLoss()

            # Train locally
            final_epoch_losses, final_epoch_accuracies = [], []
            local_model.train()
            for epoch in range(cfg.LOCAL_CNN_EPOCHS):
                epoch_loss, correct, total = 0.0, 0, 0
                for images, labels in local_loader:
                    images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                    optimizer.zero_grad()
                    outputs = local_model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                avg_epoch_loss = epoch_loss / len(local_loader)
                epoch_acc = (correct / total) * 100 if total > 0 else 0.0
                print(f"    Epoch {epoch+1}/{cfg.LOCAL_CNN_EPOCHS}: Loss: {avg_epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
            final_epoch_losses.append(avg_epoch_loss)
            final_epoch_accuracies.append(epoch_acc)

            # If actual attacker: save pre-flip snapshot, then flip all params
            if is_actual_attacker:
                preflip_h5 = os.path.join(
                    os.path.join(cfg.LOG_DIR_BASE, "stage_2_local_models_preflip", log_subdir),
                    f"Round_{round_num}_Client_{client_id}_PREATTACK.h5"
                )
                os.makedirs(os.path.dirname(preflip_h5), exist_ok=True)
                with h5py.File(preflip_h5, "w") as hf_pre:
                    for name, param in local_model.state_dict().items():
                        hf_pre.create_dataset(name, data=param.detach().cpu().numpy())

                flipped = {k: -1 * v for k, v in local_model.state_dict().items()}
                local_model.load_state_dict(flipped)

            # Save (post-flip for attackers)
            out_h5 = os.path.join(local_weights_dir, f"Round_{round_num}_Client_{client_id}_{'Attacker' if is_actual_attacker else 'Benign'}.h5")
            with h5py.File(out_h5, "w") as hf:
                for name, param in local_model.state_dict().items():
                    hf.create_dataset(name, data=param.detach().cpu().numpy())

            # Collect for FedAvg
            local_models_info.append((local_model.state_dict(), n_k))
            local_metrics_round.append({'acc': float(np.mean(final_epoch_accuracies)),
                                        'loss': float(np.mean(final_epoch_losses))})
            part_rows.append([round_num, client_id, int(is_potential_attacker), int(is_actual_attacker), n_k])

        # c) Plain FedAvg (NO defenses)
        if local_models_info:
            total_samples = sum(n_k for _, n_k in local_models_info)
            avg_weights = {k: torch.zeros_like(v, dtype=torch.float32)
                           for k, v in global_model.state_dict().items()}
            for state_dict, n_k in local_models_info:
                w = n_k / total_samples if total_samples > 0 else 0.0
                for k in avg_weights.keys():
                    avg_weights[k] += state_dict[k] * w
            global_model.load_state_dict(avg_weights)
            print("✅ Global model updated via plain FedAvg.")

            # Save global model snapshot (HDF5)
            hdf5_global_path = os.path.join(global_weights_dir, f"Round_{round_num}_GlobalModel.h5")
            with h5py.File(hdf5_global_path, "w") as hf:
                for name, param in global_model.state_dict().items():
                    hf.create_dataset(name, data=param.detach().cpu().numpy())
            print("  💾 Saved global model HDF5.")

        # d) Evaluate & log
        test_acc, prec, rec, f1 = evaluate_global_model(global_model, test_loader)
        if local_metrics_round:
            avg_local_acc  = float(np.mean([m['acc']  for m in local_metrics_round]))
            avg_local_loss = float(np.mean([m['loss'] for m in local_metrics_round]))
        else:
            avg_local_acc, avg_local_loss = 0.0, 0.0

        
        print(f"\n📊 Total Samples Processed in Round: {total_samples}")

        print(f"\n🎯 Round {round_num}: Test Accuracy: {test_acc:.2f}% | Precision: {prec:.2f}% | Recall: {rec:.2f}% | F1 Score: {f1:.2f}%")
        with open(global_perf_path, "a", newline="") as f:
            csv.writer(f).writerow([round_num, test_acc, prec, rec, f1])
        with open(local_perf_path, "a", newline="") as f:
            csv.writer(f).writerow([round_num, avg_local_acc, avg_local_loss])
        if part_rows:
            with open(participation_path, "a", newline="") as f:
                writer = csv.writer(f); writer.writerows(part_rows)

    print("\n🎉 Baseline experiment complete (plain FedAvg, no defenses).")

if __name__ == '__main__':
    main()

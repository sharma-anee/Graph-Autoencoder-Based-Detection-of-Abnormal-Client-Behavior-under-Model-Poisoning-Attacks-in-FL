# fl_project/stage_2_fedavg_freeride_nodefense.py
#
# Stage 2: Normal FedAvg with FREE-RIDING attackers (no defense).
# Attackers DO NOT train locally; they submit RANDOM weights/biases (N(0,1)).
# No anomaly detection, no thresholding, no credit scoring — plain sample-weighted FedAvg.

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import random
import csv
from tqdm import tqdm
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score
import h5py

# Import our custom modules
from config import cfg
from models import FL_CNN

random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ----------------------------- Utilities ---------------------------------

def evaluate_global_model(model, test_loader):
    """Evaluates the global model on the held-out test dataset."""
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

    accuracy = (correct / total) * 100 if total > 0 else 0
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    return accuracy, precision, recall, f1


def randomize_state_dict_like(reference_state_dict):
    """
    Create a NEW state_dict where each parameter tensor is replaced by
    pure standard-normal random values with the same shape (N(0,1)).
    """
    new_state = {}
    for k, v in reference_state_dict.items():
        ref = v.detach()
        noise = torch.randn_like(ref, dtype=ref.dtype, device=ref.device)
        new_state[k] = noise
    return new_state


# ------------------------------- Main ------------------------------------

def main():
    print(f"--- 🟣 STAGE 2: FedAvg with Free-Riders (No Defense), Rounds {cfg.R0_PRETRAIN_ROUNDS + 1} to {cfg.T_TOTAL_ROUNDS} ---")

    # --- 1. Load Pre-trained Global Model ---
    try:
        global_model = FL_CNN().to(cfg.DEVICE)
        stage1_global_path = os.path.join(cfg.LOG_DIR_BASE, "global_model_stage1.pth")
        if not os.path.exists(stage1_global_path):
            raise FileNotFoundError(f"Missing Stage-1 global model: {stage1_global_path}")
        global_model.load_state_dict(torch.load(stage1_global_path, map_location=cfg.DEVICE))
        print("✅ Loaded Stage-1 global model.")
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}. Please run Stage 1 first.")
        return

    # --- 2. Logging / Output Dirs ---
    log_subdir = f"{cfg.PERCENTAGE_ATTACKERS}p_alpha_{cfg.ATTACK_PROBABILITY}_freeride_nodefense"
    run_log_dir = os.path.join(cfg.LOG_DIR_BASE, log_subdir)
    os.makedirs(run_log_dir, exist_ok=True)

    # Keep directory names close to your original style
    local_weights_dir  = os.path.join(cfg.LOG_DIR_BASE, "stage_2_local_models", log_subdir)
    global_weights_dir = os.path.join(cfg.LOG_DIR_BASE, "stage_2_global_models", log_subdir)
    os.makedirs(local_weights_dir, exist_ok=True)
    os.makedirs(global_weights_dir, exist_ok=True)

    global_perf_path = os.path.join(run_log_dir, 'global_performance.csv')
    local_perf_path  = os.path.join(run_log_dir, 'local_performance.csv')

    with open(global_perf_path, "w", newline="") as f:
        csv.writer(f).writerow(["Round", "Test Accuracy", "Precision", "Recall", "F1 Score"])
    with open(local_perf_path, "w", newline="") as f:
        csv.writer(f).writerow(["Round", "Avg Train Accuracy", "Avg Train Loss"])

    print(f"📝 Logs will be saved in: {run_log_dir}")

    # --- 3. Prepare Clients and Attackers ---
    all_client_folders = [d for d in os.listdir(cfg.DATA_ROOT_PATH) if os.path.isdir(os.path.join(cfg.DATA_ROOT_PATH, d))]
    num_total_attackers = int(len(all_client_folders) * (cfg.PERCENTAGE_ATTACKERS / 100))
    attacker_folders = set(random.sample(all_client_folders, num_total_attackers))
    benign_folders = set(all_client_folders) - attacker_folders

    # --- 4. Data transforms and test loader ---
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    def make_target_transform_numeric(class_names):
        return lambda local_idx: int(class_names[local_idx])

    test_dataset = datasets.ImageFolder(root=cfg.TEST_DATA_PATH, transform=transform)
    test_dataset.target_transform = make_target_transform_numeric(test_dataset.classes)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # --- 5. FL Rounds (plain FedAvg) ---
    for round_num in range(cfg.R0_PRETRAIN_ROUNDS + 1, cfg.T_TOTAL_ROUNDS + 1):
        print(f"\n--- Round {round_num}/{cfg.T_TOTAL_ROUNDS} ---")

        previous_global = {k: v.clone() for k, v in global_model.state_dict().items()}

        # a) Select clients — simple version, no guards
        num_attackers_this_round = int(cfg.CLIENTS_PER_ROUND * (cfg.PERCENTAGE_ATTACKERS / 100))
        selected_benign = random.sample(list(benign_folders), cfg.CLIENTS_PER_ROUND - num_attackers_this_round)
        selected_attackers = random.sample(list(attacker_folders), num_attackers_this_round)
        selected_clients = selected_benign + selected_attackers
        random.shuffle(selected_clients)

        print(f"👥 Selected {len(selected_clients)} clients for this round:")
        print(f"  - Benign Clients ({len(selected_benign)}): {selected_benign}")
        print(f"  - Potential Attackers ({len(selected_attackers)}): {selected_attackers}")

        # b) Local phase: benign -> train; attacker (actual) -> random weights
        client_updates = {}
        client_sample_sizes = {}
        client_local_metrics = {}
        actual_attackers_in_round = []

        for client_id in tqdm(selected_clients, desc="Local Client Phase"):
            is_potential_attacker = client_id in attacker_folders
            is_actual_attacker = is_potential_attacker and (random.random() < cfg.ATTACK_PROBABILITY)
            if is_actual_attacker:
                actual_attackers_in_round.append(client_id)

            # Load client dataset
            client_train_path = os.path.join(cfg.DATA_ROOT_PATH, client_id, 'train')
            if not os.path.exists(client_train_path) or not os.listdir(client_train_path):
                n_k = 0
            else:
                client_dataset = datasets.ImageFolder(root=client_train_path, transform=transform)
                client_dataset.target_transform = make_target_transform_numeric(client_dataset.classes)
                local_loader = DataLoader(client_dataset, batch_size=cfg.CNN_BATCH_SIZE, shuffle=True)
                n_k = len(client_dataset)
            client_sample_sizes[client_id] = n_k

            # Initialize local with current global
            local_model = FL_CNN().to(cfg.DEVICE)
            local_model.load_state_dict(global_model.state_dict())

            if is_actual_attacker:
                # FREE-RIDER: submit random parameters (no training)
                local_model.load_state_dict(randomize_state_dict_like(local_model.state_dict()))
                #client_local_metrics[client_id] = {'acc': 0.0, 'loss': 0.0}
            else:
                # Benign: standard local training
                optimizer = optim.SGD(local_model.parameters(), lr=cfg.CNN_LEARNING_RATE)
                criterion = nn.CrossEntropyLoss()
                final_epoch_losses, final_epoch_accuracies = [], []
                local_model.train()

                if n_k > 0:
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
                        print(f"    {client_id} | Epoch {epoch+1}/{cfg.LOCAL_CNN_EPOCHS}: Loss {avg_epoch_loss:.4f}, Acc {epoch_acc:.2f}%")
                    final_epoch_losses.append(avg_epoch_loss)
                    final_epoch_accuracies.append(epoch_acc)
                else:
                    final_epoch_losses.append(0.0)
                    final_epoch_accuracies.append(0.0)

                client_local_metrics[client_id] = {
                    'acc': float(np.mean(final_epoch_accuracies)),
                    'loss': float(np.mean(final_epoch_losses))
                }

            # Save local submission to HDF5 (for parity with your original)
            hdf5_path = os.path.join(local_weights_dir, f"Round_{round_num}_Client_{client_id}_{'FreeRider' if is_actual_attacker else 'Benign'}.h5")
            with h5py.File(hdf5_path, "w") as hf:
                for name, param in local_model.state_dict().items():
                    hf.create_dataset(name, data=param.detach().cpu().numpy())

            # Stash for aggregation
            client_updates[client_id] = {k: v.clone().detach() for k, v in local_model.state_dict().items()}

        print(f"🟠 Actual free-riders this round: {actual_attackers_in_round if actual_attackers_in_round else 'None'}")

        # c) Plain FedAvg aggregation (sample-weighted)
        total_samples = sum(max(0, n) for n in client_sample_sizes.values())
        if total_samples > 0:
            avg_weights = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_model.state_dict().items()}
            for cid, state_dict in client_updates.items():
                n_k = max(0, client_sample_sizes.get(cid, 0))
                if n_k == 0:
                    continue
                weight = n_k / total_samples
                for k in avg_weights.keys():
                    avg_weights[k] += state_dict[k] * weight
            global_model.load_state_dict(avg_weights)
        else:
            print("⚠️ No samples this round; keeping previous global model.")
            global_model.load_state_dict(previous_global)

        # d) Save global model (HDF5)
        hdf5_global_path = os.path.join(global_weights_dir, f"Round_{round_num}_GlobalModel.h5")
        with h5py.File(hdf5_global_path, "w") as hf:
            for name, param in global_model.state_dict().items():
                hf.create_dataset(name, data=param.detach().cpu().numpy())
        print("  💾 Global model weights saved.")

        # e) Evaluate global model
        print("\n--- 🔬 Evaluating Global Model on Test Set ---")
        test_acc, prec, rec, f1 = evaluate_global_model(global_model, test_loader)
        avg_local_acc  = float(np.mean([v['acc'] for v in client_local_metrics.values()])) if client_local_metrics else 0.0
        avg_local_loss = float(np.mean([v['loss'] for v in client_local_metrics.values()])) if client_local_metrics else 0.0
        total_samples_processed = total_samples

        print(f"\n📊 Total Samples Processed in Round: {total_samples_processed}")
        if len(client_sample_sizes.values()) > 0:
            print(f"   Breakdown: {' + '.join(map(str, client_sample_sizes.values()))} = {total_samples_processed}")
        print(f"🎯 Round {round_num}: Test Accuracy: {test_acc:.2f}% | Precision: {prec:.2f}% | Recall: {rec:.2f}% | F1 Score: {f1:.2f}%")

        with open(global_perf_path, "a", newline="") as f:
            csv.writer(f).writerow([round_num, test_acc, prec, rec, f1])
        with open(local_perf_path, "a", newline="") as f:
            csv.writer(f).writerow([round_num, avg_local_acc, avg_local_loss])

    print("🎉 Experiment complete.")


if __name__ == '__main__':
    main()

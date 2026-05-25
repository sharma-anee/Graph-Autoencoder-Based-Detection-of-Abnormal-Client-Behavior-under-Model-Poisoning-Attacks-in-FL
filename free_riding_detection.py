# fl_project/stage_2_live_detection_freeride.py
#
# Stage 2: Live Detection for FREE-RIDING attack
# Attack behavior: attackers DO NOT train locally; they submit RANDOM weights/biases to the server.
# The rest of the pipeline (graph construction, GAE-based anomaly scoring, and secure aggregation)
# mirrors the sign-flipping script.

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
import torch.nn.functional as F
import math
import h5py

# Import our custom modules
from config import cfg
from models import FL_CNN, GraphAutoencoder
from feature_extractor import convert_update_to_graph

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
    """Runs the live detection phase (Stage 2) for FREE-RIDING attack."""
    print(f"--- 🟣 STAGE 2: Live Detection (Free-Riding Attack, Rounds {cfg.R0_PRETRAIN_ROUNDS + 1} to {cfg.T_TOTAL_ROUNDS}) ---")

    # --- 1. Load Pre-trained Models and Reference Data ---
    try:
        global_model = FL_CNN().to(cfg.DEVICE)
        stage1_global_path = os.path.join(cfg.LOG_DIR_BASE, "global_model_stage1.pth")
        if not os.path.exists(stage1_global_path):
            raise FileNotFoundError(f"Missing Stage-1 global model: {stage1_global_path}")
        global_model.load_state_dict(torch.load(stage1_global_path, map_location=cfg.DEVICE))

        num_node_features = 7  # 6 rich features + 1 alignment feature
        detector_model = GraphAutoencoder(
            in_channels=num_node_features,
            hidden_channels=cfg.GAE_HIDDEN_CHANNELS,
            out_channels=cfg.GAE_EMBEDDING_SIZE
        ).to(cfg.DEVICE)
        detector_model.load_state_dict(torch.load(os.path.join(cfg.LOG_DIR_BASE, 'detector_model.pth'), map_location=cfg.DEVICE))
        detector_model.eval()  # Set to evaluation mode

        benign_ref_dirs = torch.load(os.path.join(cfg.LOG_DIR_BASE, 'benign_ref_dirs.pth'), map_location=cfg.DEVICE)
        print("✅ Models and reference data loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ ERROR: A required file was not found: {e}. Please run Stage 1 first.")
        return

    # --- 2. Setup Logging ---
    log_subdir = f"{cfg.PERCENTAGE_ATTACKERS}p_alpha_{cfg.ATTACK_PROBABILITY}_freeride_{cfg.DEFENSE_METHOD}"
    run_log_dir = os.path.join(cfg.LOG_DIR_BASE, log_subdir)
    os.makedirs(run_log_dir, exist_ok=True)

    local_weights_dir  = os.path.join(cfg.LOG_DIR_BASE, "stage_2_local_models", log_subdir)
    global_weights_dir = os.path.join(cfg.LOG_DIR_BASE, "stage_2_global_models", log_subdir)

    os.makedirs(local_weights_dir, exist_ok=True)
    os.makedirs(global_weights_dir, exist_ok=True)

    global_perf_path = os.path.join(run_log_dir, 'global_performance.csv')
    local_perf_path = os.path.join(run_log_dir, 'local_performance.csv')
    detection_log_path = os.path.join(run_log_dir, 'detection_log.csv')
    detector_metrics_path = os.path.join(run_log_dir, 'detector_metrics_summary.csv')

    with open(global_perf_path, "w", newline="") as f: csv.writer(f).writerow(["Round", "Test Accuracy", "Precision", "Recall", "F1 Score"])
    with open(local_perf_path, "w", newline="") as f: csv.writer(f).writerow(["Round", "Avg Train Accuracy", "Avg Train Loss"])
    with open(detection_log_path, "w", newline="") as f: csv.writer(f).writerow(["Round", "ClientID", "Is_Actual_Attacker", "Recon_Error", "Anomaly_Score", "Decision", "Aggregation_Weight"])
    with open(detector_metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["Round", "TP", "FP", "TN", "FN", "TPR", "FPR", "Err_Mean", "Err_p50", "Err_p90", "Score_Mean", "Score_p50", "Score_p90"])

    print(f"📝 Logging configured. Results will be saved in: {run_log_dir}")
    
    # --- 3. Prepare Clients and Attackers ---
    all_client_folders = [d for d in os.listdir(cfg.DATA_ROOT_PATH) if os.path.isdir(os.path.join(cfg.DATA_ROOT_PATH, d))]
    num_total_attackers = int(len(all_client_folders) * (cfg.PERCENTAGE_ATTACKERS / 100))
    attacker_folders = set(random.sample(all_client_folders, num_total_attackers))
    benign_folders = set(all_client_folders) - attacker_folders

    # --- 4. Live FL Loop with Detection ---
    transform = transforms.Compose([transforms.Grayscale(1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    def make_target_transform_numeric(class_names):
        return lambda local_idx: int(class_names[local_idx])

    test_dataset = datasets.ImageFolder(root=cfg.TEST_DATA_PATH, transform=transform)
    test_dataset.target_transform = make_target_transform_numeric(test_dataset.classes)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for round_num in range(cfg.R0_PRETRAIN_ROUNDS + 1, cfg.T_TOTAL_ROUNDS + 1):
        print(f"\n--- Round {round_num}/{cfg.T_TOTAL_ROUNDS} ---")

        previous_global_weights = {k: v.clone() for k, v in global_model.state_dict().items()}

        # a. Select clients (same style as sign-flip)
        num_attackers_this_round = int(cfg.CLIENTS_PER_ROUND * (cfg.PERCENTAGE_ATTACKERS / 100))
        selected_benign = random.sample(list(benign_folders), cfg.CLIENTS_PER_ROUND - num_attackers_this_round)
        selected_attackers = random.sample(list(attacker_folders), num_attackers_this_round)
        selected_clients = selected_benign + selected_attackers
        random.shuffle(selected_clients)

        print(f"\n👥 Selected {len(selected_clients)} clients for this round:")
        print(f"  - Benign Clients ({len(selected_benign)}): {selected_benign}")
        print(f"  - Potential Attackers ({len(selected_attackers)}): {selected_attackers}")
        
        client_updates, client_is_actual_attacker, client_local_metrics = {}, {}, {}
        client_sample_sizes = {}  # To store n_k for each client

        # b. Local Training Phase
        actual_attackers_in_round = []
        for client_id in tqdm(selected_clients, desc="Local Client Training"):
            is_potential_attacker = client_id in attacker_folders
            is_actual_attacker = is_potential_attacker and (random.random() < cfg.ATTACK_PROBABILITY)
            client_is_actual_attacker[client_id] = is_actual_attacker
            if is_actual_attacker:
                actual_attackers_in_round.append(client_id)

            # load client dataset
            client_train_path = os.path.join(cfg.DATA_ROOT_PATH, client_id, 'train')
            if not os.path.exists(client_train_path) or not os.listdir(client_train_path): 
                # EXACTLY like sign-flip: skip clients with no data
                continue

            num_classes = len(os.listdir(client_train_path))

            client_dataset = datasets.ImageFolder(root=client_train_path, transform=transform)
            client_dataset.target_transform = make_target_transform_numeric(client_dataset.classes)
            local_loader = DataLoader(client_dataset, batch_size=cfg.CNN_BATCH_SIZE, shuffle=True)
            n_k = len(client_dataset)
            client_sample_sizes[client_id] = n_k

            client_type = "ACTUAL ATTACKER" if is_actual_attacker else ("POTENTIAL ATTACKER" if is_potential_attacker else "Benign")
            print(f"\n--- 🔄 Training Client: {client_id} (Type: {client_type}) ---")
            print(f"  - Training on {n_k} samples across {num_classes} classes.")

            local_model = FL_CNN().to(cfg.DEVICE)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = optim.SGD(local_model.parameters(), lr=cfg.CNN_LEARNING_RATE)
            criterion = nn.CrossEntropyLoss()

            final_epoch_losses, final_epoch_accuracies = [], []
            if not is_actual_attacker:
                # ✅ Benign client: do standard local training
                local_model.train()
                for epoch in range(cfg.LOCAL_CNN_EPOCHS):
                    epoch_loss, correct, total = 0, 0, 0
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
                    epoch_acc = (correct / total) * 100
                    print(f"    Epoch {epoch+1}/{cfg.LOCAL_CNN_EPOCHS}: Loss: {avg_epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

                final_epoch_losses.append(avg_epoch_loss)
                final_epoch_accuracies.append(epoch_acc)

                # Only benign clients contribute to the averages (match sign-flip behavior)
                client_local_metrics[client_id] = {'acc': np.mean(final_epoch_accuracies), 'loss': np.mean(final_epoch_losses)}

            else:
                # 🚨 FREE-RIDER: skip training and submit RANDOM weights/biases
                random_state = randomize_state_dict_like(local_model.state_dict())
                local_model.load_state_dict(random_state)
                # No entry in client_local_metrics (so attackers don't affect averages)

            # Save local submission
            hdf5_path = os.path.join(local_weights_dir, f"Round_{round_num}_Client_{client_id}_{'FreeRider' if is_actual_attacker else 'Benign'}.h5")
            with h5py.File(hdf5_path, "w") as hf:
                for name, param in local_model.state_dict().items():
                    hf.create_dataset(name, data=param.cpu().numpy())

            client_updates[client_id] = local_model.state_dict()

        # c. Server-side Detection Phase (unchanged)
        anomaly_scores, recon_errors = {}, {}
        bce_loss_fn, mse_loss_fn = torch.nn.BCELoss(), torch.nn.MSELoss()
        for client_id, local_weights in tqdm(client_updates.items(), desc="Server-Side Detection"):
            conv2_update = local_weights['conv2.weight'] - previous_global_weights['conv2.weight']
            
            # Alignment feature
            u = conv2_update.view(32, -1)
            r = benign_ref_dirs.view(32, -1)
            u = u / (u.norm(dim=1, keepdim=True).clamp_min(1e-12))
            r = r / (r.norm(dim=1, keepdim=True).clamp_min(1e-12))
            alignment_feature = (u * r).sum(dim=1)

            # Create the 7-feature graph
            graph = convert_update_to_graph(conv2_update, alignment_feature)
            graph = graph.to(cfg.DEVICE) 
            assert graph.x.size(1) == 7, f"Expected 7 node features, got {graph.x.size(1)}"
            with torch.no_grad():
                z = detector_model.encode(graph.x, graph.edge_index)
                recon_adj, recon_feat = detector_model.decode(z)
                true_adj = torch.zeros((graph.num_nodes, graph.num_nodes), device=cfg.DEVICE)
                true_adj[graph.edge_index[0], graph.edge_index[1]] = 1
                loss_a = bce_loss_fn(recon_adj, true_adj)
                loss_x = mse_loss_fn(recon_feat, graph.x)
                err_g = cfg.LAMBDA_ADJACENCY * loss_a + cfg.LAMBDA_FEATURES * loss_x
                recon_errors[client_id] = err_g.item()

        min_err = min(recon_errors.values()) if recon_errors else 0
        for cid, err in recon_errors.items():
            anomaly_scores[cid] = (1 + err) / (1 + min_err)

        # d. Secure Aggregation (same as before)
        aggregation_weights, TP, FP, TN, FN = {}, 0, 0, 0, 0

        if cfg.DEFENSE_METHOD == 'thresholding':
            # Paper-faithful: A_th = mean(anomaly scores)
            # Approve if A_k <= A_th
            # Weight for approved clients = n_k / n  (n = sum over ALL selected clients)
            A_th = float(np.mean(list(anomaly_scores.values()))) if anomaly_scores else 0.0
            approved = {cid for cid, score in anomaly_scores.items() if score <= A_th}

            n_total = sum(client_sample_sizes.values())
            for cid in client_updates.keys():
                if cid in approved and n_total > 0:
                    aggregation_weights[cid] = client_sample_sizes.get(cid, 0) / n_total
                else:
                    aggregation_weights[cid] = 0.0

        elif cfg.DEFENSE_METHOD == 'credit_scoring':
            denominator = sum(client_sample_sizes.get(cid, 0) * (anomaly_scores.get(cid, 1)**(-cfg.CREDIT_SCORE_L)) for cid in client_updates.keys())
            if not math.isfinite(denominator) or denominator <= 0:
                total_samples = sum(client_sample_sizes.values())
                for cid in client_updates.keys():
                    aggregation_weights[cid] = client_sample_sizes.get(cid, 0) / total_samples if total_samples > 0 else 0.0
            else:
                for cid in client_updates.keys():
                    n_k = client_sample_sizes.get(cid, 0)
                    A_k = anomaly_scores.get(cid, 1.0)
                    aggregation_weights[cid] = (n_k * (A_k**(-cfg.CREDIT_SCORE_L))) / denominator

        # detection metrics
        for cid in client_updates.keys():
            is_actual = client_is_actual_attacker.get(cid, False)
            is_approved = aggregation_weights.get(cid, 0) > 0
            if is_approved:
                if is_actual: FN += 1
                else: TN += 1
            else:
                if is_actual: TP += 1
                else: FP += 1

        print(f"\n--- 🛡️ Detection Round {round_num} ---")
        print(f" Attackers (free-riders): {actual_attackers_in_round if actual_attackers_in_round else 'None'}")
        print(f" Dropped: {[cid for cid, w in aggregation_weights.items() if w == 0]}")
        print(f" TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")

        # e. FedAvg
        avg_weights = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_model.state_dict().items()}
        total_weight = sum(aggregation_weights.values())
        if total_weight > 0:
            for cid, state_dict in client_updates.items():
                w = aggregation_weights.get(cid, 0.0)
                if w <= 0: continue
                for k in avg_weights.keys():
                    avg_weights[k] += state_dict[k] * w
            global_model.load_state_dict(avg_weights)
        else:
            print("⚠️ No models approved. Global model remains unchanged from previous round.")
            global_model.load_state_dict(previous_global_weights)

        # Save global model
        hdf5_global_path = os.path.join(global_weights_dir, f"Round_{round_num}_GlobalModel.h5")
        with h5py.File(hdf5_global_path, "w") as hf:
            for name, param in global_model.state_dict().items():
                hf.create_dataset(name, data=param.cpu().numpy())
        print("  💾 Global model weights saved to HDF5.")

        # f. Evaluate global model
        print("\n--- 🔬 Evaluating Global Model on Test Set ---")
        test_acc, prec, rec, f1 = evaluate_global_model(global_model, test_loader)
        # EXACTLY like sign-flip: averages over whatever is in client_local_metrics (benign only here)
        avg_local_acc = np.mean([v['acc'] for v in client_local_metrics.values()])
        avg_local_loss = np.mean([v['loss'] for v in client_local_metrics.values()])
        total_samples_processed = sum(client_sample_sizes.values())
        print(f"\n📊 Total Samples Processed in Round: {total_samples_processed}")
        print(f"   Breakdown: {' + '.join(map(str, client_sample_sizes.values()))} = {total_samples_processed}")
        print(f"\n🎯 Round {round_num}: Test Accuracy: {test_acc:.2f}% | Precision: {prec:.2f}% | Recall: {rec:.2f}% | F1 Score: {f1:.2f}%")

        # logs
        with open(global_perf_path, "a", newline="") as f:
            csv.writer(f).writerow([round_num, test_acc, prec, rec, f1])
        with open(local_perf_path, "a", newline="") as f:
            csv.writer(f).writerow([round_num, avg_local_acc, avg_local_loss])
        with open(detection_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            for cid in client_updates.keys():
                writer.writerow([round_num, cid, int(client_is_actual_attacker.get(cid, False)),
                                recon_errors.get(cid, 0.0), anomaly_scores.get(cid, 1.0),
                                "Approved" if aggregation_weights.get(cid, 0) > 0 else "Dropped",
                                aggregation_weights.get(cid, 0.0)])
        
        # --- Write the summary detector metrics log
        errs = np.array(list(recon_errors.values())) if recon_errors else np.array([0.0])
        scores = np.array(list(anomaly_scores.values())) if anomaly_scores else np.array([1.0])
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        with open(detector_metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([round_num, TP, FP, TN, FN, TPR, FPR,
                                    errs.mean(), np.percentile(errs, 50), np.percentile(errs, 90),
                                    scores.mean(), np.percentile(scores, 50), np.percentile(scores, 90)])
    print("🎉 Experiment complete.")


if __name__ == '__main__':
    main()

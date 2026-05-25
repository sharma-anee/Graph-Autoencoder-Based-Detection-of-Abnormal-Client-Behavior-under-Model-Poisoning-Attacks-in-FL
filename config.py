# fl_project/config.py

import torch

class Config:
    ## 🟢 STAGE 0: PREPARATION
    # --- Paths ---
    DATA_ROOT_PATH = 'E:/CODES/3500_client_img'
    TEST_DATA_PATH = 'E:/CODES/test_subset_20%'
    LOG_DIR_BASE = 'E:/CODES/GAE/Paper_Implementation_2/Logs_Final'

    # --- Experiment Phases ---
    R0_PRETRAIN_ROUNDS = 50  # Rounds to collect benign data and pre-train the CNN
    T_TOTAL_ROUNDS = 100    # Total rounds for the full experiment (pre-training + detection)

    # --- FL Client Settings ---
    CLIENTS_PER_ROUND = 40
    LOCAL_CNN_EPOCHS = 20
    CNN_BATCH_SIZE = 16
    CNN_LEARNING_RATE = 0.06

    # --- Attack Settings (for Stage 2) ---
    PERCENTAGE_ATTACKERS = 30
    ATTACK_PROBABILITY = 0.5  # Alpha: The probability that a designated attacker will actually attack
    #LABEL_FLIP_TARGET = 0  # -1 ⇒ auto-pick rarest class or any class index you want


    # --- GAE Detector Settings ---
    GRAPH_KNN = 8             # Number of neighbors for k-NN graph edge construction
    GAE_HIDDEN_CHANNELS = 32
    GAE_EMBEDDING_SIZE = 16
    GAE_EPOCHS = 300
    GAE_LEARNING_RATE = 0.001
    GAE_BATCH_SIZE = 32
    # Loss weights for the combined reconstruction loss
    LAMBDA_ADJACENCY = 0.0  # Weight for structural loss (BCE)
    LAMBDA_FEATURES = 1.0   # Weight for feature loss (MSE)

    # --- Defense Mechanism (for Stage 2) ---
    # Choose 'thresholding' or 'credit_scoring'
    DEFENSE_METHOD = 'credit_scoring'
    CREDIT_SCORE_L = 2.0      # Hyperparameter L for credit scoring

    # --- General ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()
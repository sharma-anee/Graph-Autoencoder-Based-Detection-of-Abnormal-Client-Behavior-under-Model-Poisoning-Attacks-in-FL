# fl_project/feature_extractor.py

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from config import cfg

def get_rich_features(filter_tensor):
    """Calculates a rich feature vector for a single filter."""
    # Flatten to avoid linalg.norm 3D error
    flat = filter_tensor.reshape(-1)

    mean = flat.mean()
    std = flat.std(unbiased=False)  # stable, matches default behavior for vectors
    # Elementwise norms on the vectorized filter
    l1_norm = torch.linalg.vector_norm(flat, ord=1)
    l2_norm = torch.linalg.vector_norm(flat, ord=2)
    
    # Sign ratio: proportion of positive weights
    sign_ratio = (flat > 0).float().mean()
    
    # Top-k absolute mean
    k = 5
    top_k_vals, _ = torch.topk(flat.abs(), k)
    top_k_mean = top_k_vals.mean()
    
    # Combine all features into a single tensor
    return torch.stack([mean, std, l1_norm, l2_norm, sign_ratio, top_k_mean])

@torch.no_grad()
def convert_update_to_graph(conv2_weights, alignment_feature=None):
    """
    Converts a conv2 weight tensor [32, 32, 3, 3] into a rich graph.
    Optionally includes an alignment feature for the detection phase.
    """
    num_filters = conv2_weights.size(0)
    
    # --- 1. Create Nodes and Features ---
    node_features = []
    for i in range(num_filters):
        filter_tensor = conv2_weights[i]
        features = get_rich_features(filter_tensor)
        
        # In the detection phase, add the alignment feature
        if alignment_feature is not None:
            # Expect a 1D tensor of length = num_filters, on same device
            assert alignment_feature.dim() == 1 and alignment_feature.numel() == num_filters, \
                f"alignment_feature must be shape [{num_filters}]"
            if alignment_feature.device != filter_tensor.device:
                alignment_feature = alignment_feature.to(filter_tensor.device)
            features = torch.cat([features, alignment_feature[i].view(1)])
            
        node_features.append(features)
    x = torch.stack(node_features)

    # --- 2. Create Edges via Cosine-kNN ---
    flattened_weights = conv2_weights.view(num_filters, -1)
    sim_matrix = F.cosine_similarity(flattened_weights.unsqueeze(1), flattened_weights.unsqueeze(0), dim=-1)
    sim_matrix.fill_diagonal_(-1.0)  # mask self before topk
    _, top_k_indices = torch.topk(sim_matrix, k=cfg.GRAPH_KNN, dim=1)
    
    edge_list = []
    for i in range(num_filters):
        for j in top_k_indices[i]:
            if i != j.item():
                edge_list.append((i, j.item()))
    dev = conv2_weights.device
    edge_index = torch.tensor(edge_list, dtype=torch.long, device=dev).t().contiguous()
    edge_index = to_undirected(edge_index, num_nodes=num_filters)
    
    return Data(x=x, edge_index=edge_index)
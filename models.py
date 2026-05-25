# fl_project/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU
from torch_geometric.nn import GCNConv

class FL_CNN(nn.Module):
    """The global CNN model trained via Federated Learning."""
    def __init__(self, num_classes=62):
        super(FL_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GraphAutoencoder(nn.Module):
    """The GAE detector model trained on benign graph representations."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphAutoencoder, self).__init__()
        # Encoder compresses the graph into a low-dimensional embedding 'z'
        self.encoder_conv1 = GCNConv(in_channels, hidden_channels)
        self.encoder_conv2 = GCNConv(hidden_channels, out_channels)

        # Decoders attempt to reconstruct the original graph from the embedding 'z'
        # 1. Adjacency Decoder: Predicts connections via inner product
        self.adj_decoder = lambda z: torch.sigmoid((z @ z.t()))
        
        # 2. Feature Decoder: Predicts node features via a small MLP
        self.feature_decoder = nn.Sequential(
            Linear(out_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, in_channels)
        )

    def encode(self, x, edge_index):
        """Encodes the graph into a latent embedding."""
        x = F.relu(self.encoder_conv1(x, edge_index))
        z = self.encoder_conv2(x, edge_index)
        return z

    def decode(self, z):
        """Decodes the latent embedding back into a graph."""
        reconstructed_adj = self.adj_decoder(z)
        reconstructed_features = self.feature_decoder(z)
        return reconstructed_adj, reconstructed_features
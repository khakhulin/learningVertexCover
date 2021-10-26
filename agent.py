import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class Agent(nn.Module):
    def __init__(self, ndim, hidden_dim=8, device=None):
        super(Agent, self).__init__()
        self.conv_layers = nn.ModuleList([
            GCNConv(ndim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim)
            ])

        self.policy = nn.Linear(hidden_dim, 1)
        self.value = nn.Linear(hidden_dim, 1)
        self.device = device

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        for layer in self.conv_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        mean_val = global_mean_pool(x, torch.zeros(x.size()[0], device=self.device, dtype=torch.long))
        proba = self.policy(x)
        value = self.value(mean_val)
        return proba, value
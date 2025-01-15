import torch.nn as nn
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        return x
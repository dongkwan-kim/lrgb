import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class MLPLayer(nn.Module):
    """MLP Baseline that applies on features 
    """

    def __init__(self, dim_in, dim_out, dropout, residual):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual

        self.model = nn.Sequential(*[pyg_nn.Linear(dim_in, dim_in), nn.ReLU()])

    def forward(self, batch):
        x_in = batch.x

        batch.x = self.model(batch.x)

        batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)

        if self.residual:
            batch.x = x_in + batch.x  # residual connection

        return batch

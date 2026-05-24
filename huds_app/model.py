import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.linear1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return residual + out


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, residual_blocks, dropout=0.1):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.residual_blocks = nn.ModuleList(
            ResidualBlock(hidden_dim, dropout) for _ in range(residual_blocks)
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_features=False):
        features = F.relu(self.input_layer(x))
        for block in self.residual_blocks:
            features = block(features)

        output = self.output_layer(features)
        if return_features:
            return output, features
        return output


def build_model(config):
    input_dim = len(config.variables)
    output_dim = len(config.model.output_names)
    hidden_dim = config.model.hidden_dim
    residual_blocks = config.model.residual_blocks
    dropout = config.model.dropout
    return ResidualMLP(input_dim, output_dim, hidden_dim, residual_blocks, dropout)

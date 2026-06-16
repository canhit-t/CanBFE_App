"""Reusable multi-layer perceptron building block.

Constructs a stack of Linear → BatchNorm1d → ReLU → Dropout layers,
followed by a final Linear projection to ``out_dim``.

Example
-------
::

    mlp = MLP(in_dim=864, hidden_dims=[512, 256], out_dim=1, dropout=0.2)
    y   = mlp(x)   # x: [B, 864]
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Fully-connected network with BN + ReLU + Dropout hidden layers.

    Parameters
    ----------
    in_dim:
        Input feature dimension.
    hidden_dims:
        Sizes of each hidden layer.  An empty list means a single linear
        projection from ``in_dim`` → ``out_dim``.
    out_dim:
        Output dimension (1 for scalar ΔG prediction).
    dropout:
        Dropout probability applied after each hidden activation.
    use_bn:
        Whether to include BatchNorm1d after each Linear (default True).
    """

    def __init__(
        self,
        in_dim:      int,
        hidden_dims: Sequence[int],
        out_dim:     int,
        dropout:     float = 0.2,
        use_bn:      bool  = True,
    ) -> None:
        super().__init__()
        dims   = [in_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_bn:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, in_dim] → [B, out_dim]
        return self.net(x)

"""Manual message-passing GNN operating on plain torch graph dicts.

Does NOT import torch_geometric.  All operations are expressed in terms
of plain ``torch`` scatter/index operations, safe on Windows with any
GPU generation.

Architecture
------------
Two edge-conditioned GraphConv layers with sum-aggregation:

    h^(0) = Linear_in(x)                         [N, hidden_dim]
    h^(l+1)[v] = ReLU( BN(
                    W_self * h^(l)[v]
                  + Σ_{u∈N(v)} ( W_msg * h^(l)[u]  +  W_edge * RBF(d_{uv}) )
                  + b ))

    pool = mean( h^(L)[all nodes] )               [1, hidden_dim]

RBF distance encoding (SchNet-style):
    RBF_k(d) = exp( -(d - μ_k)^2 / (2σ^2) )
    μ_k evenly spaced in [d_min, d_max],  σ = spacing

For batched graphs the pool mean is masked by node count.
The module also returns per-node embeddings for cross-attention consumers.

Graph dict keys consumed:
    x          : FloatTensor  [N, 10]
    edge_index : LongTensor   [2, E]
    edge_attr  : FloatTensor  [E, 5]   column 0 = distance (Å)
    (node_type, n_ligand_atoms, n_pocket_residues available but not required)
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# RBF distance encoder
# ---------------------------------------------------------------------------

class RBFDistanceEncoder(nn.Module):
    """Encode scalar pairwise distances into a fixed-size RBF feature vector.

    Parameters
    ----------
    n_rbf:
        Number of Gaussian basis functions.
    d_min, d_max:
        Range of distances (Å) to cover with the basis centers.
    trainable:
        If True, μ and σ are learnable parameters.
    """

    def __init__(
        self,
        n_rbf:     int   = 16,
        d_min:     float = 0.0,
        d_max:     float = 10.0,
        trainable: bool  = False,
    ) -> None:
        super().__init__()
        centers = torch.linspace(d_min, d_max, n_rbf)        # [n_rbf]
        sigma   = torch.full((n_rbf,), (d_max - d_min) / n_rbf)
        if trainable:
            self.centers = nn.Parameter(centers)
            self.sigma   = nn.Parameter(sigma)
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("sigma",   sigma)
        self.n_rbf = n_rbf

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        dist : FloatTensor [E] — per-edge scalar distances

        Returns
        -------
        FloatTensor [E, n_rbf]
        """
        d = dist.unsqueeze(-1)                                # [E, 1]
        return torch.exp(-((d - self.centers) ** 2) / (2 * self.sigma ** 2 + 1e-8))


# ---------------------------------------------------------------------------
# Message-passing layer
# ---------------------------------------------------------------------------

class GraphConvLayer(nn.Module):
    """Edge-conditioned sum-aggregation message-passing layer.

    h'[v] = ReLU( BN( W_self·h[v]
                    + Σ_{u→v}( W_msg·h[u]  +  W_edge·rbf(d_{uv}) )
                    + b ) )

    If edge_dim == 0 the layer degrades to the original topology-only MPNN.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 0) -> None:
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_msg  = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_edge = nn.Linear(edge_dim, out_dim, bias=False) if edge_dim > 0 else None
        self.act      = nn.ReLU(inplace=True)
        self.bn       = nn.BatchNorm1d(out_dim)

    def forward(
        self,
        h:          torch.Tensor,                   # [N, in_dim]
        edge_index: torch.Tensor,                   # [2, E]
        N:          int,
        edge_rbf:   torch.Tensor | None = None,     # [E, edge_dim]
    ) -> torch.Tensor:                              # [N, out_dim]
        src, dst = edge_index[0], edge_index[1]     # [E] each

        # Node-based message
        msg = self.lin_msg(h[src])                  # [E, out_dim]

        # Add distance contribution to each message
        if self.lin_edge is not None and edge_rbf is not None:
            msg = msg + self.lin_edge(edge_rbf)     # [E, out_dim]

        agg = torch.zeros(N, msg.shape[1], dtype=h.dtype, device=h.device)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)  # [N, out_dim]

        out = self.lin_self(h) + agg                # [N, out_dim]
        out = self.bn(out)
        out = self.act(out)
        return out


class ManualGNN(nn.Module):
    """Distance-aware two-layer GNN with mean global pooling.

    Parameters
    ----------
    in_dim:
        Input node-feature dimension (10 for the complex graph).
    hidden_dim:
        Width of each GNN layer.
    out_dim:
        Dimension of the pooled graph embedding returned by ``forward``.
    n_layers:
        Number of message-passing layers (default 2).
    dropout:
        Dropout applied before the final projection (default 0.1).
    rbf_dim:
        Number of RBF basis functions used to encode edge distances.
        Set to 0 to disable distance encoding (topology-only, legacy).
    rbf_d_max:
        Maximum distance (Å) covered by RBF centers (default 10.0).
    """

    def __init__(
        self,
        in_dim:     int   = 10,
        hidden_dim: int   = 128,
        out_dim:    int   = 128,
        n_layers:   int   = 2,
        dropout:    float = 0.1,
        rbf_dim:    int   = 16,
        rbf_d_max:  float = 10.0,
    ) -> None:
        super().__init__()

        self.rbf_dim = rbf_dim

        # RBF distance encoder (shared across all layers)
        self.rbf_encoder = (
            RBFDistanceEncoder(n_rbf=rbf_dim, d_min=0.0, d_max=rbf_d_max)
            if rbf_dim > 0 else None
        )

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.input_bn   = nn.BatchNorm1d(hidden_dim)
        self.input_act  = nn.ReLU(inplace=True)

        # Message-passing layers — edge_dim=rbf_dim makes them distance-aware
        self.conv_layers = nn.ModuleList(
            GraphConvLayer(hidden_dim, hidden_dim, edge_dim=rbf_dim)
            for _ in range(n_layers)
        )

        # Output projection
        self.dropout    = nn.Dropout(p=dropout)
        self.output_lin = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x:          torch.Tensor,                    # [N, in_dim]
        edge_index: torch.Tensor,                    # [2, E]
        batch:      torch.Tensor | None = None,      # [N] — node-to-graph mapping
        edge_attr:  torch.Tensor | None = None,      # [E, ≥1]  col 0 = distance (Å)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run message passing and return (graph_embedding, node_embeddings).

        Parameters
        ----------
        x:
            Node feature matrix ``[N, in_dim]``.
        edge_index:
            COO edge index ``[2, E]``.
        batch:
            Integer tensor ``[N]`` mapping each node to its graph index
            (0-based).  ``None`` means all nodes belong to graph 0.
        edge_attr:
            Edge feature matrix ``[E, ≥1]``.  Column 0 is the Euclidean
            distance in Å (as stored by graph_complex.py).  Remaining
            columns (bond types) are ignored here.

        Returns
        -------
        graph_emb : FloatTensor ``[B, out_dim]``
            Mean-pooled graph representation, one row per graph.
        node_emb  : FloatTensor ``[N, hidden_dim]``
            Per-node representations after the last GNN layer (before the
            output projection), useful for cross-attention.
        """
        N = x.shape[0]

        # --- encode distances into RBF features ---
        edge_rbf: torch.Tensor | None = None
        if self.rbf_encoder is not None and edge_attr is not None:
            dist     = edge_attr[:, 0].clamp(min=0.0)    # [E]
            edge_rbf = self.rbf_encoder(dist)             # [E, rbf_dim]

        # --- input projection ---
        h = self.input_act(self.input_bn(self.input_proj(x)))  # [N, hidden]

        # --- message passing ---
        for conv in self.conv_layers:
            h = conv(h, edge_index, N, edge_rbf)          # [N, hidden]

        node_emb = h  # save for cross-attention consumers  [N, hidden]

        # --- pooling ---
        if batch is None:
            # Single graph: mean over all nodes
            pooled = h.mean(dim=0, keepdim=True)           # [1, hidden]
        else:
            B      = int(batch.max().item()) + 1
            pooled = torch.zeros(B, h.shape[1], dtype=h.dtype, device=h.device)
            count  = torch.zeros(B, 1, dtype=h.dtype, device=h.device)
            idx    = batch.unsqueeze(1).expand_as(h)
            pooled.scatter_add_(0, idx, h)
            count.scatter_add_(0, batch.unsqueeze(1), torch.ones(N, 1, dtype=h.dtype, device=h.device))
            pooled = pooled / count.clamp(min=1)           # [B, hidden]

        graph_emb = self.output_lin(self.dropout(pooled))  # [B, out_dim]
        return graph_emb, node_emb

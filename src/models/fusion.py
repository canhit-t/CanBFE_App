"""Affinity prediction model architectures for Phase 3.

All models predict a scalar ΔG (kcal mol⁻¹) from cached feature inputs.
No ESM2 / ChemBERTa fine-tuning occurs here.

Models
------
SequenceOnlyModel
    protein_global_emb [480] + ligand_smiles_emb [384] → cat [864] → MLP → ΔG

GraphOnlyModel
    graph dict → ManualGNN → pooled [gnn_out_dim] → MLP → ΔG

ConcatFusionModel
    protein_global_emb + ligand_smiles_emb + graph_pool → MLP → ΔG

CrossAttentionFusionModel
    Global tokens : project(prot_emb), project(lig_emb)       [2, H]
    Local  tokens : GNN node embeddings                        [N, H]
    CrossAttentionBlock → concat learned-pool(local, global)  [2H] → MLP → ΔG

AllConcatFusionModel
    global_prot_emb [480] + pocket_prot_emb [480] + lig_emb [384] + graph_pool [128]
    → cat [1472] → MLP → ΔG

AllCrossAttentionFusionModel
    Global tokens : project(global_prot), project(pocket_prot), project(lig_emb)  [3, H]
    Local  tokens : GNN node embeddings                                            [N, H]
    CrossAttentionBlock → concat learned-pool(local, global)  [2H] → MLP → ΔG

Input tensor names
------------------
prot_emb   : FloatTensor [B, 480]   protein global embedding
pocket_emb : FloatTensor [B, 480]   pocket-contextual ESM2 embedding (all-features models)
lig_emb    : FloatTensor [B, 384]   ligand SMILES embedding
x          : FloatTensor [N, 10]    graph node features  (N = sum of batch nodes)
edge_index : LongTensor [2, E]      COO edge indices
batch      : LongTensor [N]         node-to-graph mapping (0..B-1)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.manual_gnn import ManualGNN
from src.models.mlp import MLP

# Default embedding sizes — match cached feature dimensions
PROT_EMB_DIM  = 480   # ESM2 t12_35M
LIG_EMB_DIM   = 384   # ChemBERTa 77M
GRAPH_NODE_DIM = 10   # NUM_NODE_FEATURES from graph_complex


# ---------------------------------------------------------------------------
# 1. Sequence-only
# ---------------------------------------------------------------------------

class SequenceOnlyModel(nn.Module):
    """Predict ΔG from protein and ligand global embeddings only.

    Data flow:
        prot_emb [B,480] + lig_emb [B,384] → cat [B,864] → MLP → [B,1]
    """

    def __init__(
        self,
        prot_dim:    int            = PROT_EMB_DIM,
        lig_dim:     int            = LIG_EMB_DIM,
        hidden_dims: list[int]      = None,
        dropout:     float          = 0.2,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [512, 256]
        self.mlp = MLP(
            in_dim      = prot_dim + lig_dim,
            hidden_dims = hidden_dims,
            out_dim     = 1,
            dropout     = dropout,
        )

    def encode(
        self,
        prot_emb: torch.Tensor,   # [B, prot_dim]
        lig_emb:  torch.Tensor,   # [B, lig_dim]
        **_,
    ) -> torch.Tensor:            # [B, prot_dim + lig_dim]
        """Return the pre-MLP fused representation."""
        return torch.cat([prot_emb, lig_emb], dim=-1)

    def forward(
        self,
        prot_emb: torch.Tensor,   # [B, prot_dim]
        lig_emb:  torch.Tensor,   # [B, lig_dim]
        **_,
    ) -> torch.Tensor:            # [B, 1]
        return self.mlp(self.encode(prot_emb, lig_emb))


# ---------------------------------------------------------------------------
# 2. Graph-only
# ---------------------------------------------------------------------------

class GraphOnlyModel(nn.Module):
    """Predict ΔG from the structural graph alone.

    Data flow:
        graph (x, edge_index, batch) → ManualGNN → [B, gnn_out] → MLP → [B,1]
    """

    def __init__(
        self,
        node_dim:    int       = GRAPH_NODE_DIM,
        gnn_hidden:  int       = 128,
        gnn_out:     int       = 128,
        n_layers:    int       = 2,
        hidden_dims: list[int] = None,
        dropout:     float     = 0.2,
        rbf_dim:     int       = 16,   # 0 = topology-only (no distance encoding)
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        self.gnn = ManualGNN(
            in_dim     = node_dim,
            hidden_dim = gnn_hidden,
            out_dim    = gnn_out,
            n_layers   = n_layers,
            dropout    = dropout,
            rbf_dim    = rbf_dim,
        )
        self.mlp = MLP(
            in_dim      = gnn_out,
            hidden_dims = hidden_dims,
            out_dim     = 1,
            dropout     = dropout,
        )

    def encode(
        self,
        x:          torch.Tensor,                # [N, node_dim]
        edge_index: torch.Tensor,                # [2, E]
        batch:      torch.Tensor | None = None,  # [N]
        edge_attr:  torch.Tensor | None = None,  # [E, 5]
        **_,
    ) -> torch.Tensor:                           # [B, gnn_out]
        """Return the pre-MLP pooled graph embedding."""
        graph_emb, _ = self.gnn(x, edge_index, batch, edge_attr)
        return graph_emb

    def forward(
        self,
        x:          torch.Tensor,                # [N, node_dim]
        edge_index: torch.Tensor,                # [2, E]
        batch:      torch.Tensor | None = None,  # [N]
        edge_attr:  torch.Tensor | None = None,  # [E, 5]  col 0 = distance (Å)
        **_,
    ) -> torch.Tensor:                           # [B, 1]
        graph_emb, _ = self.gnn(x, edge_index, batch, edge_attr)
        return self.mlp(graph_emb)


# ---------------------------------------------------------------------------
# 3. Concat fusion
# ---------------------------------------------------------------------------

class ConcatFusionModel(nn.Module):
    """Predict ΔG from sequences + graph via concatenation.

    Data flow:
        prot_emb [B, P] + lig_emb [B, L] + graph_pool [B, G]
            → cat [B, P+L+G] → MLP → [B,1]
    """

    def __init__(
        self,
        prot_dim:    int       = PROT_EMB_DIM,
        lig_dim:     int       = LIG_EMB_DIM,
        node_dim:    int       = GRAPH_NODE_DIM,
        gnn_hidden:  int       = 128,
        gnn_out:     int       = 128,
        n_layers:    int       = 2,
        hidden_dims: list[int] = None,
        dropout:     float     = 0.2,
        rbf_dim:     int       = 16,   # 0 = topology-only (no distance encoding)
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [512, 256]
        self.gnn = ManualGNN(
            in_dim     = node_dim,
            hidden_dim = gnn_hidden,
            out_dim    = gnn_out,
            n_layers   = n_layers,
            dropout    = dropout,
            rbf_dim    = rbf_dim,
        )
        self.mlp = MLP(
            in_dim      = prot_dim + lig_dim + gnn_out,
            hidden_dims = hidden_dims,
            out_dim     = 1,
            dropout     = dropout,
        )

    def encode(
        self,
        prot_emb:   torch.Tensor,                # [B, P]
        lig_emb:    torch.Tensor,                # [B, L]
        x:          torch.Tensor,                # [N, node_dim]
        edge_index: torch.Tensor,                # [2, E]
        batch:      torch.Tensor | None = None,  # [N]
        edge_attr:  torch.Tensor | None = None,  # [E, 5]
        **_,
    ) -> torch.Tensor:                           # [B, P+L+gnn_out]
        """Return the pre-MLP fused representation."""
        graph_emb, _ = self.gnn(x, edge_index, batch, edge_attr)
        return torch.cat([prot_emb, lig_emb, graph_emb], dim=-1)

    def forward(
        self,
        prot_emb:   torch.Tensor,                # [B, P]
        lig_emb:    torch.Tensor,                # [B, L]
        x:          torch.Tensor,                # [N, node_dim]
        edge_index: torch.Tensor,                # [2, E]
        batch:      torch.Tensor | None = None,  # [N]
        edge_attr:  torch.Tensor | None = None,  # [E, 5]  col 0 = distance (Å)
        **_,
    ) -> torch.Tensor:                           # [B, 1]
        return self.mlp(self.encode(prot_emb, lig_emb, x, edge_index, batch, edge_attr))


# ---------------------------------------------------------------------------
# 4. Cross-attention fusion
# ---------------------------------------------------------------------------

class CrossAttentionFusionModel(nn.Module):
    """Predict ΔG using bidirectional cross-attention between
    global sequence tokens and local graph node embeddings.

    Data flow
    ---------
    1. Project prot_emb and lig_emb into a shared hidden space H:
           global_tokens [B, 2, H]

    2. GNN over the structural graph → node_emb [N, H]  (one big batch)

    3. For each graph in the batch, gather its node embeddings and
       run two cross-attention passes with optional residual + LayerNorm:

       a. Local → Global:
          Q = node_emb [n_i, H],  K = V = global_tokens [2, H]
          attended_local  = LayerNorm(node_emb + attn_output)   [n_i, H]

       b. Global → Local:
          Q = global_tokens [2, H],  K = V = node_emb [n_i, H]
          attended_global = LayerNorm(global_tokens + attn_output)  [2, H]

    4. Pool (learned attention pooling):
           local_scores  = Linear(attended_local)  → softmax → weighted sum  [H]
           global_scores = Linear(attended_global) → softmax → weighted sum  [H]
           fused = cat([local_pool, global_pool])  [2H]

    5. MLP → ΔG [1]

    Config flags (all default True for best performance; set False for ablations)
    ---------------------------------------------------------------------------
    use_attention_residuals     : residual + LayerNorm around attention outputs
    use_attention_pooling       : learned softmax pooling over local nodes
    use_global_attention_pooling: learned softmax pooling over global tokens

    Optional diagnostics
    --------------------
    Call forward(..., return_diagnostics=True) to get a (prediction, diag) tuple
    where diag contains mean_local_attn_entropy and mean_global_attn_entropy.

    Note: cross-attention is computed sample-by-sample in the batch loop
    to handle variable-length node sequences.
    """

    def __init__(
        self,
        prot_dim:   int       = PROT_EMB_DIM,
        lig_dim:    int       = LIG_EMB_DIM,
        node_dim:   int       = GRAPH_NODE_DIM,
        hidden_dim: int       = 128,
        n_layers:   int       = 2,
        n_heads:    int       = 4,
        hidden_dims: list[int] = None,
        dropout:    float     = 0.2,
        use_attention_residuals:      bool = True,
        use_attention_pooling:        bool = True,
        use_global_attention_pooling: bool = True,
        rbf_dim:                      int  = 16,   # 0 = topology-only (no distance encoding)
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]

        self.hidden_dim = hidden_dim
        self.use_attention_residuals      = use_attention_residuals
        self.use_attention_pooling        = use_attention_pooling
        self.use_global_attention_pooling = use_global_attention_pooling

        assert hidden_dim % n_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
        )

        # Project sequence embeddings into hidden space
        self.prot_proj = nn.Linear(prot_dim,  hidden_dim)
        self.lig_proj  = nn.Linear(lig_dim,   hidden_dim)

        # GNN — output dim == hidden_dim so node embeddings are in the same space
        self.gnn = ManualGNN(
            in_dim     = node_dim,
            hidden_dim = hidden_dim,
            out_dim    = hidden_dim,
            n_layers   = n_layers,
            dropout    = dropout,
            rbf_dim    = rbf_dim,
        )
        self.node_proj = nn.Linear(hidden_dim, hidden_dim)

        # Cross-attention layers
        self.attn_local_to_global = nn.MultiheadAttention(
            embed_dim   = hidden_dim,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.attn_global_to_local = nn.MultiheadAttention(
            embed_dim   = hidden_dim,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,
        )

        # A1: residual LayerNorms
        self.local_layernorm  = nn.LayerNorm(hidden_dim)
        self.global_layernorm = nn.LayerNorm(hidden_dim)

        # A2/A3: learned attention pooling projections
        self.local_pool_proj  = nn.Linear(hidden_dim, 1)
        self.global_pool_proj = nn.Linear(hidden_dim, 1)

        # MLP head: 2 × hidden_dim input
        self.mlp = MLP(
            in_dim      = 2 * hidden_dim,
            hidden_dims = hidden_dims,
            out_dim     = 1,
            dropout     = dropout,
        )

    def _build_fused(
        self,
        prot_emb:   torch.Tensor,
        lig_emb:    torch.Tensor,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        batch:      torch.Tensor | None,
        edge_attr:  torch.Tensor | None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Shared internals: returns fused [B, 2H] (or tuple with diag)."""
        B = prot_emb.shape[0]
        device = prot_emb.device

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)

        _, node_emb = self.gnn(x, edge_index, batch, edge_attr)   # [N, hidden]
        node_emb    = self.node_proj(node_emb)                     # [N, hidden]

        p = self.prot_proj(prot_emb).unsqueeze(1)      # [B, 1, H]
        l = self.lig_proj(lig_emb).unsqueeze(1)        # [B, 1, H]
        global_tokens = torch.cat([p, l], dim=1)       # [B, 2, H]

        local_pools:  list[torch.Tensor] = []
        global_pools: list[torch.Tensor] = []
        local_entropies:  list[float] = []
        global_entropies: list[float] = []

        for i in range(B):
            mask = (batch == i)
            ni   = node_emb[mask]                      # [n_i, H]
            gi   = global_tokens[i].unsqueeze(0)       # [1, 2, H]

            attn_local_out, local_w = self.attn_local_to_global(
                ni.unsqueeze(0), gi, gi,
                need_weights=return_diagnostics,
            )
            attn_local_out = attn_local_out.squeeze(0)
            attended_local = (
                self.local_layernorm(ni + attn_local_out)
                if self.use_attention_residuals else attn_local_out
            )

            attn_global_out, global_w = self.attn_global_to_local(
                gi, ni.unsqueeze(0), ni.unsqueeze(0),
                need_weights=return_diagnostics,
            )
            attn_global_out = attn_global_out.squeeze(0)
            attended_global = (
                self.global_layernorm(gi.squeeze(0) + attn_global_out)
                if self.use_attention_residuals else attn_global_out
            )

            if self.use_attention_pooling:
                lp_w = torch.softmax(self.local_pool_proj(attended_local), dim=0)
                local_pool_i = (lp_w * attended_local).sum(dim=0)
            else:
                local_pool_i = attended_local.mean(dim=0)

            if self.use_global_attention_pooling:
                gp_w = torch.softmax(self.global_pool_proj(attended_global), dim=0)
                global_pool_i = (gp_w * attended_global).sum(dim=0)
            else:
                global_pool_i = attended_global.mean(dim=0)

            local_pools.append(local_pool_i)
            global_pools.append(global_pool_i)

            if return_diagnostics and local_w is not None:
                lw = local_w.squeeze(0).clamp(min=1e-9)
                local_entropies.append(-(lw * lw.log()).sum(dim=-1).mean().item())
                gw = global_w.squeeze(0).clamp(min=1e-9)
                global_entropies.append(-(gw * gw.log()).sum(dim=-1).mean().item())

        local_pool  = torch.stack(local_pools,  dim=0)  # [B, H]
        global_pool = torch.stack(global_pools, dim=0)  # [B, H]
        fused = torch.cat([local_pool, global_pool], dim=-1)  # [B, 2H]

        if return_diagnostics:
            diag: dict = {}
            if local_entropies:
                n = len(local_entropies)
                diag["mean_local_attn_entropy"]  = sum(local_entropies)  / n
                diag["mean_global_attn_entropy"] = sum(global_entropies) / n
            return fused, diag
        return fused

    def encode(
        self,
        prot_emb:   torch.Tensor,
        lig_emb:    torch.Tensor,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        batch:      torch.Tensor | None = None,
        edge_attr:  torch.Tensor | None = None,
        **_,
    ) -> torch.Tensor:                           # [B, 2*hidden_dim]
        """Return the pre-MLP fused representation."""
        return self._build_fused(prot_emb, lig_emb, x, edge_index, batch, edge_attr)

    def forward(
        self,
        prot_emb:   torch.Tensor,                # [B, prot_dim]
        lig_emb:    torch.Tensor,                # [B, lig_dim]
        x:          torch.Tensor,                # [N, node_dim]
        edge_index: torch.Tensor,                # [2, E]
        batch:      torch.Tensor | None = None,  # [N]
        edge_attr:  torch.Tensor | None = None,  # [E, 5]  col 0 = distance (Å)
        return_diagnostics: bool = False,
        **_,
    ) -> torch.Tensor:                           # [B, 1]  (or tuple when return_diagnostics)
        result = self._build_fused(
            prot_emb, lig_emb, x, edge_index, batch, edge_attr,
            return_diagnostics=return_diagnostics,
        )

        if return_diagnostics:
            fused, diag = result
            out = self.mlp(fused)
            return out, diag
        return self.mlp(result)


# ---------------------------------------------------------------------------
# 5. All-concat fusion
# ---------------------------------------------------------------------------

class AllConcatFusionModel(nn.Module):
    """Predict ΔG from global protein, pocket protein, ligand, and structural graph.

    Data flow:
        global_emb  [B, P]  (full-protein ESM2)
        pocket_emb  [B, K]  (pocket-contextual ESM2)
        lig_emb     [B, L]
        graph_pool  [B, G]  ← ManualGNN
            → cat [B, P+K+L+G] → MLP → [B,1]
    """

    def __init__(
        self,
        prot_dim:    int       = PROT_EMB_DIM,   # global ESM2 dim
        pocket_dim:  int       = PROT_EMB_DIM,   # pocket ESM2 dim (same encoder)
        lig_dim:     int       = LIG_EMB_DIM,
        node_dim:    int       = GRAPH_NODE_DIM,
        gnn_hidden:  int       = 128,
        gnn_out:     int       = 128,
        n_layers:    int       = 2,
        hidden_dims: list[int] = None,
        dropout:     float     = 0.2,
        rbf_dim:     int       = 16,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [512, 256]
        self.gnn = ManualGNN(
            in_dim     = node_dim,
            hidden_dim = gnn_hidden,
            out_dim    = gnn_out,
            n_layers   = n_layers,
            dropout    = dropout,
            rbf_dim    = rbf_dim,
        )
        self.mlp = MLP(
            in_dim      = prot_dim + pocket_dim + lig_dim + gnn_out,
            hidden_dims = hidden_dims,
            out_dim     = 1,
            dropout     = dropout,
        )

    def forward(
        self,
        prot_emb:   torch.Tensor,                # [B, P]   global ESM2
        pocket_emb: torch.Tensor,                # [B, K]   pocket ESM2
        lig_emb:    torch.Tensor,                # [B, L]
        x:          torch.Tensor,                # [N, node_dim]
        edge_index: torch.Tensor,                # [2, E]
        batch:      torch.Tensor | None = None,  # [N]
        edge_attr:  torch.Tensor | None = None,  # [E, 5]  col 0 = distance (Å)
        **_,
    ) -> torch.Tensor:                           # [B, 1]
        graph_emb, _ = self.gnn(x, edge_index, batch, edge_attr)
        fused = torch.cat([prot_emb, pocket_emb, lig_emb, graph_emb], dim=-1)
        return self.mlp(fused)

    def encode(
        self,
        prot_emb:   torch.Tensor,
        pocket_emb: torch.Tensor,
        lig_emb:    torch.Tensor,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        batch:      torch.Tensor | None = None,
        edge_attr:  torch.Tensor | None = None,
        **_,
    ) -> torch.Tensor:                           # [B, prot_dim+pocket_dim+lig_dim+gnn_out]
        """Return the pre-MLP fused representation."""
        graph_emb, _ = self.gnn(x, edge_index, batch, edge_attr)
        return torch.cat([prot_emb, pocket_emb, lig_emb, graph_emb], dim=-1)


# ---------------------------------------------------------------------------
# 6. All-cross-attention fusion
# ---------------------------------------------------------------------------

class AllCrossAttentionFusionModel(nn.Module):
    """Predict ΔG using cross-attention over global protein, pocket protein, ligand, and graph.

    Data flow
    ---------
    1. Project global_prot, pocket_prot, lig_emb → shared hidden H:
           global_tokens [B, 3, H]  (one token per sequence source)

    2. GNN over the structural graph → node_emb [N, H]  (one big batch)

    3. For each graph in the batch, bidirectional cross-attention:

       a. Local → Global:
          Q = node_emb [n_i, H],  K = V = global_tokens [3, H]
          attended_local  = LayerNorm(node_emb + attn_output)   [n_i, H]

       b. Global → Local:
          Q = global_tokens [3, H],  K = V = node_emb [n_i, H]
          attended_global = LayerNorm(global_tokens + attn_output)  [3, H]

    4. Learned attention pooling:
           local_pool  = weighted sum of attended_local   [H]
           global_pool = weighted sum of attended_global  [H]
           fused = cat([local_pool, global_pool])  [2H] → MLP → ΔG

    The MLP head has the same input size (2H) as CrossAttentionFusionModel;
    only the global token sequence grows from 2 to 3 tokens.
    """

    def __init__(
        self,
        prot_dim:    int       = PROT_EMB_DIM,
        pocket_dim:  int       = PROT_EMB_DIM,
        lig_dim:     int       = LIG_EMB_DIM,
        node_dim:    int       = GRAPH_NODE_DIM,
        hidden_dim:  int       = 128,
        n_layers:    int       = 2,
        n_heads:     int       = 4,
        hidden_dims: list[int] = None,
        dropout:     float     = 0.2,
        use_attention_residuals:      bool = True,
        use_attention_pooling:        bool = True,
        use_global_attention_pooling: bool = True,
        rbf_dim:                      int  = 16,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]

        self.hidden_dim = hidden_dim
        self.use_attention_residuals      = use_attention_residuals
        self.use_attention_pooling        = use_attention_pooling
        self.use_global_attention_pooling = use_global_attention_pooling

        assert hidden_dim % n_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
        )

        # Project three sequence sources into hidden space
        self.prot_proj   = nn.Linear(prot_dim,   hidden_dim)
        self.pocket_proj = nn.Linear(pocket_dim, hidden_dim)
        self.lig_proj    = nn.Linear(lig_dim,    hidden_dim)

        # GNN — output dim == hidden_dim so node embeddings share the same space
        self.gnn = ManualGNN(
            in_dim     = node_dim,
            hidden_dim = hidden_dim,
            out_dim    = hidden_dim,
            n_layers   = n_layers,
            dropout    = dropout,
            rbf_dim    = rbf_dim,
        )
        self.node_proj = nn.Linear(hidden_dim, hidden_dim)

        # Cross-attention layers
        self.attn_local_to_global = nn.MultiheadAttention(
            embed_dim   = hidden_dim,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.attn_global_to_local = nn.MultiheadAttention(
            embed_dim   = hidden_dim,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,
        )

        # A1: residual LayerNorms
        self.local_layernorm  = nn.LayerNorm(hidden_dim)
        self.global_layernorm = nn.LayerNorm(hidden_dim)

        # A2/A3: learned attention pooling projections
        self.local_pool_proj  = nn.Linear(hidden_dim, 1)
        self.global_pool_proj = nn.Linear(hidden_dim, 1)

        # MLP head: 2 × hidden_dim (same as CrossAttentionFusionModel)
        self.mlp = MLP(
            in_dim      = 2 * hidden_dim,
            hidden_dims = hidden_dims,
            out_dim     = 1,
            dropout     = dropout,
        )

    def forward(
        self,
        prot_emb:   torch.Tensor,                # [B, prot_dim]   global ESM2
        pocket_emb: torch.Tensor,                # [B, pocket_dim] pocket ESM2
        lig_emb:    torch.Tensor,                # [B, lig_dim]
        x:          torch.Tensor,                # [N, node_dim]
        edge_index: torch.Tensor,                # [2, E]
        batch:      torch.Tensor | None = None,  # [N]
        edge_attr:  torch.Tensor | None = None,  # [E, 5]  col 0 = distance (Å)
        **_,
    ) -> torch.Tensor:                           # [B, 1]
        B = prot_emb.shape[0]
        device = prot_emb.device

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)

        # 1. GNN → per-node embeddings [N, hidden_dim]
        _, node_emb = self.gnn(x, edge_index, batch, edge_attr)   # [N, hidden]
        node_emb    = self.node_proj(node_emb)                     # [N, hidden]

        # 2. Project three sequence sources → global_tokens [B, 3, H]
        p_g = self.prot_proj(prot_emb).unsqueeze(1)      # [B, 1, H]
        p_p = self.pocket_proj(pocket_emb).unsqueeze(1)  # [B, 1, H]
        l   = self.lig_proj(lig_emb).unsqueeze(1)        # [B, 1, H]
        global_tokens = torch.cat([p_g, p_p, l], dim=1)  # [B, 3, H]

        # 3. Per-sample cross-attention
        local_pools:  list[torch.Tensor] = []
        global_pools: list[torch.Tensor] = []

        for i in range(B):
            mask = (batch == i)
            ni   = node_emb[mask]                       # [n_i, H]
            gi   = global_tokens[i].unsqueeze(0)        # [1, 3, H]

            # ── Local → Global ────────────────────────────────────────────
            attn_local_out, _ = self.attn_local_to_global(
                ni.unsqueeze(0), gi, gi,                # Q=[1,n_i,H], K=V=[1,3,H]
                need_weights=False,
            )
            attn_local_out = attn_local_out.squeeze(0)  # [n_i, H]

            if self.use_attention_residuals:
                attended_local = self.local_layernorm(ni + attn_local_out)
            else:
                attended_local = attn_local_out

            # ── Global → Local ────────────────────────────────────────────
            attn_global_out, _ = self.attn_global_to_local(
                gi, ni.unsqueeze(0), ni.unsqueeze(0),   # Q=[1,3,H], K=V=[1,n_i,H]
                need_weights=False,
            )
            attn_global_out = attn_global_out.squeeze(0)  # [3, H]

            if self.use_attention_residuals:
                attended_global = self.global_layernorm(gi.squeeze(0) + attn_global_out)
            else:
                attended_global = attn_global_out

            # ── A2: learned attention pooling — local ─────────────────────
            if self.use_attention_pooling:
                lp_w = torch.softmax(self.local_pool_proj(attended_local), dim=0)   # [n_i, 1]
                local_pool_i = (lp_w * attended_local).sum(dim=0)                  # [H]
            else:
                local_pool_i = attended_local.mean(dim=0)                           # [H]

            # ── A3: learned attention pooling — global ────────────────────
            if self.use_global_attention_pooling:
                gp_w = torch.softmax(self.global_pool_proj(attended_global), dim=0)  # [3, 1]
                global_pool_i = (gp_w * attended_global).sum(dim=0)                 # [H]
            else:
                global_pool_i = attended_global.mean(dim=0)                          # [H]

            local_pools.append(local_pool_i)
            global_pools.append(global_pool_i)

        local_pool  = torch.stack(local_pools,  dim=0)  # [B, H]
        global_pool = torch.stack(global_pools, dim=0)  # [B, H]
        fused = torch.cat([local_pool, global_pool], dim=-1)  # [B, 2H]
        return self.mlp(fused)

    def encode(
        self,
        prot_emb:   torch.Tensor,
        pocket_emb: torch.Tensor,
        lig_emb:    torch.Tensor,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        batch:      torch.Tensor | None = None,
        edge_attr:  torch.Tensor | None = None,
        **_,
    ) -> torch.Tensor:                           # [B, 2*hidden_dim]
        """Return the pre-MLP fused representation (no pocket_emb vs prot_emb distinction needed)."""
        B = prot_emb.shape[0]
        device = prot_emb.device
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        _, node_emb = self.gnn(x, edge_index, batch, edge_attr)
        node_emb    = self.node_proj(node_emb)
        p_g = self.prot_proj(prot_emb).unsqueeze(1)
        p_p = self.pocket_proj(pocket_emb).unsqueeze(1)
        l   = self.lig_proj(lig_emb).unsqueeze(1)
        global_tokens = torch.cat([p_g, p_p, l], dim=1)
        local_pools:  list[torch.Tensor] = []
        global_pools: list[torch.Tensor] = []
        for i in range(B):
            mask = (batch == i)
            ni = node_emb[mask]
            gi = global_tokens[i].unsqueeze(0)
            attn_local_out, _ = self.attn_local_to_global(ni.unsqueeze(0), gi, gi, need_weights=False)
            attn_local_out = attn_local_out.squeeze(0)
            attended_local = self.local_layernorm(ni + attn_local_out) if self.use_attention_residuals else attn_local_out
            attn_global_out, _ = self.attn_global_to_local(gi, ni.unsqueeze(0), ni.unsqueeze(0), need_weights=False)
            attn_global_out = attn_global_out.squeeze(0)
            attended_global = self.global_layernorm(gi.squeeze(0) + attn_global_out) if self.use_attention_residuals else attn_global_out
            lp_w = torch.softmax(self.local_pool_proj(attended_local), dim=0) if self.use_attention_pooling else None
            local_pool_i = (lp_w * attended_local).sum(dim=0) if lp_w is not None else attended_local.mean(dim=0)
            gp_w = torch.softmax(self.global_pool_proj(attended_global), dim=0) if self.use_global_attention_pooling else None
            global_pool_i = (gp_w * attended_global).sum(dim=0) if gp_w is not None else attended_global.mean(dim=0)
            local_pools.append(local_pool_i)
            global_pools.append(global_pool_i)
        local_pool  = torch.stack(local_pools,  dim=0)
        global_pool = torch.stack(global_pools, dim=0)
        return torch.cat([local_pool, global_pool], dim=-1)

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "sequence_only":           SequenceOnlyModel,
    "graph_only":              GraphOnlyModel,
    "concat_fusion":           ConcatFusionModel,
    "cross_attention_fusion":  CrossAttentionFusionModel,
    # Pocket-contextual ESM2 variants.
    # Architecture is identical to the base classes; the only difference is
    # that the training pipeline feeds a mean-pooled pocket ESM2 vector as
    # the ``prot_emb`` argument instead of the full-protein global embedding.
    "pocket_sequence_only":          SequenceOnlyModel,
    "pocket_esm2_graph_fusion":       ConcatFusionModel,
    "pocket_cross_attention_fusion":  CrossAttentionFusionModel,
    # All-features fusion: global protein + pocket protein + ligand + graph.
    # These models receive both prot_emb (global) and pocket_emb (pocket)
    # simultaneously via the dataset's dual-embedding mode.
    "all_concat_fusion":              AllConcatFusionModel,
    "all_cross_attention_fusion":     AllCrossAttentionFusionModel,
}


def build_model(name: str, cfg: dict) -> nn.Module:
    """Instantiate a model by name using a config dict.

    Parameters
    ----------
    name:
        One of the keys in ``MODEL_REGISTRY``.
    cfg:
        Sub-dict from ``phase3.yaml`` ``models.<name>`` section.

    Returns
    -------
    nn.Module
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}. Available: {sorted(MODEL_REGISTRY)}"
        )
    cls = MODEL_REGISTRY[name]
    return cls(**cfg)

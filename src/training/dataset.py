"""PyTorch Dataset and DataLoader utilities for Phase 3.

Each sample corresponds to one protein–ligand complex and may include:
  - protein global embedding    (always required)
  - ligand SMILES embedding     (always required)
  - structural graph dict       (required for graph-based models)

All heavy data (embedding dicts, graph .pt files) are memory-mapped on
first access so they can be shared across DataLoader workers via
fork/spawn without re-loading.

``BindingAffinityDataset`` reads the split parquet file, filters to the
requested split(s), and returns items as plain dicts of tensors.

Batch collation
---------------
Because structural graphs have variable node / edge counts, the standard
DataLoader collate_fn cannot stack them.  ``collate_fn`` is provided
here; it packs graphs in COO format by incrementing node indices and
building a ``batch`` tensor that maps each node to its graph index.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)

# Graph features the collate_fn will stack across samples
_GRAPH_TENSOR_KEYS = ("x", "pos", "edge_attr", "node_type",
                      "residue_index", "ligand_atom_index", "edge_type")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BindingAffinityDataset(Dataset):
    """Load cached embeddings and (optionally) graph dicts for a split.

    Parameters
    ----------
    split_df:
        DataFrame with columns ``pdb_id``, ``split``, ``delta_g_kcal_mol``.
        Should already be pre-filtered to the desired split.
    prot_emb_path:
        ``protein_global_embeddings.pt`` — dict ``{pdb_id: Tensor [480]}``.
    lig_emb_path:
        ``ligand_smiles_embeddings.pt``  — dict ``{pdb_id: Tensor [384]}``.
    graph_dir:
        Directory containing ``{pdb_id}.pt`` graph dicts.  Pass ``None``
        to disable graph loading (for sequence-only model).
    pocket_emb_path:
        ``pocket_esm2_embeddings.pt`` — dict ``{pdb_id: Tensor [480]}``.
        Required for all-features fusion models that need both global and
        pocket-contextual ESM2 embeddings simultaneously.  Pass ``None``
        to disable (most models do not need this second protein embedding).
    """

    def __init__(
        self,
        split_df:        pd.DataFrame,
        prot_emb_path:   Path,
        lig_emb_path:    Path,
        graph_dir:       Path | None = None,
        pocket_emb_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.records    = split_df.reset_index(drop=True)
        self.graph_dir  = Path(graph_dir) if graph_dir else None

        LOGGER.info(
            "Loading protein global embeddings from %s", prot_emb_path
        )
        self.prot_embs: dict[str, torch.Tensor] = torch.load(
            prot_emb_path, map_location="cpu", weights_only=False
        )
        LOGGER.info(
            "Loading ligand SMILES embeddings from %s", lig_emb_path
        )
        self.lig_embs: dict[str, torch.Tensor] = torch.load(
            lig_emb_path, map_location="cpu", weights_only=False
        )

        # Optional second protein embedding (pocket-contextual)
        self.pocket_embs: dict[str, torch.Tensor] | None = None
        if pocket_emb_path is not None:
            LOGGER.info(
                "Loading pocket ESM2 embeddings from %s", pocket_emb_path
            )
            self.pocket_embs = torch.load(
                pocket_emb_path, map_location="cpu", weights_only=False
            )

        # Validate availability
        missing_prot   = 0
        missing_lig    = 0
        missing_graph  = 0
        missing_pocket = 0
        valid_mask     = []
        for pdb_id in self.records["pdb_id"]:
            ok = True
            if pdb_id not in self.prot_embs:
                missing_prot += 1
                ok = False
            if pdb_id not in self.lig_embs:
                missing_lig += 1
                ok = False
            if self.graph_dir is not None:
                gpath = self.graph_dir / f"{pdb_id}.pt"
                if not gpath.exists():
                    missing_graph += 1
                    ok = False
            if self.pocket_embs is not None and pdb_id not in self.pocket_embs:
                missing_pocket += 1
                ok = False
            valid_mask.append(ok)

        before = len(self.records)
        self.records = self.records[valid_mask].reset_index(drop=True)
        after  = len(self.records)

        if missing_prot or missing_lig or missing_graph or missing_pocket:
            LOGGER.warning(
                "Dropped %d / %d samples: "
                "%d missing prot_emb, %d missing lig_emb, %d missing graph, "
                "%d missing pocket_emb",
                before - after, before,
                missing_prot, missing_lig, missing_graph, missing_pocket,
            )
        LOGGER.info("Dataset ready: %d samples", after)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row    = self.records.iloc[idx]
        pdb_id = row["pdb_id"]
        y      = torch.tensor([float(row["delta_g_kcal_mol"])], dtype=torch.float)

        item: dict = {
            "pdb_id":   pdb_id,
            "split":    row["split"],
            "y":        y,
            "prot_emb": self.prot_embs[pdb_id].float(),
            "lig_emb":  self.lig_embs[pdb_id].float(),
        }

        if self.pocket_embs is not None:
            item["pocket_emb"] = self.pocket_embs[pdb_id].float()

        if self.graph_dir is not None:
            g = torch.load(
                self.graph_dir / f"{pdb_id}.pt",
                map_location="cpu",
                weights_only=False,
            )
            item["x"]                 = g["x"].float()
            item["pos"]               = g["pos"].float()
            item["edge_index"]        = g["edge_index"].long()
            item["edge_attr"]         = g["edge_attr"].float()
            item["edge_type"]         = g["edge_type"].long()
            item["node_type"]         = g["node_type"].long()
            item["residue_index"]     = g["residue_index"].long()
            item["ligand_atom_index"] = g["ligand_atom_index"].long()
            item["n_ligand_atoms"]    = int(g["n_ligand_atoms"])
            item["n_pocket_residues"] = int(g["n_pocket_residues"])

        return item


# ---------------------------------------------------------------------------
# Collate function for batching variable-size graphs
# ---------------------------------------------------------------------------

def collate_fn(samples: Sequence[dict]) -> dict:
    """Collate a list of dataset items into a batched dict.

    Sequence embeddings (prot_emb, lig_emb, y) are stacked normally.

    Graph tensors are concatenated and a ``batch`` index tensor is built
    that maps each node to its position in the batch.

    Edge indices are shifted by cumulative node offsets so all graphs
    share a unified node namespace within the batch.
    """
    batch: dict = {}

    # ── scalars / metadata ─────────────────────────────────────────────────
    batch["pdb_id"] = [s["pdb_id"] for s in samples]
    batch["split"]  = [s["split"]  for s in samples]

    # ── sequence tensors ──────────────────────────────────────────────────
    batch["y"]        = torch.stack([s["y"]        for s in samples])  # [B,1]
    batch["prot_emb"] = torch.stack([s["prot_emb"] for s in samples])  # [B,480]
    batch["lig_emb"]  = torch.stack([s["lig_emb"]  for s in samples])  # [B,384]
    # pocket_emb — only present for all-features fusion models
    if "pocket_emb" in samples[0]:
        batch["pocket_emb"] = torch.stack([s["pocket_emb"] for s in samples])  # [B,480]
    # ── graph tensors (optional) ──────────────────────────────────────────
    if "x" not in samples[0]:
        return batch

    node_counts     = [s["x"].shape[0] for s in samples]
    cumulative      = [0] + list(_cumsum(node_counts))
    batch_idx_parts = [
        torch.full((n,), i, dtype=torch.long)
        for i, n in enumerate(node_counts)
    ]
    batch["batch"] = torch.cat(batch_idx_parts)  # [N_total]

    # Concatenate node-level tensors
    for key in _GRAPH_TENSOR_KEYS:
        batch[key] = torch.cat([s[key] for s in samples], dim=0)

    # Shift edge indices by cumulative node offset
    edge_parts = []
    for i, s in enumerate(samples):
        edge_parts.append(s["edge_index"] + cumulative[i])
    batch["edge_index"] = torch.cat(edge_parts, dim=1)   # [2, E_total]

    # Scalar metadata per graph
    batch["n_ligand_atoms"]    = [s["n_ligand_atoms"]    for s in samples]
    batch["n_pocket_residues"] = [s["n_pocket_residues"] for s in samples]

    return batch


def _cumsum(lst: list[int]) -> list[int]:
    out, acc = [], 0
    for v in lst:
        acc += v
        out.append(acc)
    return out

"""Assemble the full ligand–pocket complex graph.

Unified node feature vector  (NUM_NODE_FEATURES = 10):

    Ligand-atom features (indices 0–6, mirrors graph_ligand.LIGAND_FEATURE_DIM):
        [0]  atomic_num
        [1]  degree
        [2]  formal_charge
        [3]  is_aromatic
        [4]  hybridization
        [5]  is_hbond_donor
        [6]  is_hbond_acceptor

    Pocket-residue features (indices 7–8, mirrors graph_pocket.RESIDUE_FEATURE_DIM):
        [7]  aa_index
        [8]  res_idx_normalized

    Shared:
        [9]  node_type   (0 = ligand atom,  1 = pocket residue)

    Non-applicable positions are zero-padded.

Edge feature vector  (NUM_EDGE_FEATURES = 5):
    [0]  distance (Å)
    [1]  bond_is_single
    [2]  bond_is_double
    [3]  bond_is_triple
    [4]  bond_is_aromatic

    Bond type features are 0 for non-covalent edges.

Edge type codes  (``edge_type`` tensor):
    0  ligand covalent bond
    1  ligand–pocket contact  (≤ contact_cutoff Å)
    2  pocket–pocket contact  (≤ pocket_cutoff Å)

Graph dict schema
-----------------
pdb_id            : str
dataset_name      : str
split             : str
x                 : FloatTensor  [N, 10]
pos               : FloatTensor  [N, 3]
node_type         : LongTensor   [N]      0 = ligand, 1 = residue
residue_index     : LongTensor   [N]      0-based full-protein idx; -1 for ligand
ligand_atom_index : LongTensor   [N]      0-based ligand idx; -1 for residues
edge_index        : LongTensor   [2, E]
edge_attr         : FloatTensor  [E, 5]
edge_type         : LongTensor   [E]
y                 : FloatTensor  [1]
n_ligand_atoms    : int
n_pocket_residues : int
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

from src.features.graph_ligand import LigandGraph, LIGAND_FEATURE_DIM
from src.features.graph_pocket import PocketGraph, RESIDUE_FEATURE_DIM

LOGGER = logging.getLogger(__name__)

NUM_NODE_FEATURES = 10   # see module docstring
NUM_EDGE_FEATURES = 5    # [distance, single, double, triple, aromatic]

EDGE_LIGAND_BOND   = 0
EDGE_LIGAND_POCKET = 1
EDGE_POCKET_POCKET = 2


def build_complex_graph(
    pdb_id:         str,
    dataset_name:   str,
    split:          str,
    ligand_graph:   LigandGraph,
    pocket_graph:   PocketGraph,
    delta_g:        float,
    contact_cutoff: float = 6.0,
    pocket_cutoff:  float = 8.0,
) -> Optional[dict]:
    """Combine ligand and pocket into a single graph dict.

    Returns
    -------
    dict or None
        Graph dict (schema in module docstring) or ``None`` if the graph
        cannot be built (empty ligand/pocket or parsing error).
    """
    if ligand_graph.error:
        LOGGER.warning("%s: ligand error — %s", pdb_id, ligand_graph.error)
        return None
    if pocket_graph.error:
        LOGGER.warning("%s: pocket error — %s", pdb_id, pocket_graph.error)
        return None
    if ligand_graph.n_atoms == 0:
        LOGGER.warning("%s: empty ligand (0 atoms).", pdb_id)
        return None
    if pocket_graph.n_residues == 0:
        LOGGER.warning("%s: empty pocket (0 residues).", pdb_id)
        return None

    N_lig = ligand_graph.n_atoms
    N_res = pocket_graph.n_residues
    N     = N_lig + N_res

    # ── 1. Node feature matrix  [N, 10] ───────────────────────────────────
    x = torch.zeros(N, NUM_NODE_FEATURES, dtype=torch.float)
    # Ligand atoms: features at columns 0..6; node_type = 0
    x[:N_lig, :LIGAND_FEATURE_DIM]        = ligand_graph.atom_features
    x[:N_lig, NUM_NODE_FEATURES - 1]      = 0.0
    # Pocket residues: features at columns 7..8; node_type = 1
    x[N_lig:, 7:7 + RESIDUE_FEATURE_DIM] = pocket_graph.res_features
    x[N_lig:, NUM_NODE_FEATURES - 1]     = 1.0

    # ── 2. Position matrix  [N, 3] ────────────────────────────────────────
    pos = torch.cat([ligand_graph.pos, pocket_graph.pos], dim=0)

    # ── 3. Node metadata ──────────────────────────────────────────────────
    node_type = torch.zeros(N, dtype=torch.long)
    node_type[N_lig:] = 1

    residue_index = torch.full((N,), -1, dtype=torch.long)
    residue_index[N_lig:] = torch.tensor(pocket_graph.res_indices, dtype=torch.long)

    ligand_atom_index = torch.full((N,), -1, dtype=torch.long)
    ligand_atom_index[:N_lig] = torch.arange(N_lig, dtype=torch.long)

    # ── 4. Edges ──────────────────────────────────────────────────────────
    src_list:   list[torch.Tensor] = []
    dst_list:   list[torch.Tensor] = []
    attr_list:  list[torch.Tensor] = []
    etype_list: list[torch.Tensor] = []

    # 4a. Ligand covalent bonds (already both-direction from build_ligand_graph)
    E_lig = ligand_graph.bond_index.shape[1]
    if E_lig > 0:
        u = ligand_graph.bond_index[0]   # [E_lig]
        v = ligand_graph.bond_index[1]   # [E_lig]
        dist_lig = (ligand_graph.pos[u] - ligand_graph.pos[v]).norm(dim=1, keepdim=True)
        attr_lig = torch.cat([dist_lig, ligand_graph.bond_attr], dim=1)   # [E_lig, 5]
        src_list.append(u)
        dst_list.append(v)
        attr_list.append(attr_lig)
        etype_list.append(torch.full((E_lig,), EDGE_LIGAND_BOND, dtype=torch.long))

    # 4b. Ligand–pocket contacts  (within contact_cutoff; add both directions)
    if N_lig > 0 and N_res > 0:
        # [N_lig, N_res, 3]
        diff_lr = ligand_graph.pos.unsqueeze(1) - pocket_graph.pos.unsqueeze(0)
        dist_lr = diff_lr.norm(dim=2)                        # [N_lig, N_res]
        mask_lr = dist_lr <= contact_cutoff
        li, ri_local = mask_lr.nonzero(as_tuple=True)        # [E_lr] each
        E_lr = li.shape[0]
        if E_lr > 0:
            ri_global = ri_local + N_lig
            d_lr      = dist_lr[li, ri_local].unsqueeze(1)   # [E_lr, 1]
            no_bond   = torch.zeros(E_lr, 4, dtype=torch.float)
            attr_lr   = torch.cat([d_lr, no_bond], dim=1)    # [E_lr, 5]
            # l → r
            src_list.append(li);         dst_list.append(ri_global)
            attr_list.append(attr_lr);   etype_list.append(
                torch.full((E_lr,), EDGE_LIGAND_POCKET, dtype=torch.long))
            # r → l
            src_list.append(ri_global);  dst_list.append(li)
            attr_list.append(attr_lr);   etype_list.append(
                torch.full((E_lr,), EDGE_LIGAND_POCKET, dtype=torch.long))

    # 4c. Pocket–pocket contacts  (within pocket_cutoff; both directions via nonzero)
    if N_res > 1:
        diff_rr = pocket_graph.pos.unsqueeze(1) - pocket_graph.pos.unsqueeze(0)
        dist_rr = diff_rr.norm(dim=2)                                # [N_res, N_res]
        eye     = torch.eye(N_res, dtype=torch.bool)
        mask_rr = (dist_rr <= pocket_cutoff) & (~eye)
        ri, rj  = mask_rr.nonzero(as_tuple=True)
        E_rr    = ri.shape[0]
        if E_rr > 0:
            ri_g       = ri + N_lig
            rj_g       = rj + N_lig
            d_rr       = dist_rr[ri, rj].unsqueeze(1)
            no_bond_rr = torch.zeros(E_rr, 4, dtype=torch.float)
            attr_rr    = torch.cat([d_rr, no_bond_rr], dim=1)       # [E_rr, 5]
            src_list.append(ri_g);       dst_list.append(rj_g)
            attr_list.append(attr_rr);   etype_list.append(
                torch.full((E_rr,), EDGE_POCKET_POCKET, dtype=torch.long))

    # ── 5. Assemble edge tensors ──────────────────────────────────────────
    if src_list:
        edge_index = torch.stack(
            [torch.cat(src_list), torch.cat(dst_list)], dim=0
        )                                              # [2, E_total]
        edge_attr  = torch.cat(attr_list,  dim=0)     # [E_total, 5]
        edge_type  = torch.cat(etype_list, dim=0)     # [E_total]
    else:
        edge_index = torch.zeros(2, 0,              dtype=torch.long)
        edge_attr  = torch.zeros(0, NUM_EDGE_FEATURES, dtype=torch.float)
        edge_type  = torch.zeros(0,                 dtype=torch.long)

    return {
        "pdb_id":            pdb_id,
        "dataset_name":      dataset_name,
        "split":             split,
        "x":                 x,
        "pos":               pos,
        "node_type":         node_type,
        "residue_index":     residue_index,
        "ligand_atom_index": ligand_atom_index,
        "edge_index":        edge_index,
        "edge_attr":         edge_attr,
        "edge_type":         edge_type,
        "y":                 torch.tensor([delta_g], dtype=torch.float),
        "n_ligand_atoms":    N_lig,
        "n_pocket_residues": N_res,
    }

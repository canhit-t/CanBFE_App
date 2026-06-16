from __future__ import annotations

from pathlib import Path

import torch

from src.features.graph_complex import build_complex_graph
from src.features.graph_ligand import load_ligand_graph
from src.features.graph_pocket import PocketGraph

_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "HYP": "P", "TPO": "T", "SEP": "S", "PTR": "Y",
    "CSO": "C", "SEC": "C", "PYL": "K",
}
_AA1_TO_IDX = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
_AA_UNKNOWN = 20


def _residue_position(residue):
    if "CA" in residue:
        p = residue["CA"].get_coord()
        return [float(p[0]), float(p[1]), float(p[2])]

    atoms = [a for a in residue.get_atoms() if getattr(a, "element", "") != "H"]
    if not atoms:
        return None

    coords = torch.tensor([a.get_coord() for a in atoms], dtype=torch.float)
    p = coords.mean(dim=0)
    return [float(p[0]), float(p[1]), float(p[2])]


def _extract_pocket_from_full_protein(protein_pdb: Path, ligand_pos: torch.Tensor, cutoff: float) -> PocketGraph:
    from Bio.PDB import PDBParser
    from Bio.PDB.PDBExceptions import PDBConstructionWarning
    import warnings

    parser = PDBParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        structure = parser.get_structure("target", str(protein_pdb))

    residues = [r for r in structure.get_residues() if r.get_id()[0] == " "]
    if not residues:
        residues = list(structure.get_residues())

    features, positions, indices = [], [], []
    max_idx = max(1, len(residues) - 1)

    for idx, residue in enumerate(residues):
        pos = _residue_position(residue)
        if pos is None:
            continue

        min_dist = torch.cdist(torch.tensor(pos, dtype=torch.float).view(1, 3), ligand_pos).min().item()
        if min_dist <= cutoff:
            aa1 = _AA3_TO_1.get(residue.get_resname().strip(), "X")
            aa_idx = _AA1_TO_IDX.get(aa1, _AA_UNKNOWN)
            features.append([float(aa_idx), float(idx / max_idx)])
            positions.append(pos)
            indices.append(idx)

    if not features:
        return PocketGraph(
            res_features=torch.zeros(0, 2),
            pos=torch.zeros(0, 3),
            res_indices=[],
            n_residues=0,
            error=f"no pocket residues within {cutoff} Å",
        )

    return PocketGraph(
        res_features=torch.tensor(features, dtype=torch.float),
        pos=torch.tensor(positions, dtype=torch.float),
        res_indices=indices,
        n_residues=len(indices),
    )


def build_exact_complex_graph_from_pair(
    protein_pdb: Path,
    ligand_sdf: Path,
    complex_id: str,
    pocket_cutoff: float = 8.0,
    contact_cutoff: float = 6.0,
) -> dict:
    ligand_graph = load_ligand_graph(ligand_sdf)
    if ligand_graph.error:
        raise RuntimeError(f"Ligand graph failed for {ligand_sdf.name}: {ligand_graph.error}")

    pocket_graph = _extract_pocket_from_full_protein(protein_pdb, ligand_graph.pos, pocket_cutoff)
    if pocket_graph.error:
        raise RuntimeError(f"Pocket graph failed for {ligand_sdf.name}: {pocket_graph.error}")

    graph = build_complex_graph(
        pdb_id=complex_id,
        dataset_name="screening",
        split="screening",
        ligand_graph=ligand_graph,
        pocket_graph=pocket_graph,
        delta_g=0.0,
        contact_cutoff=contact_cutoff,
        pocket_cutoff=pocket_cutoff,
    )
    if graph is None:
        raise RuntimeError(f"Complex graph assembly failed for {ligand_sdf.name}")

    return graph

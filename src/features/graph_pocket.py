"""Pocket residue featurization from PDB files using Biopython.

Node feature vector layout for pocket residues (RESIDUE_FEATURE_DIM = 2):
    [0]  aa_index          (0–19 for standard AAs; 20 for unknown)
    [1]  res_idx_normalized (0-based index in full protein / max_index)
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

LOGGER = logging.getLogger(__name__)

RESIDUE_FEATURE_DIM = 2

# Amino acid 3-letter → 1-letter (mirrors parse_protein._AA3_TO_1)
_AA3_TO_1: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common non-standard residues
    "MSE": "M", "HYP": "P", "TPO": "T", "SEP": "S",
    "PTR": "Y", "CSO": "C", "SEC": "C", "PYL": "K",
}

# 1-letter → 0-based index (0–19); 20 = unknown
_AA1_TO_IDX: dict[str, int] = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
_AA1_UNKNOWN = 20


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class PocketGraph:
    """Featurized pocket residues."""
    res_features: torch.Tensor   # [N, RESIDUE_FEATURE_DIM]
    pos:          torch.Tensor   # [N, 3]
    res_indices:  list[int]      # 0-based full-protein index for each residue
    n_residues:   int
    error:        Optional[str] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_pocket_graph(
    pocket_pdb:     Path,
    pocket_indices: list[int],
    full_pdb:       Optional[Path] = None,
    window_start:   int = 0,
) -> PocketGraph:
    """Featurize pocket residues from a pocket PDB file.

    Parameters
    ----------
    pocket_pdb:
        Path to the pocket ``.pdb`` file.
    pocket_indices:
        Windowed 0-based indices into the ESM2-encoded protein slice, one
        per mapped pocket residue.  Produced by
        ``build_sequence_features.run_pocket_phase``.
    full_pdb:
        Path to the full-protein ``.pdb`` file.  When provided, residues in
        the pocket PDB are matched to ``pocket_indices`` by residue key
        (chain, sequence-number, insertion-code) rather than by position.
        This is the correct behaviour and must be supplied whenever
        ``pocket_indices`` may be a strict subset of the residues in
        ``pocket_pdb`` (e.g. after ESM2 window filtering).
    window_start:
        The start of the ESM2 encoding window (0 for non-truncated proteins).
        Used to convert windowed indices back to original full-protein indices
        for the key-based lookup.
    """
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.PDBExceptions import PDBConstructionWarning
    except ImportError as exc:
        raise ImportError("Biopython is required for pocket graph building.") from exc

    if not pocket_pdb.exists():
        return _empty_graph(f"pocket file not found: {pocket_pdb}")

    if not pocket_indices:
        return _empty_graph("pocket_indices is empty")

    parser = PDBParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        structure = parser.get_structure("pkt", str(pocket_pdb))

    # Collect residues in PDB order — ATOM records first, HETATM fallback
    residues = [
        r for r in structure.get_residues()
        if r.get_id()[0].strip() == ""
    ]
    if not residues:
        residues = list(structure.get_residues())

    if not residues:
        return _empty_graph("no residues found in pocket PDB")

    # -----------------------------------------------------------------------
    # Pair each pocket-PDB residue with its windowed index.
    #
    # Correct path (full_pdb provided):
    #   1. Build key → original_full_protein_index from full_pdb.
    #   2. For each pocket residue, look up its original index.
    #   3. Compute windowed_index = original_index - window_start.
    #   4. Keep only residues whose windowed_index is in pocket_indices.
    #
    # This is necessary because pocket_indices may be a strict subset of the
    # residues in pocket_pdb (after ESM2 window filtering), so positional
    # pairing would assign wrong positions and AA types to nodes.
    #
    # Fallback (no full_pdb): pair by position (legacy; only correct when
    # len(residues) == len(pocket_indices)).
    # -----------------------------------------------------------------------
    matched_pairs: list[tuple] = []   # (residue_object, windowed_idx)

    if full_pdb is not None and full_pdb.exists():
        try:
            from src.features.map_pocket_residues import _get_residue_keys

            full_keys = _get_residue_keys(full_pdb)
            full_lookup: dict = {key: idx for idx, key in enumerate(full_keys)}
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "%s: could not build full-protein key lookup (%s); "
                "falling back to positional pairing.",
                pocket_pdb.stem, exc,
            )
            full_lookup = {}

        if full_lookup:
            pocket_indices_set = set(pocket_indices)
            seen_keys: set = set()
            for residue in residues:
                hetfield, seqnum, icode = residue.get_id()
                chain_id = residue.get_parent().get_id()
                key = (chain_id, int(seqnum), icode.strip())
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                orig_idx = full_lookup.get(key)
                if orig_idx is None:
                    continue
                windowed_idx = orig_idx - window_start
                if windowed_idx in pocket_indices_set:
                    matched_pairs.append((residue, windowed_idx))
            # Sort by windowed index to keep consistent ordering
            matched_pairs.sort(key=lambda p: p[1])

            if not matched_pairs:
                LOGGER.warning(
                    "%s: key-based matching found 0 pairs "
                    "(pocket_indices=%d, residues_in_pdb=%d); "
                    "falling back to positional pairing.",
                    pocket_pdb.stem, len(pocket_indices), len(residues),
                )
                matched_pairs = []   # trigger fallback below

    if not matched_pairs:
        # Positional fallback
        n_use = min(len(residues), len(pocket_indices))
        if len(residues) != len(pocket_indices):
            LOGGER.warning(
                "%s: residues in PDB (%d) ≠ pocket_indices length (%d); "
                "using first %d pairs (positional fallback — supply full_pdb "
                "for correct key-based matching).",
                pocket_pdb.stem, len(residues), len(pocket_indices), n_use,
            )
        matched_pairs = [(residues[i], pocket_indices[i]) for i in range(n_use)]

    n_total = max(pocket_indices) + 1 if pocket_indices else 1

    feats:         list[list[float]] = []
    positions:     list[list[float]] = []
    valid_indices: list[int]         = []
    n_skipped = 0

    for residue, res_idx in matched_pairs:
        ca_pos = _ca_or_centroid(residue)
        if ca_pos is None:
            LOGGER.warning(
                "Residue %s (full-prot idx=%d) has no atoms; skipping.",
                residue.get_resname(), res_idx,
            )
            n_skipped += 1
            continue

        aa1    = _AA3_TO_1.get(residue.get_resname().strip(), "X")
        aa_idx = _AA1_TO_IDX.get(aa1, _AA1_UNKNOWN)

        feats.append([float(aa_idx), float(res_idx) / float(n_total)])
        positions.append(list(ca_pos))
        valid_indices.append(res_idx)

    if not feats:
        return _empty_graph(f"all {n_skipped} residues had no atoms")

    return PocketGraph(
        res_features=torch.tensor(feats,     dtype=torch.float),
        pos=torch.tensor(positions,          dtype=torch.float),
        res_indices=valid_indices,
        n_residues=len(valid_indices),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ca_or_centroid(residue) -> Optional[tuple[float, float, float]]:
    """Return Cα coordinate, or centroid of all atoms, or None if no atoms."""
    try:
        v = residue["CA"].get_vector()
        return float(v[0]), float(v[1]), float(v[2])
    except KeyError:
        pass
    atoms = list(residue.get_atoms())
    if not atoms:
        return None
    x = sum(float(a.get_vector()[0]) for a in atoms) / len(atoms)
    y = sum(float(a.get_vector()[1]) for a in atoms) / len(atoms)
    z = sum(float(a.get_vector()[2]) for a in atoms) / len(atoms)
    return x, y, z


def _empty_graph(error: str) -> PocketGraph:
    return PocketGraph(
        res_features=torch.zeros(0, RESIDUE_FEATURE_DIM, dtype=torch.float),
        pos=torch.zeros(0, 3,                            dtype=torch.float),
        res_indices=[],
        n_residues=0,
        error=error,
    )

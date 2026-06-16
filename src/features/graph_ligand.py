"""Ligand atom featurization from RDKit molecules.

Node feature vector layout for ligand atoms (LIGAND_FEATURE_DIM = 7):
    [0]  atomic_num
    [1]  degree
    [2]  formal_charge
    [3]  is_aromatic
    [4]  hybridization   (0=SP, 1=SP2, 2=SP3, 3=SP3D, 4=SP3D2, 5=OTHER)
    [5]  is_hbond_donor
    [6]  is_hbond_acceptor

Bond feature vector layout (BOND_FEATURE_DIM = 4):
    [0]  is_single
    [1]  is_double
    [2]  is_triple
    [3]  is_aromatic
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

LOGGER = logging.getLogger(__name__)

LIGAND_FEATURE_DIM = 7
BOND_FEATURE_DIM   = 4


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class LigandGraph:
    """Featurized ligand molecule."""
    atom_features: torch.Tensor   # [N, LIGAND_FEATURE_DIM]
    pos:           torch.Tensor   # [N, 3]
    bond_index:    torch.Tensor   # [2, E_bonds]  undirected (both directions)
    bond_attr:     torch.Tensor   # [E_bonds, BOND_FEATURE_DIM]
    n_atoms:       int
    error:         Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers (lazy rdkit import)
# ---------------------------------------------------------------------------

def _hybridization_map() -> dict:
    from rdkit.Chem import rdchem
    return {
        rdchem.HybridizationType.SP:    0,
        rdchem.HybridizationType.SP2:   1,
        rdchem.HybridizationType.SP3:   2,
        rdchem.HybridizationType.SP3D:  3,
        rdchem.HybridizationType.SP3D2: 4,
    }


def _bond_type_map() -> dict:
    from rdkit import Chem
    return {
        Chem.rdchem.BondType.SINGLE:   [1, 0, 0, 0],
        Chem.rdchem.BondType.DOUBLE:   [0, 1, 0, 0],
        Chem.rdchem.BondType.TRIPLE:   [0, 0, 1, 0],
        Chem.rdchem.BondType.AROMATIC: [0, 0, 0, 1],
    }


def _donor_acceptor_indices(mol) -> tuple[set[int], set[int]]:
    """Return sets of atom indices that are H-bond donors / acceptors."""
    from rdkit.Chem import MolFromSmarts
    donor_smarts = (
        "[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]"
    )
    acceptor_smarts = (
        "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),"
        "$([N;v3;!$(N-*=!@[O,N,P,S])]),$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]"
    )
    dp = MolFromSmarts(donor_smarts)
    ap = MolFromSmarts(acceptor_smarts)
    donors    = {m[0] for m in (mol.GetSubstructMatches(dp) or [])}
    acceptors = {m[0] for m in (mol.GetSubstructMatches(ap) or [])}
    return donors, acceptors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_ligand_graph(mol) -> LigandGraph:
    """Featurize an RDKit Mol with a 3D conformer.

    Returns a :class:`LigandGraph` with ``error`` set when building failed.
    """
    if mol is None:
        return _empty_graph("mol is None")

    if mol.GetNumConformers() == 0:
        return _empty_graph("no 3D conformer available")

    hyb_map  = _hybridization_map()
    bond_map = _bond_type_map()

    try:
        donors, acceptors = _donor_acceptor_indices(mol)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("H-bond pattern match failed (%s); defaulting to 0.", exc)
        donors, acceptors = set(), set()

    conf = mol.GetConformer()
    n    = mol.GetNumAtoms()

    feats:     list[list[float]] = []
    positions: list[list[float]] = []

    for i, atom in enumerate(mol.GetAtoms()):
        p = conf.GetAtomPosition(i)
        positions.append([p.x, p.y, p.z])
        feats.append([
            float(atom.GetAtomicNum()),
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            float(atom.GetIsAromatic()),
            float(hyb_map.get(atom.GetHybridization(), 5)),
            float(i in donors),
            float(i in acceptors),
        ])

    atom_features = torch.tensor(feats,     dtype=torch.float)   # [N, 7]
    pos           = torch.tensor(positions, dtype=torch.float)   # [N, 3]

    # Bonds — add both (u → v) and (v → u) for undirected graph
    src:   list[int]         = []
    dst:   list[int]         = []
    attrs: list[list[float]] = []

    for bond in mol.GetBonds():
        u     = bond.GetBeginAtomIdx()
        v     = bond.GetEndAtomIdx()
        btype = bond_map.get(bond.GetBondType(), [0, 0, 0, 0])
        src  += [u, v]
        dst  += [v, u]
        attrs += [btype, btype]

    if src:
        bond_index = torch.tensor([src, dst], dtype=torch.long)
        bond_attr  = torch.tensor(attrs,      dtype=torch.float)
    else:
        bond_index = torch.zeros(2, 0,              dtype=torch.long)
        bond_attr  = torch.zeros(0, BOND_FEATURE_DIM, dtype=torch.float)

    return LigandGraph(
        atom_features=atom_features,
        pos=pos,
        bond_index=bond_index,
        bond_attr=bond_attr,
        n_atoms=n,
    )


def load_ligand_graph(ligand_path: Path) -> LigandGraph:
    """Load a ligand from an SDF or MOL2 file and featurize it.

    Parameters
    ----------
    ligand_path:
        Path to a ``.sdf`` or ``.mol2`` file with 3D coordinates.
    """
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError as exc:
        raise ImportError("RDKit is required for ligand graph building.") from exc

    if not ligand_path.exists():
        return _empty_graph(f"ligand file not found: {ligand_path}")

    suffix = ligand_path.suffix.lower()
    mol    = None

    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(ligand_path), removeHs=True, sanitize=True)
        for m in supplier:
            if m is not None:
                mol = m
                break
    elif suffix in (".mol2", ".mol"):
        mol = Chem.MolFromMol2File(str(ligand_path), removeHs=True)
    else:
        # Fall back to SDF parser for unknown extensions
        supplier = Chem.SDMolSupplier(str(ligand_path), removeHs=True, sanitize=True)
        for m in supplier:
            if m is not None:
                mol = m
                break

    if mol is None:
        return _empty_graph(f"could not parse ligand from {ligand_path}")

    if mol.GetNumConformers() == 0:
        return _empty_graph(f"no 3D conformer in {ligand_path}")

    return build_ligand_graph(mol)


def _empty_graph(error: str) -> LigandGraph:
    return LigandGraph(
        atom_features=torch.zeros(0, LIGAND_FEATURE_DIM, dtype=torch.float),
        pos=torch.zeros(0, 3,                            dtype=torch.float),
        bond_index=torch.zeros(2, 0,                     dtype=torch.long),
        bond_attr=torch.zeros(0, BOND_FEATURE_DIM,       dtype=torch.float),
        n_atoms=0,
        error=error,
    )

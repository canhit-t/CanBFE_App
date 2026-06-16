from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.parse_protein import parse_protein

REQUIRED_SMILES_COL = "SMILES"
_ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWYUOX")


@dataclass
class StandardizedProtein:
    sequence: str | None
    num_residues: int | None
    parser_used: str | None
    warning: str | None = None


def save_uploaded_file(uploaded_file, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(uploaded_file.getvalue())
    return path


def standardize_sequence(seq: str | None) -> str | None:
    if not seq:
        return None
    seq = "".join(str(seq).split()).upper()
    seq = "".join(ch if ch in _ALLOWED_AA else "X" for ch in seq)
    return seq or None


def standardize_protein_from_pdb(pdb_path: Path) -> StandardizedProtein:
    info = parse_protein(pdb_path)
    seq = standardize_sequence(info.sequence)
    return StandardizedProtein(
        sequence=seq,
        num_residues=len(seq) if seq else info.num_residues,
        parser_used=info.parser_used,
        warning=info.parse_warning,
    )


def load_ligand_table(csv_upload) -> pd.DataFrame:
    df = pd.read_csv(csv_upload)
    if REQUIRED_SMILES_COL not in df.columns:
        candidates = [c for c in df.columns if c.strip().lower() == "smiles"]
        if candidates:
            df = df.rename(columns={candidates[0]: REQUIRED_SMILES_COL})
        else:
            raise ValueError(f"CSV must contain a '{REQUIRED_SMILES_COL}' column. Found: {list(df.columns)}")

    df = df.copy()
    df[REQUIRED_SMILES_COL] = df[REQUIRED_SMILES_COL].astype(str).str.strip()
    df = df[df[REQUIRED_SMILES_COL].notna() & (df[REQUIRED_SMILES_COL] != "")]
    if "ligand_id" not in df.columns:
        df.insert(0, "ligand_id", [f"ligand_{i+1:06d}" for i in range(len(df))])
    return df.reset_index(drop=True)


def resolve_sdf_for_row(row: pd.Series, sdf_paths_by_name: dict[str, Path]) -> Path | None:
    for col in ("pose_file", "ligand_sdf", "sdf_file", "complex_sdf"):
        if col in row and pd.notna(row[col]):
            raw = str(row[col]).strip()
            for key in (raw, Path(raw).stem):
                if key in sdf_paths_by_name:
                    return sdf_paths_by_name[key]

    for col in ("ligand_id", "name", "catalogue_id", "catalog_id"):
        if col in row and pd.notna(row[col]):
            raw = str(row[col]).strip()
            for key in (raw, f"{raw}.sdf", Path(raw).stem):
                if key in sdf_paths_by_name:
                    return sdf_paths_by_name[key]

    return None

"""Build structural ligand–pocket graphs for all complexes.

For each complex with successful metadata and known pocket residues, a
PyTorch graph dict is saved at::

    {output_dir}/{pdb_id}.pt

A summary index and build report are also saved::

    {output_dir}/graph_index.parquet
    {output_dir}/graph_build_report.json

Graph dict schema
-----------------
pdb_id            : str
dataset_name      : str
split             : str
x                 : FloatTensor  [N, 10]   — unified node features
pos               : FloatTensor  [N, 3]    — Cα / atom 3-D coordinates
node_type         : LongTensor   [N]       — 0 = ligand atom, 1 = pocket residue
residue_index     : LongTensor   [N]       — 0-based full-protein index; -1 for ligand
ligand_atom_index : LongTensor   [N]       — 0-based ligand index; -1 for residues
edge_index        : LongTensor   [2, E]
edge_attr         : FloatTensor  [E, 5]    — [distance, single, double, triple, aromatic]
edge_type         : LongTensor   [E]       — 0 bond / 1 lig-pock / 2 pock-pock
y                 : FloatTensor  [1]       — delta_g_kcal_mol
n_ligand_atoms    : int
n_pocket_residues : int

Usage
-----
::

    python -m src.features.build_structure_graphs \\
        --metadata       data/processed/all_metadata.parquet \\
        --pocket-indices data/features/pocket_residue_indices.pt \\
        --output-dir     data/features/graphs \\
        --contact-cutoff 6.0 \\
        --pocket-cutoff  8.0
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.data.utils import setup_logging
from src.features.graph_complex import (
    NUM_EDGE_FEATURES,
    NUM_NODE_FEATURES,
    build_complex_graph,
)
from src.features.graph_ligand import load_ligand_graph
from src.features.graph_pocket import build_pocket_graph

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_filter(metadata_path: Path) -> pd.DataFrame:
    """Load metadata and keep only rows with all required fields."""
    df = pd.read_parquet(metadata_path)
    LOGGER.info("Loaded metadata: %d rows", len(df))

    before = len(df)
    df = df[df["parse_status"] == "success"].copy()
    df = df[df["protein_sequence"].notna()].copy()
    df = df[df["ligand_smiles"].notna()].copy()
    df = df[df["delta_g_kcal_mol"].notna()].copy()
    df = df[df["ligand_file"].notna()].copy()
    df = df[df["pocket_file"].notna()].copy()
    after = len(df)

    LOGGER.info(
        "After filtering (success + non-null fields): %d rows (%d dropped)",
        after, before - after,
    )
    return df.reset_index(drop=True)


def _load_pocket_indices(path: Path) -> dict[str, list[int]]:
    """Load pocket_residue_indices.pt — a dict mapping pdb_id → list[int]."""
    if not path.exists():
        raise FileNotFoundError(
            f"Pocket indices file not found: {path}\n"
            "Run the sequence featurization pipeline first "
            "(src.features.build_sequence_features)."
        )
    LOGGER.info("Loading pocket indices from %s", path)
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_window_meta(path: Path) -> dict[str, dict]:
    """Load protein_window_meta.pt — optional; returns {} if not found."""
    if not path.exists():
        LOGGER.warning(
            "protein_window_meta.pt not found at %s; "
            "build_pocket_graph will use positional pairing (may be incorrect "
            "for truncated proteins).",
            path,
        )
        return {}
    LOGGER.info("Loading window meta from %s", path)
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    setup_logging(args.log_level)

    t0 = time.time()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_filter(args.metadata)
    if args.max_samples:
        df = df.head(args.max_samples)
        LOGGER.info("--max-samples: using first %d rows.", len(df))

    pocket_indices_dict = _load_pocket_indices(args.pocket_indices)

    window_meta_path = args.window_meta
    window_meta_dict = _load_window_meta(window_meta_path)
    if window_meta_dict:
        LOGGER.info("Window meta loaded: %d entries.", len(window_meta_dict))

    # ── Process complexes ─────────────────────────────────────────────────
    index_rows: list[dict] = []
    n_ok = n_failed = n_skipped = 0

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Building graphs", unit="complex"
    ):
        pdb_id   = row.pdb_id
        out_path = output_dir / f"{pdb_id}.pt"

        # Resume: skip already-built graphs
        if out_path.exists():
            n_skipped += 1
            index_rows.append({
                "pdb_id":            pdb_id,
                "dataset_name":      row.dataset_name,
                "split":             row.split,
                "build_status":      "cached",
                "n_ligand_atoms":    None,
                "n_pocket_residues": None,
                "n_nodes":           None,
                "n_edges":           None,
                "error":             None,
            })
            continue

        # Pocket indices
        pocket_idx = pocket_indices_dict.get(pdb_id)
        if not pocket_idx:
            LOGGER.warning("%s: no pocket indices found; skipping.", pdb_id)
            n_failed += 1
            index_rows.append(_fail_row(row, "no pocket indices"))
            continue

        # Ligand
        lig_graph = load_ligand_graph(Path(row.ligand_file))
        if lig_graph.error:
            LOGGER.warning("%s: %s", pdb_id, lig_graph.error)
            n_failed += 1
            index_rows.append(_fail_row(row, f"ligand: {lig_graph.error}"))
            continue

        # Pocket
        wm           = window_meta_dict.get(pdb_id, {})
        window_start = wm.get("window_start", 0)
        full_pdb_path = (
            Path(row.protein_file)
            if pd.notna(row.get("protein_file")) and row.protein_file
            else None
        )
        pkt_graph = build_pocket_graph(
            pocket_pdb   = Path(row.pocket_file),
            pocket_indices = pocket_idx,
            full_pdb     = full_pdb_path,
            window_start = window_start,
        )
        if pkt_graph.error:
            LOGGER.warning("%s: %s", pdb_id, pkt_graph.error)
            n_failed += 1
            index_rows.append(_fail_row(row, f"pocket: {pkt_graph.error}"))
            continue

        # Assemble graph
        graph = build_complex_graph(
            pdb_id=pdb_id,
            dataset_name=row.dataset_name,
            split=row.split,
            ligand_graph=lig_graph,
            pocket_graph=pkt_graph,
            delta_g=float(row.delta_g_kcal_mol),
            contact_cutoff=args.contact_cutoff,
            pocket_cutoff=args.pocket_cutoff,
        )
        if graph is None:
            n_failed += 1
            index_rows.append(_fail_row(row, "build_complex_graph returned None"))
            continue

        torch.save(graph, out_path)
        n_ok += 1
        index_rows.append({
            "pdb_id":            pdb_id,
            "dataset_name":      row.dataset_name,
            "split":             row.split,
            "build_status":      "ok",
            "n_ligand_atoms":    graph["n_ligand_atoms"],
            "n_pocket_residues": graph["n_pocket_residues"],
            "n_nodes":           int(graph["x"].shape[0]),
            "n_edges":           int(graph["edge_index"].shape[1]),
            "error":             None,
        })

    # ── Save index + report ───────────────────────────────────────────────
    index_df   = pd.DataFrame(index_rows)
    index_path = output_dir / "graph_index.parquet"
    index_df.to_parquet(index_path, index=False)
    LOGGER.info("Graph index saved: %d rows → %s", len(index_df), index_path)

    elapsed = time.time() - t0
    report  = {
        "metadata":          str(args.metadata),
        "pocket_indices":    str(args.pocket_indices),
        "output_dir":        str(output_dir),
        "contact_cutoff":    args.contact_cutoff,
        "pocket_cutoff":     args.pocket_cutoff,
        "total_input_rows":  len(df),
        "n_ok":              n_ok,
        "n_failed":          n_failed,
        "n_skipped_cached":  n_skipped,
        "num_node_features": NUM_NODE_FEATURES,
        "num_edge_features": NUM_EDGE_FEATURES,
        "elapsed_seconds":   round(elapsed, 1),
    }
    report_path = output_dir / "graph_build_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    LOGGER.info(
        "Report saved → %s  (%.1f s total)  ok=%d  failed=%d  skipped=%d",
        report_path, elapsed, n_ok, n_failed, n_skipped,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fail_row(row, error: str) -> dict:
    return {
        "pdb_id":            row.pdb_id,
        "dataset_name":      row.dataset_name,
        "split":             row.split,
        "build_status":      "failed",
        "n_ligand_atoms":    None,
        "n_pocket_residues": None,
        "n_nodes":           None,
        "n_edges":           None,
        "error":             error,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build structural ligand–pocket graphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/all_metadata.parquet"),
        metavar="FILE",
        help="Metadata parquet file.",
    )
    p.add_argument(
        "--pocket-indices",
        type=Path,
        default=Path("data/features/pocket_residue_indices.pt"),
        metavar="FILE",
        help="pocket_residue_indices.pt dict (pdb_id → list[int]).",
    )
    p.add_argument(
        "--window-meta",
        type=Path,
        default=Path("data/features/protein_window_meta.pt"),
        metavar="FILE",
        help="protein_window_meta.pt dict (pdb_id → {window_start, …}). "
             "Used for key-based pocket residue matching in truncated proteins.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/features/graphs"),
        metavar="DIR",
        help="Directory for graph .pt files, index, and report.",
    )
    p.add_argument(
        "--contact-cutoff",
        type=float,
        default=6.0,
        metavar="Å",
        help="Ligand–pocket contact distance cutoff (Å).",
    )
    p.add_argument(
        "--pocket-cutoff",
        type=float,
        default=8.0,
        metavar="Å",
        help="Pocket residue–residue contact distance cutoff (Å).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N rows (smoke test).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


if __name__ == "__main__":
    main()

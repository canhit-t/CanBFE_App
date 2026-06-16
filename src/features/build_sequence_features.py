"""Build cached sequence/SMILES feature embeddings for downstream models.

Phases
------
1. **Protein phase** — ESM2 encodes every protein sequence:
   - Global mean-pooled embedding → ``protein_global_embeddings.pt``
   - Per-residue embeddings      → ``protein_residue_embeddings/{pdb_id}.pt``

2. **Pocket phase** — maps pocket residues to full-protein positions,
   slices residue embeddings:
   - Pocket residue indices  → ``pocket_residue_indices.parquet``
   - Pocket residue tensors  → ``pocket_residue_embeddings/{pdb_id}.pt``

3. **Ligand phase** — ChemBERTa encodes every SMILES:
   - Global mean-pooled embedding → ``ligand_smiles_embeddings.pt``

4. **Index + report** — writes ``feature_index.parquet`` and
   ``feature_build_report.json``.

Caching
-------
Each phase checks for already-computed outputs and skips them.  Re-running
the script is safe and only processes new or missing entries.

Usage
-----
::

    python -m src.features.build_sequence_features \\
        --metadata     data/processed/all_metadata.parquet \\
        --output-dir   data/features \\
        --protein-model facebook/esm2_t12_35M_UR50D \\
        --ligand-model  DeepChem/ChemBERTa-77M-MTR \\
        --batch-size    8 \\
        --device        auto
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm

from src.data.utils import setup_logging
from src.features.encode_ligand_smiles import ChemBERTaEncoder
from src.features.encode_protein_esm import ESM2Encoder, ESM2_MAX_RESIDUES
from src.features.map_pocket_residues import map_pocket_residues

LOGGER = logging.getLogger(__name__)

# Number of batches processed before writing a checkpoint to disk
_CHECKPOINT_EVERY = 50


# ---------------------------------------------------------------------------
# Window computation
# ---------------------------------------------------------------------------

def compute_window_start(
    seq_len: int,
    pocket_indices: list[int],
    max_residues: int = ESM2_MAX_RESIDUES,
) -> int:
    """Return the optimal window start for a long protein.

    For proteins with ``seq_len <= max_residues`` this always returns 0.

    For longer proteins the window is centred around the median pocket
    residue index so that as many pocket residues as possible fall inside
    the 1022-residue window::

        window_start = max(0, min(median_pocket_index - 511, seq_len - max_residues))

    Parameters
    ----------
    seq_len:
        Full length of the protein sequence.
    pocket_indices:
        0-based indices of pocket residues in the full sequence.
    max_residues:
        ESM2 window size (default 1022).

    Returns
    -------
    int
        0-based start position of the encoding window.
    """
    if seq_len <= max_residues:
        return 0
    if not pocket_indices:
        return 0   # no pocket info — fall back to beginning
    import statistics
    median_idx = int(statistics.median(pocket_indices))
    half = max_residues // 2
    ws = max(0, min(median_idx - half, seq_len - max_residues))
    return ws


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _load_pt_dict(path: Path) -> dict:
    """Load a .pt dict file, returning an empty dict if the file does not exist."""
    if path.exists():
        LOGGER.info("Resuming from existing file: %s", path)
        return torch.load(path, map_location="cpu", weights_only=True)
    return {}


def _save_pt_dict(d: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, path)


def _chunked(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# ---------------------------------------------------------------------------
# Metadata loading + filtering
# ---------------------------------------------------------------------------

def load_and_filter(metadata_path: Path) -> pd.DataFrame:
    """Load metadata, keep only rows with all required fields."""
    df = pd.read_parquet(metadata_path)
    LOGGER.info("Loaded metadata: %d rows", len(df))

    before = len(df)
    df = df[df["parse_status"] == "success"].copy()
    df = df[df["protein_sequence"].notna()].copy()
    df = df[df["ligand_smiles"].notna()].copy()
    df = df[df["delta_g_kcal_mol"].notna()].copy()
    after = len(df)

    LOGGER.info(
        "After filtering (success + non-null sequence/smiles/delta_g): %d rows "
        "(%d dropped)",
        after, before - after,
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Phase 0: Pre-pocket mapping (Biopython only — no GPU)
# ---------------------------------------------------------------------------

def run_pre_pocket_mapping(
    df: pd.DataFrame,
    output_dir: Path,
    max_residues: int = ESM2_MAX_RESIDUES,
) -> dict[str, list[int]]:
    """Parse pocket PDB files with Biopython and return raw pocket indices.

    This is a fast CPU-only pass that runs *before* ESM2 encoding so that
    long-protein windows can be centred on the actual binding pocket.

    Results are cached in ``pre_pocket_indices.pt``; already-computed entries
    are skipped on resume.

    Returns
    -------
    dict mapping pdb_id → list[int] (0-based full-protein residue indices).
    """
    cache_path  = output_dir / "pre_pocket_indices.pt"
    cached: dict[str, list[int]] = (
        torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache_path.exists()
        else {}
    )

    pending = [
        row for _, row in df.iterrows()
        if row.pdb_id not in cached
        and pd.notna(row.get("pocket_file"))
        and pd.notna(row.get("protein_file"))
    ]

    if not pending:
        LOGGER.info(
            "Pre-pocket mapping: all %d entries already cached.", len(cached)
        )
        return cached

    LOGGER.info(
        "Pre-pocket mapping (Biopython, CPU): %d complexes to process.",
        len(pending),
    )

    n_ok = n_failed = 0
    for row in tqdm(pending, desc="Pre-pocket map", unit="complex"):
        try:
            mapping = map_pocket_residues(
                pdb_id     = row.pdb_id,
                full_pdb   = Path(row.protein_file),
                pocket_pdb = Path(row.pocket_file),
                max_residues = max_residues,
            )
            cached[row.pdb_id] = mapping.pocket_indices
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("%s: pre-pocket mapping failed: %s", row.pdb_id, exc)
            cached[row.pdb_id] = []   # empty → window_start defaults to 0
            n_failed += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cached, cache_path)
    LOGGER.info(
        "Pre-pocket mapping done: ok=%d  failed=%d  total_cached=%d",
        n_ok, n_failed, len(cached),
    )
    return cached


# ---------------------------------------------------------------------------
# Phase 1: Protein embeddings
# ---------------------------------------------------------------------------

def run_protein_phase(
    df: pd.DataFrame,
    encoder: ESM2Encoder,
    output_dir: Path,
    residue_dir: Path,
    batch_size: int,
    precomputed_pocket_indices: dict[str, list[int]] | None = None,
) -> dict:
    """Encode all protein sequences; cache global and per-residue embeddings.

    For proteins longer than ESM2_MAX_RESIDUES the encoding window is
    chosen to contain as many pocket residues as possible, centred on the
    median pocket index.  Window metadata (start, end, was_truncated) is
    stored in a sidecar file ``protein_window_meta.pt`` so the pocket phase
    can use correct remapped indices.

    Parameters
    ----------
    precomputed_pocket_indices:
        Optional dict ``{pdb_id: list[int]}`` of already-known pocket
        residue indices (from a previous run or from map_pocket_residues).
        Used to compute the optimal window for long proteins.  When absent
        or a pdb_id is missing, window_start defaults to 0 (start of seq).
    """
    global_path  = output_dir / "protein_global_embeddings.pt"
    window_path  = output_dir / "protein_window_meta.pt"
    global_embs: dict[str, torch.Tensor] = _load_pt_dict(global_path)
    # window_meta: pdb_id -> {"window_start": int, "window_end": int, "was_truncated": bool}
    window_meta: dict[str, dict] = (
        torch.load(window_path, map_location="cpu", weights_only=False)
        if window_path.exists()
        else {}
    )

    pocket_idx_source: dict[str, list[int]] = precomputed_pocket_indices or {}

    # Collect rows that still need encoding
    pending = [
        row
        for _, row in df.iterrows()
        if (
            row.pdb_id not in global_embs
            or not (residue_dir / f"{row.pdb_id}.pt").exists()
        )
    ]

    if not pending:
        LOGGER.info("All protein embeddings already cached (%d total).", len(global_embs))
        return {"skipped": len(df), "encoded": 0, "failed": 0}

    LOGGER.info(
        "Protein phase: %d/%d complexes need encoding.", len(pending), len(df)
    )

    # Sort by sequence length for minimal padding waste within batches
    pending.sort(key=lambda r: len(r.protein_sequence))

    n_encoded = n_failed = n_truncated = 0
    batches = list(_chunked(pending, batch_size))

    for batch_idx, batch in enumerate(
        tqdm(batches, desc="Protein (ESM2)", unit="batch")
    ):
        seqs   = [r.protein_sequence for r in batch]
        pids   = [r.pdb_id for r in batch]
        window_starts = [
            compute_window_start(
                seq_len       = len(r.protein_sequence),
                pocket_indices= pocket_idx_source.get(r.pdb_id, []),
                max_residues  = encoder.max_residues,
            )
            for r in batch
        ]

        try:
            embs = encoder.encode_batch(seqs, pids, window_starts=window_starts)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Protein batch %d failed entirely: %s", batch_idx, exc)
            n_failed += len(batch)
            continue

        for emb in embs:
            if emb.error:
                LOGGER.warning("Protein encoding error for %s: %s", emb.pdb_id, emb.error)
                n_failed += 1
                continue
            global_embs[emb.pdb_id] = emb.global_emb
            torch.save(emb.residue_emb, residue_dir / f"{emb.pdb_id}.pt")
            window_meta[emb.pdb_id] = {
                "window_start":   emb.window_start,
                "window_end":     emb.window_end,
                "was_truncated":  emb.was_truncated,
                "seq_len_original": emb.seq_len_original,
                "seq_len_encoded":  emb.seq_len_encoded,
            }
            n_encoded += 1
            if emb.was_truncated:
                n_truncated += 1

        # Checkpoint periodically
        if (batch_idx + 1) % _CHECKPOINT_EVERY == 0:
            _save_pt_dict(global_embs, global_path)
            _save_pt_dict(window_meta, window_path)
            LOGGER.debug("Checkpoint: %d global protein embeddings saved.", len(global_embs))

    _save_pt_dict(global_embs, global_path)
    _save_pt_dict(window_meta, window_path)
    LOGGER.info(
        "Protein phase done: encoded=%d  failed=%d  truncated=%d",
        n_encoded, n_failed, n_truncated,
    )
    return {
        "skipped": len(df) - len(pending),
        "encoded": n_encoded,
        "failed": n_failed,
        "truncated": n_truncated,
        "hidden_dim": encoder.hidden_dim,
        "model": encoder.model_name,
    }


# ---------------------------------------------------------------------------
# Phase 2: Pocket mapping + sliced embeddings
# ---------------------------------------------------------------------------

def run_pocket_phase(
    df: pd.DataFrame,
    output_dir: Path,
    residue_dir: Path,
    max_residues: int = ESM2_MAX_RESIDUES,
) -> dict:
    """Map pocket residues, remap to windowed coordinates, slice residue embeddings."""
    indices_path     = output_dir / "pocket_residue_indices.parquet"
    pt_indices_path  = output_dir / "pocket_residue_indices.pt"
    window_path      = output_dir / "protein_window_meta.pt"
    pocket_emb_dir   = output_dir / "pocket_residue_embeddings"
    pocket_emb_dir.mkdir(parents=True, exist_ok=True)

    # Load window metadata (window_start per pdb_id)
    window_meta: dict[str, dict] = (
        torch.load(window_path, map_location="cpu", weights_only=False)
        if window_path.exists()
        else {}
    )

    # ---------- resume: use the .pt dict (no pyarrow list-column issues) ----------
    pocket_indices_pt: dict[str, list[int]] = (
        torch.load(pt_indices_path, map_location="cpu", weights_only=False)
        if pt_indices_path.exists()
        else {}
    )
    already_mapped: set[str] = set(pocket_indices_pt.keys())

    existing_idx = pd.read_parquet(indices_path) if indices_path.exists() else pd.DataFrame()

    if already_mapped:
        LOGGER.info("Resuming pocket phase: %d already mapped.", len(already_mapped))

    pending = [
        row
        for _, row in df.iterrows()
        if row.pdb_id not in already_mapped
        and (residue_dir / f"{row.pdb_id}.pt").exists()
    ]

    if not pending:
        LOGGER.info("All pocket mappings already cached (%d total).", len(already_mapped))
        return {"skipped": len(already_mapped), "mapped": 0, "failed": 0}

    LOGGER.info("Pocket phase: %d complexes to process.", len(pending))

    new_rows: list[dict] = []
    n_ok = n_partial = n_failed = 0

    for row in tqdm(pending, desc="Pocket mapping", unit="complex"):
        full_pdb   = Path(row.protein_file) if pd.notna(row.get("protein_file")) else None
        pocket_pdb = Path(row.pocket_file)  if pd.notna(row.get("pocket_file"))  else None

        mapping = map_pocket_residues(
            pdb_id=row.pdb_id,
            full_pdb=full_pdb,
            pocket_pdb=pocket_pdb,
            max_residues=max_residues,
        )

        # ── Windowing remapping ───────────────────────────────────────────
        wm             = window_meta.get(row.pdb_id, {})
        window_start   = wm.get("window_start", 0)
        window_end_pos = wm.get("window_end",   max_residues)   # exclusive
        was_truncated  = wm.get("was_truncated", False)
        seq_len_orig   = wm.get("seq_len_original", len(row.protein_sequence)
                                 if pd.notna(row.get("protein_sequence")) else 0)

        n_pocket_orig  = len(mapping.pocket_indices)
        n_dropped_win  = 0

        if was_truncated and mapping.pocket_indices:
            # Keep only pocket residues inside the encoded window
            inside = [
                i for i in mapping.pocket_indices
                if window_start <= i < window_end_pos
            ]
            dropped = [
                i for i in mapping.pocket_indices
                if not (window_start <= i < window_end_pos)
            ]
            if dropped:
                n_dropped_win = len(dropped)
                LOGGER.warning(
                    "%s: %d pocket residue(s) outside ESM2 window [%d, %d); dropping: %s",
                    row.pdb_id, n_dropped_win, window_start, window_end_pos,
                    dropped[:10],
                )
            # Remap to windowed coordinates
            mapping.pocket_indices = [i - window_start for i in inside]

        # ── Slice residue embeddings to get pocket embeddings ─────────────
        if mapping.pocket_indices:
            try:
                res_emb = torch.load(
                    residue_dir / f"{row.pdb_id}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                n_residues_actual = res_emb.shape[0]
                valid_indices = [i for i in mapping.pocket_indices if i < n_residues_actual]
                n_oob = len(mapping.pocket_indices) - len(valid_indices)
                if n_oob > 0:
                    LOGGER.warning(
                        "%s: %d pocket index(es) out of res_emb bounds "
                        "(res_emb.shape[0]=%d); dropping them.",
                        row.pdb_id, n_oob, n_residues_actual,
                    )
                    mapping.pocket_indices = valid_indices
                    mapping.n_truncated_dropped += n_oob
                    if not valid_indices:
                        mapping.mapping_status = "partial"
                pocket_emb = res_emb[mapping.pocket_indices, :]   # [n_pocket, D]
                torch.save(pocket_emb, pocket_emb_dir / f"{row.pdb_id}.pt")
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "%s: failed to slice pocket embeddings: %s", row.pdb_id, exc
                )
                mapping.mapping_status = "partial"
                mapping.error = str(exc)

        n_pocket_encoded = len(mapping.pocket_indices)

        new_rows.append(
            {
                "pdb_id":                      mapping.pdb_id,
                "pocket_indices":              json.dumps(mapping.pocket_indices),
                "n_full_residues":             mapping.n_full_residues,
                "n_pocket_residues":           mapping.n_pocket_residues,
                "n_mapped":                    mapping.n_mapped,
                "n_unmapped":                  mapping.n_unmapped,
                "n_truncated_dropped":         mapping.n_truncated_dropped,
                "mapping_status":              mapping.mapping_status,
                "error":                       mapping.error,
                # window provenance fields
                "original_sequence_length":    seq_len_orig,
                "esm_window_start":            window_start,
                "esm_window_end":              window_end_pos,
                "was_truncated":               was_truncated,
                "num_pocket_residues_original":n_pocket_orig,
                "num_pocket_residues_encoded": n_pocket_encoded,
                "num_pocket_residues_dropped": n_dropped_win,
            }
        )

        if mapping.mapping_status == "ok":
            n_ok += 1
        elif mapping.mapping_status in ("partial", "no_pocket", "no_full_protein"):
            n_partial += 1
        else:
            n_failed += 1

    # Merge with existing and save
    new_df = pd.DataFrame(new_rows)
    combined = (
        pd.concat([existing_idx, new_df], ignore_index=True)
        if not existing_idx.empty
        else new_df
    )
    combined.to_parquet(indices_path, index=False)

    for r in new_rows:
        pocket_indices_pt[r["pdb_id"]] = json.loads(r["pocket_indices"])
    torch.save(pocket_indices_pt, pt_indices_path)

    LOGGER.info(
        "Pocket phase done: ok=%d  partial=%d  failed=%d  total_saved=%d",
        n_ok, n_partial, n_failed, len(combined),
    )
    return {
        "skipped": len(already_mapped),
        "ok": n_ok,
        "partial": n_partial,
        "failed": n_failed,
    }


# ---------------------------------------------------------------------------
# Phase 3: Ligand embeddings
# ---------------------------------------------------------------------------

def run_ligand_phase(
    df: pd.DataFrame,
    encoder: ChemBERTaEncoder,
    output_dir: Path,
    batch_size: int,
) -> dict:
    """Encode all SMILES strings; cache mean-pooled ligand embeddings."""
    ligand_path = output_dir / "ligand_smiles_embeddings.pt"
    ligand_embs: dict[str, torch.Tensor] = _load_pt_dict(ligand_path)

    pending = [
        row for _, row in df.iterrows()
        if row.pdb_id not in ligand_embs
    ]

    if not pending:
        LOGGER.info("All ligand embeddings already cached (%d total).", len(ligand_embs))
        return {"skipped": len(df), "encoded": 0, "failed": 0}

    LOGGER.info(
        "Ligand phase: %d/%d complexes need encoding.", len(pending), len(df)
    )

    n_encoded = n_failed = n_truncated = 0
    batches = list(_chunked(pending, batch_size))

    for batch_idx, batch in enumerate(
        tqdm(batches, desc="Ligand (ChemBERTa)", unit="batch")
    ):
        smiles = [r.ligand_smiles for r in batch]
        pids   = [r.pdb_id for r in batch]

        try:
            embs = encoder.encode_batch(smiles, pids)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Ligand batch %d failed entirely: %s", batch_idx, exc)
            n_failed += len(batch)
            continue

        for emb in embs:
            if emb.error:
                LOGGER.warning("Ligand encoding error for %s: %s", emb.pdb_id, emb.error)
                n_failed += 1
                continue
            ligand_embs[emb.pdb_id] = emb.global_emb
            n_encoded += 1
            if emb.was_truncated:
                n_truncated += 1

        if (batch_idx + 1) % _CHECKPOINT_EVERY == 0:
            _save_pt_dict(ligand_embs, ligand_path)
            LOGGER.debug("Checkpoint: %d ligand embeddings saved.", len(ligand_embs))

    _save_pt_dict(ligand_embs, ligand_path)
    LOGGER.info(
        "Ligand phase done: encoded=%d  failed=%d  truncated=%d",
        n_encoded, n_failed, n_truncated,
    )
    return {
        "skipped": len(df) - len(pending),
        "encoded": n_encoded,
        "failed": n_failed,
        "truncated": n_truncated,
        "hidden_dim": encoder.hidden_dim,
        "model": encoder.model_name,
    }


# ---------------------------------------------------------------------------
# Feature index
# ---------------------------------------------------------------------------

def build_feature_index(
    df: pd.DataFrame,
    output_dir: Path,
    residue_dir: Path,
) -> pd.DataFrame:
    """Build a flat index of which features are available per complex."""
    global_embs  = _load_pt_dict(output_dir / "protein_global_embeddings.pt")
    ligand_embs  = _load_pt_dict(output_dir / "ligand_smiles_embeddings.pt")

    pocket_emb_dir = output_dir / "pocket_residue_embeddings"

    # Load pocket index from the .pt dict (avoids pyarrow/torch Windows crash
    # that occurs when loading parquet with native list columns after torch import).
    pt_indices_path = output_dir / "pocket_residue_indices.pt"
    if pt_indices_path.exists():
        # weights_only=False because values are plain Python lists, not tensors
        pocket_indices_pt: dict[str, list[int]] = torch.load(
            pt_indices_path, map_location="cpu", weights_only=False
        )
    else:
        pocket_indices_pt = {}

    # Fall back to deserialising the parquet if the .pt is absent (e.g. old outputs).
    pocket_idx_path = output_dir / "pocket_residue_indices.parquet"
    if not pocket_indices_pt and pocket_idx_path.exists():
        _pq = pd.read_parquet(pocket_idx_path)
        for _, _r in _pq.iterrows():
            pocket_indices_pt[_r["pdb_id"]] = json.loads(_r["pocket_indices"])
        LOGGER.warning(
            "pocket_residue_indices.pt not found; fell back to parquet. "
            "Re-run the pocket phase to generate the .pt file."
        )

    # For scalar metadata (n_full_residues, n_pocket_residues) we still use parquet.
    pocket_meta_path = output_dir / "pocket_residue_indices.parquet"
    if pocket_meta_path.exists():
        pocket_meta_df = pd.read_parquet(pocket_meta_path).set_index("pdb_id")
    else:
        pocket_meta_df = pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        pid = row.pdb_id
        has_global   = pid in global_embs
        has_residue  = (residue_dir / f"{pid}.pt").exists()
        has_pocket_m = pid in pocket_indices_pt
        has_pocket_e = (pocket_emb_dir / f"{pid}.pt").exists()
        has_ligand   = pid in ligand_embs

        n_full = (
            int(pocket_meta_df.loc[pid, "n_full_residues"])
            if has_pocket_m and pid in pocket_meta_df.index else None
        )
        n_pocket = (
            int(pocket_meta_df.loc[pid, "n_pocket_residues"])
            if has_pocket_m and pid in pocket_meta_df.index else None
        )

        rows.append(
            {
                "pdb_id":                 pid,
                "dataset_name":           row.dataset_name,
                "split":                  row.split,
                "delta_g_kcal_mol":       row.delta_g_kcal_mol,
                "has_protein_global_emb": has_global,
                "has_protein_residue_emb":has_residue,
                "has_pocket_mapping":     has_pocket_m,
                "has_pocket_emb":         has_pocket_e,
                "has_ligand_emb":         has_ligand,
                "n_full_residues":        n_full,
                "n_pocket_residues":      n_pocket,
                "all_features_ok": all(
                    [has_global, has_residue, has_pocket_m, has_pocket_e, has_ligand]
                ),
            }
        )

    index_df = pd.DataFrame(rows)
    n_complete = int(index_df["all_features_ok"].sum())
    LOGGER.info(
        "Feature index: %d/%d complexes have all features.", n_complete, len(index_df)
    )
    return index_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build cached ESM2/ChemBERTa embeddings for protein-ligand complexes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/all_metadata.parquet"),
        metavar="FILE",
        help="Path to the all_metadata parquet file.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/features"),
        metavar="DIR",
        help="Directory for all feature outputs.",
    )
    p.add_argument(
        "--protein-model",
        default="facebook/esm2_t12_35M_UR50D",
        metavar="HF_ID",
        help="HuggingFace model ID for the protein ESM2 encoder.",
    )
    p.add_argument(
        "--ligand-model",
        default="DeepChem/ChemBERTa-77M-MTR",
        metavar="HF_ID",
        help="HuggingFace model ID for the ligand ChemBERTa encoder.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="Number of sequences/SMILES per encoding batch.",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N samples (useful for debugging).",
    )
    p.add_argument(
        "--skip-protein",
        action="store_true",
        help="Skip the protein encoding phase.",
    )
    p.add_argument(
        "--skip-ligand",
        action="store_true",
        help="Skip the ligand encoding phase.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    setup_logging(getattr(logging, args.log_level))

    t_start = time.time()
    device = _resolve_device(args.device)
    LOGGER.info("Device: %s", device)

    # ── Create output directories ─────────────────────────────────────────
    output_dir = args.output_dir
    residue_dir = output_dir / "protein_residue_embeddings"
    residue_dir.mkdir(parents=True, exist_ok=True)

    # ── Load + filter metadata ────────────────────────────────────────────
    df = load_and_filter(args.metadata)
    if args.max_samples:
        df = df.head(args.max_samples)
        LOGGER.info("--max-samples: using first %d rows.", len(df))

    report: dict = {
        "metadata": str(args.metadata),
        "output_dir": str(output_dir),
        "protein_model": args.protein_model,
        "ligand_model": args.ligand_model,
        "batch_size": args.batch_size,
        "device": device,
        "total_input_rows": len(df),
    }

    # ── Phase 0: Pre-pocket mapping (fast, CPU-only) ──────────────────────
    if not args.skip_protein:
        LOGGER.info("=== Phase 0: Pre-pocket mapping (for ESM2 window selection) ===")
        precomputed_pocket_indices = run_pre_pocket_mapping(df, output_dir)
    else:
        precomputed_pocket_indices = {}

    # ── Phase 1: Protein ──────────────────────────────────────────────────
    if not args.skip_protein:
        LOGGER.info("=== Phase 1: Protein (ESM2) ===")
        prot_encoder = ESM2Encoder(
            model_name=args.protein_model,
            device=device,
        )
        report["protein"] = run_protein_phase(
            df, prot_encoder, output_dir, residue_dir, args.batch_size,
            precomputed_pocket_indices=precomputed_pocket_indices,
        )
        prot_max_residues = prot_encoder.max_residues  # capture before deletion
        del prot_encoder  # free VRAM before pocket phase
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        LOGGER.info("Skipping protein phase (--skip-protein).")
        report["protein"] = {"skipped_by_flag": True}
        prot_max_residues = ESM2_MAX_RESIDUES  # default when protein phase skipped

    # ── Phase 2: Pocket ───────────────────────────────────────────────────
    LOGGER.info("=== Phase 2: Pocket mapping ===")
    report["pocket"] = run_pocket_phase(df, output_dir, residue_dir, prot_max_residues)

    # ── Phase 3: Ligand ───────────────────────────────────────────────────
    if not args.skip_ligand:
        LOGGER.info("=== Phase 3: Ligand (ChemBERTa) ===")
        lig_encoder = ChemBERTaEncoder(
            model_name=args.ligand_model,
            device=device,
        )
        report["ligand"] = run_ligand_phase(
            df, lig_encoder, output_dir, args.batch_size
        )
        del lig_encoder
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        LOGGER.info("Skipping ligand phase (--skip-ligand).")
        report["ligand"] = {"skipped_by_flag": True}

    # ── Feature index ─────────────────────────────────────────────────────
    LOGGER.info("=== Building feature index ===")
    feature_index = build_feature_index(df, output_dir, residue_dir)
    feature_index.to_parquet(output_dir / "feature_index.parquet", index=False)
    LOGGER.info("Feature index saved: %s rows.", len(feature_index))

    report["feature_index"] = {
        "total": len(feature_index),
        "all_features_ok": int(feature_index["all_features_ok"].sum()),
        "by_split": feature_index.groupby("split")["all_features_ok"]
        .sum()
        .astype(int)
        .to_dict(),
    }

    # ── Report ────────────────────────────────────────────────────────────
    report["elapsed_seconds"] = round(time.time() - t_start, 1)
    report_path = output_dir / "feature_build_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    LOGGER.info("Report saved → %s  (%.1f s total)", report_path, report["elapsed_seconds"])


if __name__ == "__main__":
    main()

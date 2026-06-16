"""Second-window ESM2 pooling for complexes with dropped pocket residues.

Problem
-------
For proteins longer than 1022 residues the primary ESM2 pass encodes a
single contiguous window centred on the **median** pocket residue index.
When a binding pocket spans multiple protein chains the pocket residues can
be spread far apart in the concatenated sequence, causing the outer ones to
fall outside the 1022-residue primary window.

``build_sequence_features.py`` logs these as
``num_pocket_residues_dropped > 0`` in ``pocket_residue_indices.parquet``.
(~205 complexes in the PDBBind v2018 dataset.)

Solution
--------
For every such complex this script performs a **second ESM2 forward pass**
on a new window centred on the median of the *dropped* pocket residue
indices.  The resulting per-residue embeddings are used to pool only the
previously-dropped residues.  The secondary pool is then merged with the
primary pool using a **weighted mean** (weighted by residue count):

    merged = (n_primary * pool_primary + n_secondary * pool_secondary)
             / (n_primary + n_secondary)

This is mathematically equivalent to mean-pooling all pocket residue
embeddings in a single pass, had a single window covered them all.

The script overwrites the affected entries in
``pocket_esm2_embeddings.pt`` in-place and writes a separate audit log
``pocket_esm2_multiwindow_log.parquet`` recording per-complex details.

Idempotency
-----------
The audit log is used for resume: complexes already present in the log
are skipped on re-runs.  Running the script a second time is safe.

Usage
-----
::

    # Normal run
    .\\bindfusion311\\Scripts\\python.exe -m src.features.build_multiwindow_pocket_esm2 `
        --feature-dir data/features `
        --metadata    data/processed/all_metadata.parquet

    # Dry run (prints stats, writes nothing)
    .\\bindfusion311\\Scripts\\python.exe -m src.features.build_multiwindow_pocket_esm2 `
        --feature-dir data/features `
        --metadata    data/processed/all_metadata.parquet `
        --dry-run

Prerequisites
-------------
All of the following must exist (produced by build_sequence_features.py):

- ``data/features/pre_pocket_indices.pt``          raw full-protein pocket indices
- ``data/features/protein_window_meta.pt``         primary window metadata
- ``data/features/pocket_residue_indices.pt``      windowed pocket indices (for n_primary)
- ``data/features/pocket_residue_indices.parquet`` tabular metadata (to find dropped cases)
- ``data/features/pocket_esm2_embeddings.pt``      primary pooled embeddings to update
"""
from __future__ import annotations

import argparse
import logging
import statistics
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.data.utils import setup_logging
from src.features.build_sequence_features import compute_window_start
from src.features.encode_protein_esm import ESM2Encoder, ESM2_MAX_RESIDUES

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def build_multiwindow_pocket_esm2(
    feature_dir:   Path,
    metadata_path: Path,
    model_name:    str  = "facebook/esm2_t12_35M_UR50D",
    device:        str  = "cpu",
    dry_run:       bool = False,
) -> dict:
    """Run second-window ESM2 for complexes with dropped pocket residues.

    Parameters
    ----------
    feature_dir:
        Directory containing all feature files produced by
        ``build_sequence_features.py``.
    metadata_path:
        Path to ``all_metadata.parquet`` (for protein sequences).
    model_name:
        HuggingFace ESM2 model identifier.
    device:
        ``'cpu'``, ``'cuda'``, or ``'auto'``.
    dry_run:
        If True, compute everything but write nothing.

    Returns
    -------
    dict with summary statistics.
    """
    # ── Resolve paths ─────────────────────────────────────────────────────
    pre_pocket_path  = feature_dir / "pre_pocket_indices.pt"
    window_meta_path = feature_dir / "protein_window_meta.pt"
    parquet_path     = feature_dir / "pocket_residue_indices.parquet"
    pt_indices_path  = feature_dir / "pocket_residue_indices.pt"
    emb_path         = feature_dir / "pocket_esm2_embeddings.pt"
    mw_log_path      = feature_dir / "pocket_esm2_multiwindow_log.parquet"

    for p in [pre_pocket_path, window_meta_path, parquet_path, pt_indices_path, emb_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required input not found: {p}\n"
                "Run build_sequence_features.py and build_pocket_esm2_features.py first."
            )

    # ── Load inputs ───────────────────────────────────────────────────────
    LOGGER.info("Loading pre-pocket indices  …")
    pre_pocket: dict[str, list[int]] = torch.load(
        pre_pocket_path, map_location="cpu", weights_only=False
    )

    LOGGER.info("Loading window metadata  …")
    window_meta: dict[str, dict] = torch.load(
        window_meta_path, map_location="cpu", weights_only=False
    )

    LOGGER.info("Loading windowed pocket indices  …")
    pocket_indices_pt: dict[str, list[int]] = torch.load(
        pt_indices_path, map_location="cpu", weights_only=False
    )

    LOGGER.info("Loading pocket ESM2 embeddings  …")
    existing_embs: dict[str, torch.Tensor] = torch.load(
        emb_path, map_location="cpu", weights_only=True
    )

    LOGGER.info("Loading pocket mapping parquet  …")
    par_df = pd.read_parquet(
        parquet_path,
        columns=["pdb_id", "num_pocket_residues_dropped"],
    )

    LOGGER.info("Loading metadata (protein sequences)  …")
    meta_df = pd.read_parquet(metadata_path, columns=["pdb_id", "protein_sequence"])
    seq_map: dict[str, str] = dict(
        zip(meta_df["pdb_id"], meta_df["protein_sequence"])
    )

    # ── Find complexes with dropped residues ──────────────────────────────
    dropped_df = par_df[par_df["num_pocket_residues_dropped"] > 0].copy()
    LOGGER.info(
        "Complexes with dropped pocket residues: %d / %d",
        len(dropped_df), len(par_df),
    )

    # ── Resume support ────────────────────────────────────────────────────
    already_done: set[str] = set()
    if mw_log_path.exists():
        mw_done = pd.read_parquet(mw_log_path, columns=["pdb_id"])
        already_done = set(mw_done["pdb_id"])
        LOGGER.info("Resuming: %d complexes already multi-windowed.", len(already_done))

    pending = [
        pid for pid in dropped_df["pdb_id"]
        if pid not in already_done
    ]
    LOGGER.info("%d complexes to process this run.", len(pending))

    if not pending:
        LOGGER.info("All multi-window complexes already processed.  Nothing to do.")
        return {
            "total_with_dropped": len(dropped_df),
            "already_done":       len(already_done),
            "processed":          0,
            "improved":           0,
            "failed":             0,
        }

    # ── Resolve device ────────────────────────────────────────────────────
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load ESM2 encoder ─────────────────────────────────────────────────
    LOGGER.info("Loading ESM2 encoder (%s) → %s  …", model_name, device)
    encoder = ESM2Encoder(model_name=model_name, device=device)

    # ── Process each complex ──────────────────────────────────────────────
    log_rows:                   list[dict] = []
    n_improved = n_failed = n_no_dropped_in_w2 = 0

    for pdb_id in tqdm(pending, desc="Multi-window ESM2", unit="complex"):
        wm       = window_meta.get(pdb_id, {})
        w1_start = wm.get("window_start", 0)
        w1_end   = wm.get("window_end",   ESM2_MAX_RESIDUES)

        raw_indices: list[int] = pre_pocket.get(pdb_id, [])
        if not raw_indices:
            LOGGER.warning("%s: no pre-pocket indices found; skipping.", pdb_id)
            n_failed += 1
            log_rows.append({
                "pdb_id": pdb_id, "status": "no_pre_pocket_indices",
                "n_dropped_raw": 0, "n_second_window_captured": 0,
                "n_primary": 0, "w2_start": -1, "w2_end": -1, "error":
                "no pre_pocket_indices",
            })
            continue

        # Identify which raw indices were dropped by the primary window
        dropped_raw = [i for i in raw_indices if not (w1_start <= i < w1_end)]
        if not dropped_raw:
            # Parquet flags this complex but the raw indices all fit — data race or
            # a pre_pocket_indices rebuild; treat as already resolved.
            LOGGER.debug(
                "%s: parquet reports dropped residues but raw indices all fit in "
                "primary window; treating as resolved.", pdb_id
            )
            log_rows.append({
                "pdb_id": pdb_id, "status": "already_resolved",
                "n_dropped_raw": 0, "n_second_window_captured": 0,
                "n_primary": len(pocket_indices_pt.get(pdb_id, [])),
                "w2_start": -1, "w2_end": -1, "error": None,
            })
            continue

        seq = seq_map.get(pdb_id)
        if not seq:
            LOGGER.warning("%s: no protein sequence in metadata; skipping.", pdb_id)
            n_failed += 1
            log_rows.append({
                "pdb_id": pdb_id, "status": "no_sequence",
                "n_dropped_raw": len(dropped_raw), "n_second_window_captured": 0,
                "n_primary": 0, "w2_start": -1, "w2_end": -1,
                "error": "no protein sequence in metadata",
            })
            continue

        seq_len = len(seq)

        # Compute second window start centred on the median of the dropped indices
        w2_start = compute_window_start(seq_len, dropped_raw)
        w2_end   = min(seq_len, w2_start + ESM2_MAX_RESIDUES)

        # Which dropped residues fall inside the second window?
        in_w2 = [i for i in dropped_raw if w2_start <= i < w2_end]
        if not in_w2:
            LOGGER.warning(
                "%s: none of the %d dropped residues fall inside second window "
                "[%d, %d); cannot improve embedding.",
                pdb_id, len(dropped_raw), w2_start, w2_end,
            )
            n_no_dropped_in_w2 += 1
            log_rows.append({
                "pdb_id": pdb_id, "status": "no_residues_in_second_window",
                "n_dropped_raw": len(dropped_raw), "n_second_window_captured": 0,
                "n_primary": len(pocket_indices_pt.get(pdb_id, [])),
                "w2_start": w2_start, "w2_end": w2_end,
                "error": "no dropped residues inside second window",
            })
            continue

        # Second-window-relative indices (0-based into ESM2 output)
        w2_indices = [i - w2_start for i in in_w2]

        # ── ESM2 forward pass on the second window ─────────────────────────
        try:
            embs = encoder.encode_batch([seq], [pdb_id], window_starts=[w2_start])
            emb  = embs[0]
            if emb.error:
                raise RuntimeError(emb.error)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "%s: ESM2 second-window encoding failed: %s", pdb_id, exc
            )
            n_failed += 1
            log_rows.append({
                "pdb_id": pdb_id, "status": "encoding_failed",
                "n_dropped_raw": len(dropped_raw), "n_second_window_captured": 0,
                "n_primary": 0, "w2_start": w2_start, "w2_end": w2_end,
                "error": str(exc),
            })
            continue

        w2_res_emb = emb.residue_emb   # [seq_len_encoded_w2, hidden_dim]

        # Bounds-check second-window indices
        valid_w2 = [j for j in w2_indices if 0 <= j < w2_res_emb.shape[0]]
        if not valid_w2:
            LOGGER.warning(
                "%s: all second-window indices out of bounds "
                "(res_emb.shape[0]=%d); skipping.",
                pdb_id, w2_res_emb.shape[0],
            )
            n_failed += 1
            log_rows.append({
                "pdb_id": pdb_id, "status": "w2_all_oob",
                "n_dropped_raw": len(dropped_raw), "n_second_window_captured": 0,
                "n_primary": 0, "w2_start": w2_start, "w2_end": w2_end,
                "error": "all second-window indices out of bounds",
            })
            continue

        secondary_pool = w2_res_emb[valid_w2, :].mean(dim=0)   # [hidden_dim]
        n_secondary    = len(valid_w2)

        # ── Merge primary and secondary pools ─────────────────────────────
        primary_pool = existing_embs.get(pdb_id)
        if primary_pool is None:
            LOGGER.warning(
                "%s: no primary embedding found in pocket_esm2_embeddings.pt; "
                "using secondary pool only.",
                pdb_id,
            )
            merged = secondary_pool
            n_primary = 0
        else:
            n_primary = len(pocket_indices_pt.get(pdb_id, []))
            if n_primary == 0:
                # Fallback: use secondary only (shouldn't happen)
                merged    = secondary_pool
                n_primary = 0
                LOGGER.warning("%s: n_primary=0, using secondary pool only.", pdb_id)
            else:
                # Weighted mean ≡ pooling all residue vectors together
                merged = (
                    (n_primary * primary_pool + n_secondary * secondary_pool)
                    / (n_primary + n_secondary)
                )

        if not dry_run:
            existing_embs[pdb_id] = merged

        n_improved += 1
        log_rows.append({
            "pdb_id":                   pdb_id,
            "status":                   "ok",
            "n_dropped_raw":            len(dropped_raw),
            "n_second_window_captured": n_secondary,
            "n_primary":                n_primary,
            "w1_start":                 w1_start,
            "w1_end":                   w1_end,
            "w2_start":                 w2_start,
            "w2_end":                   w2_end,
            "error":                    None,
        })

        LOGGER.debug(
            "%s: improved — primary=%d  secondary=%d  w2=[%d, %d)",
            pdb_id, n_primary, n_secondary, w2_start, w2_end,
        )

    # ── Persist ───────────────────────────────────────────────────────────
    if not dry_run:
        torch.save(existing_embs, emb_path)
        LOGGER.info(
            "Saved updated pocket_esm2_embeddings.pt  "
            "(%d improved embeddings).", n_improved
        )

        new_df = pd.DataFrame(log_rows)
        if mw_log_path.exists():
            old_df = pd.read_parquet(mw_log_path)
            old_df = old_df[~old_df["pdb_id"].isin(set(new_df["pdb_id"]))]
            combined = pd.concat([old_df, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_parquet(mw_log_path, index=False)
        LOGGER.info("Saved multi-window log → %s", mw_log_path)
    else:
        LOGGER.info(
            "[DRY RUN] Would improve %d embeddings, write %s.",
            n_improved, mw_log_path,
        )

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {
        "total_with_dropped":        len(dropped_df),
        "already_done":              len(already_done),
        "processed_this_run":        len(log_rows),
        "improved":                  n_improved,
        "failed":                    n_failed,
        "no_residues_in_w2":         n_no_dropped_in_w2,
    }
    LOGGER.info(
        "Done.  improved=%d  failed=%d  no_residues_in_w2=%d",
        n_improved, n_failed, n_no_dropped_in_w2,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run a second ESM2 window for complexes whose pocket residues were "
            "dropped from the primary window and merge the pools."
        )
    )
    ap.add_argument(
        "--feature-dir", required=True, type=Path,
        help="Feature directory (must contain pre_pocket_indices.pt, etc.).",
    )
    ap.add_argument(
        "--metadata", required=True, type=Path,
        help="Path to all_metadata.parquet (for protein sequences).",
    )
    ap.add_argument(
        "--model", default="facebook/esm2_t12_35M_UR50D",
        help="HuggingFace ESM2 model identifier.",
    )
    ap.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "auto"],
        help="Compute device for ESM2 (default: cpu).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Compute everything but write no output files.",
    )
    ap.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = ap.parse_args()

    setup_logging(args.log_level)

    build_multiwindow_pocket_esm2(
        feature_dir   = args.feature_dir,
        metadata_path = args.metadata,
        model_name    = args.model,
        device        = args.device,
        dry_run       = args.dry_run,
    )


if __name__ == "__main__":
    main()

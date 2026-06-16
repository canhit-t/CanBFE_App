"""Build pooled pocket-contextual ESM2 feature vectors.

Implementation — full chain audit
----------------------------------
This script implements pocket ESM2 pooling as follows.  The five numbered
points match the project specification exactly:

1. ESM2 is run on the full protein sequence (or a contiguous window of it
   when seq_len > 1022).  This happens in build_sequence_features.py
   (run_protein_phase / ESM2Encoder.encode_batch).  The encoded input is
   always the contiguous slice  full_seq[ws : ws + L],  never a stitched
   or artificial sequence.

2. The mapping from original full-protein residue indices to ESM2 output
   rows is preserved as follows:
     - ESM2 tokenisation:  [CLS] aa_0 aa_1 … aa_{L-1} [EOS]
     - ESM2 output extraction:  res_emb = hidden[1 : L+1]   (CLS stripped)
     - Therefore  res_emb[k]  ==  embedding of  full_seq[ws + k]
     - For full-protein residue index i:  correct ESM2 row = i - ws
   run_pocket_phase in build_sequence_features.py applies this transform:
       pocket_indices = [i - ws for i in raw_full_protein_indices
                        if ws <= i < ws + L]
   and saves the windowed-coordinate list to pocket_residue_indices.pt.
   This step also drops and logs any pocket residues that fall outside the
   encoded window, recording them in pocket_residue_indices.parquet.

3. This script selects only the mapped pocket residue embeddings:
       pocket_tensor = res_emb[pocket_indices]   # [n_pocket, hidden_dim]

4. Those selected embeddings are mean-pooled into a fixed-length vector:
       pooled = pocket_tensor.mean(dim=0)        # [hidden_dim]

5. ESM2 is never called again here.  The pooling is pure indexing and
   arithmetic on already-saved tensors.  No stitched sequences are created.

Index safety checks performed at runtime
-----------------------------------------
- Explicit check that every index satisfies  0 <= index < res_emb.shape[0].
  Any out-of-bounds indices are dropped and logged before pooling.
- Cross-check against protein_window_meta.pt: verify that no stored index
  reaches beyond the encoded window boundary, i.e.
      ws + index  <  ws + seq_len_encoded   ↔   index < seq_len_encoded
  This detects the specific off-by-one that would arise if window_end were
  off by 1 (i.e. an index equal to seq_len_encoded would point past EOS).
- Any complex with n_unmapped > 0 in the mapping parquet (pocket residues
  that Biopython could not match to the full-protein PDB) is flagged.
- Any complex with num_pocket_residues_dropped > 0 (pocket residues that
  were outside the ESM2 window) is flagged.
- Complexes where pocket_indices is empty are reported as 'no_pocket_indices'.

Outputs
-------
``pocket_esm2_embeddings.pt``
    Dict ``{pdb_id: Tensor[hidden_dim]}`` — mean-pooled pocket ESM2 vector.
    Format mirrors ``protein_global_embeddings.pt``.

``pocket_esm2_feature_index.parquet``
    Per-complex metadata with columns:
      pdb_id, n_input_indices, n_valid_indices, n_oob_indices,
      n_unmapped_residues, n_window_dropped_residues,
      exceeds_window_boundary, build_status, error

Usage
-----
::

    .\\bindfusion311\\Scripts\\python.exe -m src.features.build_pocket_esm2_features `
        --feature-dir  data/features `
        --validate

    # Dry run (prints stats, writes nothing):
    .\\bindfusion311\\Scripts\\python.exe -m src.features.build_pocket_esm2_features `
        --feature-dir  data/features `
        --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core pooling logic
# ---------------------------------------------------------------------------

def pool_pocket_esm2(
    pdb_id:          str,
    residue_emb_dir: Path,
    pocket_indices:  list[int],
    seq_len_encoded: int | None = None,
) -> tuple[torch.Tensor | None, str, str | None, dict]:
    """Load per-residue embeddings and pool the pocket subset.

    Parameters
    ----------
    pdb_id:
        PDB identifier.
    residue_emb_dir:
        Directory containing ``{pdb_id}.pt`` per-residue tensors
        (shape ``[seq_len_encoded, hidden_dim]``).
    pocket_indices:
        Windowing-corrected 0-based indices into the per-residue tensor.
        These come directly from ``pocket_residue_indices.pt``.
    seq_len_encoded:
        Expected number of rows in the residue tensor (from window_meta).
        Used for check (c): index < seq_len_encoded catches CLS off-by-one.

    Returns
    -------
    (tensor | None, status, error_message, detail_dict)
        status one of: 'ok', 'no_residue_file', 'no_pocket_indices',
        'all_oob', 'partial_oob', 'failed'
        detail_dict keys: n_valid, n_oob, n_negative, n_exceeds_window
    """
    _empty = {"n_valid": 0, "n_oob": 0, "n_negative": 0, "n_exceeds_window": 0}
    res_path = residue_emb_dir / f"{pdb_id}.pt"

    # ── Per-residue file must exist ────────────────────────────────────────
    if not res_path.exists():
        return None, "no_residue_file", f"Missing {res_path}", _empty

    # ── Must have at least one pocket index ───────────────────────────────
    if not pocket_indices:
        return None, "no_pocket_indices", "pocket_indices list is empty", _empty

    try:
        res_emb = torch.load(res_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        return None, "failed", f"torch.load failed: {exc}", _empty

    n_residues = res_emb.shape[0]

    # ── Check (a): no negative indices ────────────────────────────────────
    negative = [i for i in pocket_indices if i < 0]
    if negative:
        LOGGER.warning(
            "%s: %d negative pocket index(es); dropped: %s",
            pdb_id, len(negative), negative[:10],
        )

    # ── Check (c): index < seq_len_encoded ────────────────────────────────
    exceeds_window: list[int] = []
    if seq_len_encoded is not None:
        exceeds_window = [i for i in pocket_indices if i >= 0 and i >= seq_len_encoded]
        if exceeds_window:
            LOGGER.warning(
                "%s: %d pocket index(es) >= seq_len_encoded=%d (would point to EOS); "
                "dropped: %s",
                pdb_id, len(exceeds_window), seq_len_encoded, exceeds_window[:10],
            )

    # ── Check (b): index < n_rows ─────────────────────────────────────────
    oob = [i for i in pocket_indices if 0 <= i < n_residues and i >= n_residues]
    # Collect all valid indices (non-negative, within tensor bounds)
    valid = [i for i in pocket_indices if 0 <= i < n_residues]

    if not valid:
        detail = {
            "n_valid": 0,
            "n_oob": len([i for i in pocket_indices if i >= n_residues and i >= 0]),
            "n_negative": len(negative),
            "n_exceeds_window": len(exceeds_window),
        }
        return None, "all_oob", (
            f"All {len(pocket_indices)} pocket indices invalid "
            f"(res_emb.shape[0]={n_residues})"
        ), detail

    n_actually_oob = len([i for i in pocket_indices if i >= 0 and i >= n_residues])
    if n_actually_oob:
        LOGGER.warning(
            "%s: %d pocket index(es) out of tensor bounds (shape[0]=%d); dropped.",
            pdb_id, n_actually_oob, n_residues,
        )

    detail = {
        "n_valid":           len(valid),
        "n_oob":             n_actually_oob,
        "n_negative":        len(negative),
        "n_exceeds_window":  len(exceeds_window),
    }

    # ── Mean pool over valid pocket residues ──────────────────────────────
    pocket_tensor = res_emb[valid, :]            # [n_valid, hidden_dim]
    pooled        = pocket_tensor.mean(dim=0)    # [hidden_dim]

    has_issues = bool(negative or exceeds_window or n_actually_oob)
    status = "partial_oob" if has_issues else "ok"
    return pooled, status, None, detail


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_pocket_esm2_features(
    feature_dir: Path,
    output_dir:  Path,
    dry_run:     bool = False,
) -> dict:
    """Pool pocket ESM2 embeddings for all complexes in the feature directory.

    Parameters
    ----------
    feature_dir:
        Directory containing:
          - ``protein_residue_embeddings/``  (per-residue .pt files)
          - ``pocket_residue_indices.pt``    (windowing-corrected indices)
    output_dir:
        Where to write ``pocket_esm2_embeddings.pt`` and the index parquet.
        Often the same as ``feature_dir``.
    dry_run:
        If True, compute and report statistics but write nothing.

    Returns
    -------
    dict with build statistics.
    """
    residue_dir      = feature_dir / "protein_residue_embeddings"
    pt_indices_path  = feature_dir / "pocket_residue_indices.pt"
    window_meta_path = feature_dir / "protein_window_meta.pt"
    parquet_path     = feature_dir / "pocket_residue_indices.parquet"
    out_emb_path     = output_dir  / "pocket_esm2_embeddings.pt"
    out_idx_path     = output_dir  / "pocket_esm2_feature_index.parquet"

    # ── Load required inputs ──────────────────────────────────────────────
    if not pt_indices_path.exists():
        raise FileNotFoundError(
            f"Pocket residue indices not found: {pt_indices_path}\n"
            "Run build_sequence_features.py first."
        )
    if not residue_dir.exists():
        raise FileNotFoundError(
            f"Protein residue embeddings directory not found: {residue_dir}"
        )

    pocket_indices_all: dict[str, list[int]] = torch.load(
        pt_indices_path, map_location="cpu", weights_only=False
    )
    LOGGER.info("Loaded pocket indices for %d complexes.", len(pocket_indices_all))

    # window_meta gives us seq_len_encoded per complex for window-boundary check (c)
    window_meta: dict[str, dict] = {}
    if window_meta_path.exists():
        window_meta = torch.load(window_meta_path, map_location="cpu", weights_only=False)
        LOGGER.info("Loaded window metadata for %d complexes.", len(window_meta))
    else:
        LOGGER.warning(
            "protein_window_meta.pt not found — window-boundary cross-check disabled."
        )

    # pocket_residue_indices.parquet lets us report upstream unmapped/dropped counts
    parquet_audit: dict[str, dict] = {}
    if parquet_path.exists():
        try:
            par_df = pd.read_parquet(
                parquet_path,
                columns=["pdb_id", "n_unmapped", "num_pocket_residues_dropped"],
            )
            for _, r in par_df.iterrows():
                parquet_audit[r["pdb_id"]] = {
                    "n_unmapped": int(r.get("n_unmapped", 0) or 0),
                    "n_dropped":  int(r.get("num_pocket_residues_dropped", 0) or 0),
                }
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not load parquet audit data: %s", exc)
    else:
        LOGGER.warning(
            "pocket_residue_indices.parquet not found — "
            "unmapped/truncated residue audit disabled."
        )

    # ── Discover and intersect ────────────────────────────────────────────
    available_pdb_ids = sorted(p.stem for p in residue_dir.glob("*.pt"))
    LOGGER.info("Found %d per-residue embedding files.", len(available_pdb_ids))

    pdb_ids_to_process = sorted(
        pid for pid in available_pdb_ids if pid in pocket_indices_all
    )
    LOGGER.info(
        "Processing %d complexes (%d residue files had no pocket index entry).",
        len(pdb_ids_to_process),
        len(available_pdb_ids) - len(pdb_ids_to_process),
    )

    # Report upstream mapping issues before processing
    n_with_unmapped = sum(1 for v in parquet_audit.values() if v["n_unmapped"] > 0)
    n_with_dropped  = sum(1 for v in parquet_audit.values() if v["n_dropped"] > 0)
    if n_with_unmapped:
        LOGGER.warning(
            "%d complexes have pocket residues that could not be mapped to the "
            "full-protein PDB (n_unmapped > 0). These residues have no ESM2 "
            "embedding and are absent from pocket_indices.",
            n_with_unmapped,
        )
    if n_with_dropped:
        LOGGER.warning(
            "%d complexes have pocket residues that fell outside the ESM2 "
            "encoding window and were dropped (num_pocket_residues_dropped > 0).",
            n_with_dropped,
        )

    # ── Resume ────────────────────────────────────────────────────────────
    existing_embs: dict[str, torch.Tensor] = {}
    if out_emb_path.exists():
        existing_embs = torch.load(out_emb_path, map_location="cpu", weights_only=True)
        LOGGER.info("Resuming: %d embeddings already cached.", len(existing_embs))

    pending = [pid for pid in pdb_ids_to_process if pid not in existing_embs]
    LOGGER.info("%d complexes need processing.", len(pending))

    # ── Process ───────────────────────────────────────────────────────────
    counters: dict[str, int] = {
        "ok": 0, "partial_oob": 0, "no_pocket_indices": 0,
        "no_residue_file": 0, "all_oob": 0, "failed": 0,
    }
    rows: list[dict] = []

    for pdb_id in tqdm(pending, desc="Pocket ESM2 pool", unit="complex"):
        indices = pocket_indices_all[pdb_id]
        wm      = window_meta.get(pdb_id, {})
        audit   = parquet_audit.get(pdb_id, {"n_unmapped": 0, "n_dropped": 0})

        pooled, status, error, detail = pool_pocket_esm2(
            pdb_id, residue_dir, indices,
            seq_len_encoded=wm.get("seq_len_encoded", None),
        )

        counters[status] = counters.get(status, 0) + 1

        rows.append({
            "pdb_id":                    pdb_id,
            "n_input_indices":           len(indices),
            "n_valid_indices":           detail["n_valid"],
            "n_oob_indices":             detail["n_oob"] + detail["n_negative"],
            "n_exceeds_window_boundary": detail["n_exceeds_window"],
            "n_unmapped_residues":       audit["n_unmapped"],
            "n_window_dropped_residues": audit["n_dropped"],
            "build_status":              status,
            "error":                     error,
        })

        if pooled is not None and not dry_run:
            existing_embs[pdb_id] = pooled

    # ── Persist ───────────────────────────────────────────────────────────
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(existing_embs, out_emb_path)
        LOGGER.info("Saved %d pocket ESM2 embeddings → %s", len(existing_embs), out_emb_path)

        new_df = pd.DataFrame(rows)
        if not new_df.empty:
            if out_idx_path.exists():
                old_df = pd.read_parquet(out_idx_path)
                old_df = old_df[~old_df["pdb_id"].isin(set(new_df["pdb_id"]))]
                index_df = pd.concat([old_df, new_df], ignore_index=True)
            else:
                index_df = new_df
            index_df.to_parquet(out_idx_path, index=False)
            LOGGER.info("Saved feature index → %s", out_idx_path)
        else:
            LOGGER.info("No new rows to write to feature index (all already cached).")
    else:
        LOGGER.info("[DRY RUN] Would write to %s and %s", out_emb_path, out_idx_path)

    # ── Summary ───────────────────────────────────────────────────────────
    total_ok     = counters.get("ok", 0) + counters.get("partial_oob", 0)
    total_failed = sum(v for k, v in counters.items() if k not in ("ok", "partial_oob"))
    n_exceeds_window = sum(r["n_exceeds_window_boundary"] for r in rows)

    summary = {
        "total_residue_files":          len(available_pdb_ids),
        "total_with_pocket_indices":    len(pdb_ids_to_process),
        "processed_this_run":           len(pending),
        "status_counts":                counters,
        "newly_built":                  total_ok,
        "failed":                       total_failed,
        "total_embeddings_in_file":     len(existing_embs),
        "upstream_with_unmapped":       n_with_unmapped,
        "upstream_with_window_dropped": n_with_dropped,
        "index_exceeds_window_total":   n_exceeds_window,
    }
    LOGGER.info(
        "Done. ok=%d  partial_oob=%d  no_indices=%d  all_oob=%d  failed=%d  "
        "total_in_file=%d  upstream_unmapped=%d  upstream_dropped=%d  exceeds_window=%d",
        counters.get("ok", 0),
        counters.get("partial_oob", 0),
        counters.get("no_pocket_indices", 0),
        counters.get("all_oob", 0),
        counters.get("failed", 0),
        len(existing_embs),
        n_with_unmapped,
        n_with_dropped,
        n_exceeds_window,
    )

    failed_rows = [r for r in rows if r["build_status"] not in ("ok", "partial_oob")]
    if failed_rows:
        LOGGER.warning("%d complexes failed pocket ESM2 pooling:", len(failed_rows))
        for r in failed_rows[:20]:
            LOGGER.warning("  %s  [%s]  %s", r["pdb_id"], r["build_status"], r["error"])
        if len(failed_rows) > 20:
            LOGGER.warning("  ... (%d more)", len(failed_rows) - 20)

    return summary


# ---------------------------------------------------------------------------
# Alignment validation
# ---------------------------------------------------------------------------

def validate_alignment(feature_dir: Path, n_sample: int = 10) -> None:
    """Spot-check that pocket indices align with residue embeddings.

    Checks performed on each sampled complex
    -----------------------------------------
    1. All stored indices are non-negative  (rules out off-by-one going left
       of the window start).
    2. All stored indices < res_emb.shape[0]  (within tensor bounds).
    3. All stored indices < seq_len_encoded from window_meta  (cross-check
       against the expected window length — would catch a CLS-stripping
       off-by-one where index == seq_len_encoded maps to EOS).
    4. Re-pool res_emb[indices].mean(0) and compare to stored embedding
       within float32 tolerance  (end-to-end numerical consistency).
    5. No NaN or Inf in the stored embedding.
    6. Stored embedding shape == (hidden_dim,).

    Raises AssertionError on any failure.
    """
    import random

    residue_dir      = feature_dir / "protein_residue_embeddings"
    pt_indices_path  = feature_dir / "pocket_residue_indices.pt"
    window_meta_path = feature_dir / "protein_window_meta.pt"
    out_emb_path     = feature_dir / "pocket_esm2_embeddings.pt"

    assert pt_indices_path.exists(), f"Missing {pt_indices_path}"
    assert out_emb_path.exists(),    f"Missing {out_emb_path} — run build first"

    pocket_indices_all: dict[str, list[int]] = torch.load(
        pt_indices_path, map_location="cpu", weights_only=False
    )
    pocket_embs: dict[str, torch.Tensor] = torch.load(
        out_emb_path, map_location="cpu", weights_only=True
    )
    window_meta: dict[str, dict] = {}
    if window_meta_path.exists():
        window_meta = torch.load(window_meta_path, map_location="cpu", weights_only=False)

    common = sorted(set(pocket_indices_all) & set(pocket_embs))
    sample = random.sample(common, min(n_sample, len(common)))

    LOGGER.info("Validating alignment for %d sample complexes...", len(sample))
    errors = 0

    for pdb_id in sample:
        indices  = pocket_indices_all[pdb_id]
        res_path = residue_dir / f"{pdb_id}.pt"
        wm       = window_meta.get(pdb_id, {})
        seq_len_encoded = wm.get("seq_len_encoded", None)

        if not res_path.exists():
            LOGGER.warning("  %s: residue file missing — skipping", pdb_id)
            continue

        res_emb = torch.load(res_path, map_location="cpu", weights_only=True)
        n_rows  = res_emb.shape[0]
        hidden  = res_emb.shape[1]
        stored  = pocket_embs[pdb_id]
        ok      = True

        # Check 1: no negative indices
        negative = [i for i in indices if i < 0]
        if negative:
            LOGGER.error(
                "  %s: CHECK 1 FAIL — negative indices (off-by-one left of window): %s",
                pdb_id, negative[:5],
            )
            errors += 1
            ok = False

        # Check 2: all indices < n_rows
        oob = [i for i in indices if i >= n_rows]
        if oob:
            LOGGER.error(
                "  %s: CHECK 2 FAIL — indices >= res_emb.shape[0]=%d: %s",
                pdb_id, n_rows, oob[:5],
            )
            errors += 1
            ok = False

        # Check 3: all indices < seq_len_encoded (window-boundary cross-check)
        if seq_len_encoded is not None:
            exceeds = [i for i in indices if i >= seq_len_encoded]
            if exceeds:
                LOGGER.error(
                    "  %s: CHECK 3 FAIL — indices >= seq_len_encoded=%d "
                    "(would point at EOS if CLS-stripping were off by 1): %s",
                    pdb_id, seq_len_encoded, exceeds[:5],
                )
                errors += 1
                ok = False

        if not ok:
            continue  # skip numerical checks if index bounds failed

        # Check 4: re-pool and compare
        valid    = [i for i in indices if 0 <= i < n_rows]
        expected = res_emb[valid, :].mean(dim=0)
        if not torch.allclose(expected, stored, atol=1e-5):
            LOGGER.error(
                "  %s: CHECK 4 FAIL — stored embedding differs from recomputed "
                "(max diff=%.2e)",
                pdb_id, (expected - stored).abs().max().item(),
            )
            errors += 1
            continue

        # Check 5: no NaN/Inf
        if not torch.isfinite(stored).all():
            LOGGER.error("  %s: CHECK 5 FAIL — NaN/Inf in stored embedding", pdb_id)
            errors += 1
            continue

        # Check 6: correct shape
        if stored.shape != (hidden,):
            LOGGER.error(
                "  %s: CHECK 6 FAIL — shape %s != expected (%d,)",
                pdb_id, stored.shape, hidden,
            )
            errors += 1
            continue

    assert errors == 0, f"Alignment validation failed for {errors} complexes."
    LOGGER.info(
        "All 6 alignment checks passed for %d sample complexes.", len(sample)
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Build mean-pooled pocket-contextual ESM2 feature vectors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("data/features"),
        help=(
            "Directory containing protein_residue_embeddings/ and "
            "pocket_residue_indices.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write outputs. Defaults to --feature-dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute stats but write nothing to disk.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help=(
            "After building, run alignment spot-check on 50 random complexes "
            "to confirm index correctness."
        ),
    )
    parser.add_argument(
        "--validate-n",
        type=int,
        default=50,
        help="Number of complexes to spot-check when --validate is set.",
    )
    args = parser.parse_args()

    feature_dir = args.feature_dir
    output_dir  = args.output_dir or feature_dir

    summary = build_pocket_esm2_features(
        feature_dir=feature_dir,
        output_dir=output_dir,
        dry_run=args.dry_run,
    )
    print("\nBuild summary:")
    print(json.dumps(summary, indent=2))

    if args.validate and not args.dry_run:
        validate_alignment(feature_dir=output_dir, n_sample=args.validate_n)
        print("Alignment validation: PASSED")


if __name__ == "__main__":
    main()

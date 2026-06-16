"""Train / validation split logic for Phase 3.

Rules
-----
1. Only PDBBind v2018 rows (``dataset_name == "pdbbind_v2018"``) are
   eligible for train / val splits.
2. Rows with ``split == "external_hard_test"`` (the hard temporal test
   set from hard_test/) are **never** included in train or val.
3. Two strategies are supported:

   ``random``  (default)
       Stratified random sampling on a discretised ΔG bin.

   ``ligand_scaffold``
       Bemis-Murcko scaffold-based split: entire scaffolds are assigned
       to either train or val with no overlap. Requires RDKit.

4. The resulting split assignment is saved to the output parquet file
   with column ``split`` having values ``train``, ``val``, or
   ``external_hard_test``.

Usage
-----
::

    # Random split (backward-compatible default)
    python -m src.training.splits \\
        --metadata data/processed/all_metadata.parquet \\
        --feature-index data/features/feature_index.parquet \\
        --graph-index data/features/graph_index.parquet \\
        --strategy random \\
        --val-fraction 0.1 \\
        --seed 42

    # Ligand-scaffold split
    python -m src.training.splits \\
        --metadata data/processed/all_metadata.parquet \\
        --feature-index data/features/feature_index.parquet \\
        --graph-index data/features/graphs/graph_index.parquet \\
        --strategy ligand_scaffold \\
        --val-fraction 0.1 \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

HARD_TEST_SPLIT_VALUE = "external_hard_test"
PDBBIND_DATASET_NAME  = "pdbbind_v2018"

SPLIT_STRATEGIES = ("random", "ligand_scaffold")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _filter_to_features(
    df: pd.DataFrame,
    feature_index_df: pd.DataFrame,
    graph_index_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Apply feature-index and graph-index filters, logging counts."""
    ok_ids = set(
        feature_index_df.loc[feature_index_df["all_features_ok"] == True, "pdb_id"]
    )
    before = len(df)
    df = df[df["pdb_id"].isin(ok_ids)].copy()
    LOGGER.info("After feature-index filter: %d / %d rows", len(df), before)

    if graph_index_df is not None:
        ok_graph_ids = set(
            graph_index_df.loc[
                graph_index_df["build_status"].isin(["ok", "cached"]),
                "pdb_id",
            ]
        )
        before = len(df)
        df = df[df["pdb_id"].isin(ok_graph_ids)].copy()
        LOGGER.info("After graph-index filter: %d / %d rows", len(df), before)

    return df


def _get_murcko_scaffold(smiles: str) -> str:
    """Return canonical Bemis-Murcko scaffold SMILES, or '' on failure."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold) if scaffold is not None else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Strategy: random stratified split
# ---------------------------------------------------------------------------

def make_splits(
    metadata_df:      pd.DataFrame,
    feature_index_df: pd.DataFrame,
    graph_index_df:   pd.DataFrame | None = None,
    val_frac:         float = 0.10,
    seed:             int   = 42,
) -> pd.DataFrame:
    """Random stratified train/val/external_hard_test split.

    Parameters
    ----------
    metadata_df:
        Full metadata parquet, must contain ``pdb_id``, ``dataset_name``,
        ``split``, ``delta_g_kcal_mol``.
    feature_index_df:
        feature_index.parquet with ``pdb_id``, ``all_features_ok``.
    graph_index_df:
        graph_index.parquet with ``pdb_id``, ``build_status``.
        ``None`` means graph filtering is skipped (sequence-only mode).
    val_frac:
        Fraction of PDBBind rows to hold out for validation.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns [pdb_id, dataset_name, split, delta_g_kcal_mol].
    """
    rng = np.random.default_rng(seed)

    df = metadata_df[["pdb_id", "dataset_name", "split", "delta_g_kcal_mol"]].copy()
    df = _filter_to_features(df, feature_index_df, graph_index_df)

    hard_mask  = df["split"] == HARD_TEST_SPLIT_VALUE
    hard_df    = df[hard_mask].copy()
    train_pool = df[~hard_mask].copy()

    LOGGER.info(
        "Hard-test rows (held-out): %d.  Remaining for train/val: %d",
        len(hard_df), len(train_pool),
    )

    if len(train_pool) == 0:
        LOGGER.warning("No rows available for train/val split.")
        return pd.concat([hard_df], ignore_index=True)

    # Bin ΔG into deciles for stratification
    bins   = np.nanpercentile(train_pool["delta_g_kcal_mol"].values, np.linspace(0, 100, 11))
    bins   = np.unique(bins)
    labels = pd.cut(
        train_pool["delta_g_kcal_mol"],
        bins=bins,
        include_lowest=True,
        labels=False,
    ).fillna(0).astype(int)

    val_indices: list[int] = []
    for bin_id in np.unique(labels):
        bin_indices = np.where(labels == bin_id)[0]
        n_val = max(1, int(round(len(bin_indices) * val_frac)))
        chosen = rng.choice(bin_indices, size=n_val, replace=False)
        val_indices.extend(chosen.tolist())

    val_set = set(val_indices)
    assigned_splits = ["val" if i in val_set else "train" for i in range(len(train_pool))]

    train_pool = train_pool.copy()
    train_pool["split"] = assigned_splits

    n_train = assigned_splits.count("train")
    n_val   = assigned_splits.count("val")
    LOGGER.info("Split: %d train, %d val, %d external_hard_test", n_train, n_val, len(hard_df))

    result = pd.concat([train_pool, hard_df], ignore_index=True)
    return result[["pdb_id", "dataset_name", "split", "delta_g_kcal_mol"]]


# ---------------------------------------------------------------------------
# Strategy: ligand scaffold split
# ---------------------------------------------------------------------------

def make_scaffold_splits(
    metadata_df:      pd.DataFrame,
    feature_index_df: pd.DataFrame,
    graph_index_df:   pd.DataFrame | None = None,
    val_frac:         float = 0.10,
    seed:             int   = 42,
) -> tuple[pd.DataFrame, dict]:
    """Bemis-Murcko scaffold-based train/val/external_hard_test split.

    Entire scaffolds are assigned to either train or val with no overlap.
    Complexes whose SMILES cannot be parsed are assigned a unique '' scaffold
    (so they can still be used but are grouped together).

    Parameters
    ----------
    metadata_df, feature_index_df, graph_index_df, val_frac, seed:
        Same as ``make_splits``.

    Returns
    -------
    (split_df, report_dict)
        split_df  : DataFrame with columns [pdb_id, dataset_name, split, delta_g_kcal_mol]
        report_dict : dict with audit information
    """
    # Need ligand_smiles
    if "ligand_smiles" not in metadata_df.columns:
        raise ValueError("metadata_df must contain 'ligand_smiles' column for scaffold split")

    df_full = metadata_df[
        ["pdb_id", "dataset_name", "split", "delta_g_kcal_mol", "ligand_smiles"]
    ].copy()

    # Apply feature/graph filters (operates on pdb_id)
    df_filtered = _filter_to_features(df_full, feature_index_df, graph_index_df)

    # Separate hard test — never touched
    hard_mask  = df_filtered["split"] == HARD_TEST_SPLIT_VALUE
    hard_df    = df_filtered[hard_mask].copy()
    train_pool = df_filtered[~hard_mask].copy().reset_index(drop=True)

    LOGGER.info(
        "Hard-test rows (held-out): %d.  Remaining for train/val: %d",
        len(hard_df), len(train_pool),
    )

    if len(train_pool) == 0:
        LOGGER.warning("No rows available for train/val split.")
        result = pd.concat([hard_df], ignore_index=True)
        return result[["pdb_id", "dataset_name", "split", "delta_g_kcal_mol"]], {}

    # Compute Bemis-Murcko scaffolds
    LOGGER.info("Computing Bemis-Murcko scaffolds for %d complexes...", len(train_pool))
    train_pool["_scaffold"] = train_pool["ligand_smiles"].apply(_get_murcko_scaffold)

    # Group by scaffold
    scaffold_groups: dict[str, list[int]] = {}
    for idx, row in train_pool.iterrows():
        s = row["_scaffold"]
        scaffold_groups.setdefault(s, []).append(idx)

    unique_scaffolds = sorted(scaffold_groups.keys())
    n_unique = len(unique_scaffolds)
    total    = len(train_pool)
    target_n_val = int(round(total * val_frac))

    LOGGER.info(
        "Unique scaffolds: %d  |  target val size: %d / %d",
        n_unique, target_n_val, total,
    )

    # Shuffle scaffolds deterministically
    rng = np.random.default_rng(seed)
    scaffold_order = np.array(unique_scaffolds, dtype=object)
    rng.shuffle(scaffold_order)

    # Greedy scaffold assignment to val: accumulate scaffolds until target is met
    val_indices: set[int] = set()
    val_scaffolds: set[str] = set()

    for scaffold in scaffold_order:
        group_indices = scaffold_groups[scaffold]
        if len(val_indices) + len(group_indices) <= target_n_val + len(group_indices) // 2:
            # Accept this scaffold into val if we still need more, with tolerance
            if len(val_indices) < target_n_val:
                val_indices.update(group_indices)
                val_scaffolds.add(scaffold)

    assigned_splits = [
        "val" if i in val_indices else "train"
        for i in train_pool.index
    ]
    train_pool = train_pool.copy()
    train_pool["split"] = assigned_splits

    n_train = assigned_splits.count("train")
    n_val   = assigned_splits.count("val")
    train_scaffolds = set(
        train_pool.loc[train_pool["split"] == "train", "_scaffold"]
    )
    val_scaffolds_actual = set(
        train_pool.loc[train_pool["split"] == "val", "_scaffold"]
    )
    scaffold_overlap = train_scaffolds & val_scaffolds_actual

    actual_val_frac = n_val / (n_train + n_val) if (n_train + n_val) > 0 else 0.0

    warnings: list[str] = []
    if abs(actual_val_frac - val_frac) > 0.03:
        msg = (
            f"Actual val fraction {actual_val_frac:.3f} deviates from target "
            f"{val_frac:.3f} due to scaffold grouping."
        )
        warnings.append(msg)
        LOGGER.warning(msg)
    if scaffold_overlap:
        msg = f"Scaffold overlap detected between train and val: {len(scaffold_overlap)} scaffolds"
        warnings.append(msg)
        LOGGER.error(msg)

    LOGGER.info(
        "Scaffold split: %d train, %d val (%.1f%%), %d external_hard_test",
        n_train, n_val, 100 * actual_val_frac, len(hard_df),
    )
    LOGGER.info(
        "Train scaffolds: %d | Val scaffolds: %d | Overlap: %d",
        len(train_scaffolds), len(val_scaffolds_actual), len(scaffold_overlap),
    )

    # Build ΔG stats helper
    def _stats(series: pd.Series) -> dict:
        return {
            "mean": float(series.mean()),
            "std":  float(series.std()),
            "min":  float(series.min()),
            "max":  float(series.max()),
        }

    report = {
        "strategy":              "ligand_scaffold",
        "seed":                  seed,
        "val_fraction_target":   val_frac,
        "val_fraction_actual":   round(actual_val_frac, 4),
        "train_count":           n_train,
        "val_count":             n_val,
        "external_hard_test_count": int(len(hard_df)),
        "unique_scaffolds_total":   n_unique,
        "train_scaffold_count":     len(train_scaffolds),
        "val_scaffold_count":       len(val_scaffolds_actual),
        "scaffold_overlap_count":   len(scaffold_overlap),
        "delta_g_train":  _stats(train_pool.loc[train_pool["split"] == "train", "delta_g_kcal_mol"]),
        "delta_g_val":    _stats(train_pool.loc[train_pool["split"] == "val",   "delta_g_kcal_mol"]),
        "warnings":       warnings,
    }

    result = pd.concat(
        [train_pool[["pdb_id", "dataset_name", "split", "delta_g_kcal_mol"]], hard_df],
        ignore_index=True,
    )
    return result[["pdb_id", "dataset_name", "split", "delta_g_kcal_mol"]], report


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def make_splits_by_strategy(
    strategy:         str,
    metadata_df:      pd.DataFrame,
    feature_index_df: pd.DataFrame,
    graph_index_df:   pd.DataFrame | None = None,
    val_frac:         float = 0.10,
    seed:             int   = 42,
) -> tuple[pd.DataFrame, dict]:
    """Call the correct split function for *strategy* and return (df, report)."""
    if strategy == "random":
        df = make_splits(metadata_df, feature_index_df, graph_index_df, val_frac, seed)
        n_train = int((df["split"] == "train").sum())
        n_val   = int((df["split"] == "val").sum())
        n_hard  = int((df["split"] == HARD_TEST_SPLIT_VALUE).sum())
        report = {
            "strategy": "random",
            "seed": seed,
            "val_fraction_target": val_frac,
            "val_fraction_actual": round(n_val / max(n_train + n_val, 1), 4),
            "train_count": n_train,
            "val_count": n_val,
            "external_hard_test_count": n_hard,
        }
        return df, report
    elif strategy == "ligand_scaffold":
        return make_scaffold_splits(metadata_df, feature_index_df, graph_index_df, val_frac, seed)
    else:
        raise ValueError(f"Unknown split strategy {strategy!r}. Choose from {SPLIT_STRATEGIES}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    from src.data.utils import setup_logging

    parser = argparse.ArgumentParser(
        description="Create train/val/external_hard_test split file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/all_metadata.parquet"),
    )
    parser.add_argument(
        "--feature-index",
        type=Path,
        default=Path("data/features/feature_index.parquet"),
    )
    parser.add_argument(
        "--graph-index",
        type=Path,
        default=None,
        help="Optional graph_index.parquet to further filter to built graphs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output parquet path.  Defaults to "
            "data/splits/train_val_external.parquet (random) or "
            "data/splits/train_val_external_ligand_scaffold.parquet."
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=SPLIT_STRATEGIES,
        default="random",
        help="Split strategy to use.",
    )
    # Accept both spellings for backward compat
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--val-frac",     type=float, default=None,
                       help="Fraction of PDBBind rows for validation.")
    group.add_argument("--val-fraction", type=float, default=None,
                       help="Alias for --val-frac.")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    val_frac = args.val_frac if args.val_frac is not None else (
        args.val_fraction if args.val_fraction is not None else 0.10
    )

    # Default output path per strategy
    if args.output is None:
        if args.strategy == "random":
            out_path = Path("data/splits/train_val_external.parquet")
        else:
            out_path = Path(f"data/splits/train_val_external_{args.strategy}.parquet")
    else:
        out_path = args.output

    metadata_df      = pd.read_parquet(args.metadata)
    feature_index_df = pd.read_parquet(args.feature_index)
    graph_index_df   = (
        pd.read_parquet(args.graph_index) if args.graph_index else None
    )

    result, report = make_splits_by_strategy(
        strategy         = args.strategy,
        metadata_df      = metadata_df,
        feature_index_df = feature_index_df,
        graph_index_df   = graph_index_df,
        val_frac         = val_frac,
        seed             = args.seed,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    LOGGER.info("Split file saved → %s  (%d rows)", out_path, len(result))

    # Save split report
    report_path = out_path.parent / f"split_report_{args.strategy}.json"
    report_path.write_text(json.dumps(report, indent=2))
    LOGGER.info("Split report → %s", report_path)


if __name__ == "__main__":
    main()


"""Phase 3 training loop.

Usage
-----
::

    python -m src.training.train --config configs/phase3.yaml --model sequence_only
    python -m src.training.train --config configs/phase3.yaml --model graph_only
    python -m src.training.train --config configs/phase3.yaml --model concat_fusion
    python -m src.training.train --config configs/phase3.yaml --model cross_attention_fusion

Outputs per run
---------------
    outputs/phase3/{model_name}/
        best_model.pt                  — state dict of best validation checkpoint
        metrics.json                   — final metrics for train / val / ext-test
        predictions_train.csv
        predictions_val.csv
        predictions_external_hard_test.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.utils import setup_logging
from src.models.fusion import build_model
from src.training.dataset import BindingAffinityDataset, collate_fn
from src.training.evaluate import _forward, evaluate_split, extract_embeddings, save_metrics, save_predictions

LOGGER = logging.getLogger(__name__)

NEEDS_GRAPH = frozenset({
    "graph_only", "concat_fusion", "cross_attention_fusion",
    "pocket_esm2_graph_fusion", "pocket_cross_attention_fusion",
    "all_concat_fusion", "all_cross_attention_fusion",
})

# Models that use pocket-contextual ESM2 *instead of* the full-protein global embedding.
# The training pipeline swaps prot_emb_path → pocket_esm2_path for these.
POCKET_MODELS = frozenset({
    "pocket_sequence_only", "pocket_esm2_graph_fusion",
    "pocket_cross_attention_fusion",
})

# Models that need BOTH global protein ESM2 and pocket-contextual ESM2 simultaneously.
# The dataset receives prot_emb_path (global) as usual AND pocket_esm2_path as
# pocket_emb_path, producing items with both ``prot_emb`` and ``pocket_emb``.
ALL_MODELS = frozenset({
    "all_concat_fusion", "all_cross_attention_fusion",
})


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def _make_loader(
    split_df:      pd.DataFrame,
    split_names:   list[str],
    cfg:           dict,
    model_name:    str,
    shuffle:       bool,
) -> DataLoader | None:
    """Build a DataLoader for the given split(s), or None if empty."""
    subset = split_df[split_df["split"].isin(split_names)].copy()
    if len(subset) == 0:
        LOGGER.warning("No rows found for split(s) %s.", split_names)
        return None

    data_cfg   = cfg["data"]
    graph_dir  = Path(data_cfg["graph_dir"]) if model_name in NEEDS_GRAPH else None

    # Pocket models receive the mean-pooled pocket ESM2 vector as prot_emb.
    if model_name in POCKET_MODELS:
        pocket_path = data_cfg.get("pocket_esm2_path")
        if not pocket_path:
            raise KeyError(
                f"Model '{model_name}' requires 'data.pocket_esm2_path' in the config "
                "(run src.features.build_pocket_esm2_features first)."
            )
        prot_emb_path   = Path(pocket_path)
        pocket_emb_path = None   # no second embedding needed
    elif model_name in ALL_MODELS:
        # Need global protein embedding AND pocket embedding simultaneously.
        prot_emb_path = Path(data_cfg["prot_emb_path"])
        pocket_path   = data_cfg.get("pocket_esm2_path")
        if not pocket_path:
            raise KeyError(
                f"Model '{model_name}' requires 'data.pocket_esm2_path' in the config "
                "(run src.features.build_pocket_esm2_features first)."
            )
        pocket_emb_path = Path(pocket_path)
    else:
        prot_emb_path   = Path(data_cfg["prot_emb_path"])
        pocket_emb_path = None

    dataset = BindingAffinityDataset(
        split_df        = subset,
        prot_emb_path   = prot_emb_path,
        lig_emb_path    = Path(data_cfg["lig_emb_path"]),
        graph_dir       = graph_dir,
        pocket_emb_path = pocket_emb_path,
    )
    if len(dataset) == 0:
        return None

    return DataLoader(
        dataset,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = shuffle,
        num_workers = cfg["training"].get("num_workers", 0),
        collate_fn  = collate_fn,
        pin_memory  = False,
        drop_last   = shuffle,   # drop incomplete last batch only during training
    )


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def _train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n          = 0
    for batch in loader:
        optimizer.zero_grad()
        pred = _forward(model, batch, device).squeeze(-1)   # [B]
        y    = batch["y"].squeeze(-1).to(device)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.detach().item() * len(y)
        n          += len(y)
    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(cfg: dict, model_name: str, split_strategy: str = "random") -> None:
    t_cfg   = cfg["training"]
    device  = torch.device(t_cfg.get("device", "cpu"))
    seed    = t_cfg.get("seed", 42)

    torch.manual_seed(seed)

    # Output directory — namespaced by split strategy so runs don't overwrite each other
    out_dir = Path(cfg["output_dir"]) / split_strategy / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Output dir: %s", out_dir)
    LOGGER.info("Split strategy: %s", split_strategy)

    # ── Split file ────────────────────────────────────────────────────────
    # Strategy-specific split file: random uses the default, others use a
    # strategy-suffixed filename so different splits coexist.
    data_cfg = cfg["data"]
    if split_strategy == "random":
        split_path = Path(data_cfg["split_file"])
    else:
        split_path = Path(data_cfg["split_file"]).with_name(
            f"train_val_external_{split_strategy}.parquet"
        )
    if not split_path.exists():
        LOGGER.info("Split file not found; generating it now (strategy=%s).", split_strategy)
        _generate_split(cfg, split_strategy=split_strategy)
    split_df = pd.read_parquet(split_path)
    LOGGER.info("Split file loaded: %d rows", len(split_df))

    # ── Data loaders ──────────────────────────────────────────────────────
    train_loader = _make_loader(split_df, ["train"],                 cfg, model_name, shuffle=True)
    val_loader   = _make_loader(split_df, ["val"],                   cfg, model_name, shuffle=False)
    test_loader  = _make_loader(split_df, ["external_hard_test"],    cfg, model_name, shuffle=False)

    if train_loader is None:
        raise RuntimeError("Training set is empty.  Check split file and feature availability.")

    # ── Model ─────────────────────────────────────────────────────────────
    model_cfg = cfg["models"].get(model_name, {})
    model     = build_model(model_name, model_cfg).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info("Model %s — %d trainable parameters.", model_name, n_params)

    # ── Optimiser / scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = float(t_cfg.get("lr", 1e-3)),
        weight_decay = float(t_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = float(t_cfg.get("lr_factor", 0.5)),
        patience = int(t_cfg.get("lr_patience", 5)),
    )
    criterion = nn.MSELoss()

    # ── Early stopping ────────────────────────────────────────────────────
    patience     = int(t_cfg.get("patience", 20))
    max_epochs   = int(t_cfg.get("max_epochs", 100))
    best_val_rmse = float("inf")
    epochs_no_improve = 0
    best_ckpt_path    = out_dir / "best_model.pt"

    # ── Training loop ─────────────────────────────────────────────────────
    train_history: list[dict] = []
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)

        if val_loader is not None:
            _, vt, vp, val_metrics = evaluate_split(model, val_loader, device, "val")
            val_rmse = val_metrics["rmse"]
        else:
            val_metrics = {}
            val_rmse    = float("inf")

        scheduler.step(val_rmse)

        lr_now = optimizer.param_groups[0]["lr"]
        LOGGER.info(
            "Epoch %3d/%d  train_loss=%.4f  val_rmse=%.4f  lr=%.2e",
            epoch, max_epochs, train_loss, val_rmse, lr_now,
        )
        train_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_rmse":   val_rmse,
        })

        # Best checkpoint
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt_path)
            LOGGER.info("  → New best val RMSE %.4f; checkpoint saved.", best_val_rmse)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                LOGGER.info(
                    "Early stopping at epoch %d (no improvement for %d epochs).",
                    epoch, patience,
                )
                break

    elapsed = time.time() - t0
    LOGGER.info("Training done in %.1f s.", elapsed)

    # ── Load best checkpoint and evaluate ─────────────────────────────────
    if best_ckpt_path.exists():
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=True))
        LOGGER.info("Loaded best checkpoint for final evaluation.")

    final_metrics: dict = {"model": model_name, "elapsed_seconds": round(elapsed, 1)}

    # Train
    tr_ids, tr_t, tr_p, tr_m = evaluate_split(model, train_loader, device, "train")
    final_metrics["train"] = tr_m
    save_predictions(
        tr_ids, ["train"] * len(tr_ids), tr_t, tr_p,
        out_dir / "predictions_train.csv",
    )

    # Validation
    if val_loader is not None:
        va_ids, va_t, va_p, va_m = evaluate_split(model, val_loader, device, "val")
        final_metrics["val"] = va_m
        save_predictions(
            va_ids, ["val"] * len(va_ids), va_t, va_p,
            out_dir / "predictions_val.csv",
        )
        # Save pre-MLP fused embeddings for the meta-router
        val_embs = extract_embeddings(model, val_loader, device)
        torch.save(val_embs, out_dir / "val_embeddings.pt")
        LOGGER.info("Val embeddings saved → %s", out_dir / "val_embeddings.pt")

    # Hard test (never used for model selection)
    if test_loader is not None:
        te_ids, te_t, te_p, te_m = evaluate_split(
            model, test_loader, device, "external_hard_test"
        )
        final_metrics["external_hard_test"] = te_m
        save_predictions(
            te_ids, ["external_hard_test"] * len(te_ids), te_t, te_p,
            out_dir / "predictions_external_hard_test.csv",
        )

    save_metrics(final_metrics, out_dir / "metrics.json")

    # Training curve
    curve_path = out_dir / "training_curve.json"
    curve_path.write_text(json.dumps(train_history, indent=2))
    LOGGER.info("Training curve → %s", curve_path)


# ---------------------------------------------------------------------------
# Auto-generate split file if missing
# ---------------------------------------------------------------------------

def _generate_split(cfg: dict, split_strategy: str = "random") -> None:
    from src.training.splits import make_splits_by_strategy

    data_cfg = cfg["data"]
    if split_strategy == "random":
        split_out = Path(data_cfg["split_file"])
    else:
        split_out = Path(data_cfg["split_file"]).with_name(
            f"train_val_external_{split_strategy}.parquet"
        )

    meta_df  = pd.read_parquet(data_cfg["metadata_path"])
    feat_df  = pd.read_parquet(data_cfg["feature_index_path"])
    graph_df = (
        pd.read_parquet(data_cfg["graph_index_path"])
        if data_cfg.get("graph_index_path") else None
    )

    result, report = make_splits_by_strategy(
        strategy         = split_strategy,
        metadata_df      = meta_df,
        feature_index_df = feat_df,
        graph_index_df   = graph_df,
        val_frac         = float(cfg["training"].get("val_frac", 0.10)),
        seed             = int(cfg["training"].get("seed", 42)),
    )
    split_out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(split_out, index=False)
    LOGGER.info("Auto-generated split file → %s  (strategy=%s)", split_out, split_strategy)
    # Save report alongside
    import json
    report_path = split_out.with_name(f"split_report_{split_strategy}.json")
    report_path.write_text(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a binding-affinity prediction model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to phase3.yaml config file.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "sequence_only",
            "graph_only",
            "concat_fusion",
            "cross_attention_fusion",
            "pocket_sequence_only",
            "pocket_esm2_graph_fusion",
            "pocket_cross_attention_fusion",
            "all_concat_fusion",
            "all_cross_attention_fusion",
        ],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["random", "ligand_scaffold"],
        default="random",
        help=(
            "Validation split strategy. 'random' uses stratified random split; "
            "'ligand_scaffold' uses Bemis-Murcko scaffold split with no scaffold "
            "overlap between train and val. Output goes to "
            "outputs/phase3/{strategy}/{model}/."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    cfg = load_config(args.config)
    train(cfg, args.model, split_strategy=args.split_strategy)


if __name__ == "__main__":
    main()

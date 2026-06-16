"""Evaluation utilities for Phase 3 models.

``evaluate_split`` runs inference over a DataLoader and returns
predictions, true labels, and metrics.

``save_predictions`` writes a CSV with columns:
    pdb_id, split, y_true, y_pred

``save_metrics`` writes a JSON file with the metric dict.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.metrics import compute_metrics

LOGGER = logging.getLogger(__name__)


def _forward(
    model:  nn.Module,
    batch:  dict,
    device: torch.device,
) -> torch.Tensor:
    """Route a collated batch through any model variant.

    Moves required tensors to ``device`` before calling ``model.forward``.
    Extra keys (metadata strings, Python lists) are silently ignored.
    """
    # Sequence embeddings — always present
    kwargs: dict[str, torch.Tensor] = {
        "prot_emb": batch["prot_emb"].to(device),
        "lig_emb":  batch["lig_emb"].to(device),
    }

    # Pocket ESM2 — present for all-features fusion models
    if "pocket_emb" in batch:
        kwargs["pocket_emb"] = batch["pocket_emb"].to(device)

    # Graph tensors — present when graph_dir was set in the dataset
    if "x" in batch:
        kwargs["x"]          = batch["x"].to(device)
        kwargs["edge_index"] = batch["edge_index"].to(device)
        kwargs["batch"]      = batch["batch"].to(device)
        kwargs["edge_attr"]  = batch["edge_attr"].to(device)

    return model(**kwargs)   # [B, 1]


@torch.no_grad()
def evaluate_split(
    model:    nn.Module,
    loader:   DataLoader,
    device:   torch.device,
    split_name: str = "",
) -> tuple[list[str], list[float], list[float], dict[str, float]]:
    """Run model in eval mode over *loader* and collect predictions.

    Returns
    -------
    pdb_ids  : list[str]
    y_trues  : list[float]
    y_preds  : list[float]
    metrics  : dict with rmse, mae, pearson_r, spearman_r
    """
    model.eval()
    pdb_ids:  list[str]   = []
    y_trues:  list[float] = []
    y_preds:  list[float] = []

    for batch in loader:
        preds = _forward(model, batch, device).squeeze(-1)  # [B]
        yt    = batch["y"].squeeze(-1).tolist()
        yp    = preds.cpu().tolist()

        pdb_ids.extend(batch["pdb_id"])
        y_trues.extend(yt)
        y_preds.extend(yp)

    metrics = compute_metrics(y_trues, y_preds)
    tag = f" [{split_name}]" if split_name else ""
    LOGGER.info(
        "%sRMSE=%.4f  MAE=%.4f  Pearson=%.4f  Spearman=%.4f",
        tag, metrics["rmse"], metrics["mae"],
        metrics["pearson_r"], metrics["spearman_r"],
    )
    return pdb_ids, y_trues, y_preds, metrics


def save_predictions(
    pdb_ids: list[str],
    splits:  list[str] | None,
    y_trues: list[float],
    y_preds: list[float],
    path:    Path,
) -> None:
    """Write predictions CSV."""
    df = pd.DataFrame({
        "pdb_id": pdb_ids,
        "split":  splits if splits is not None else [""] * len(pdb_ids),
        "y_true": y_trues,
        "y_pred": y_preds,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOGGER.info("Predictions saved → %s  (%d rows)", path, len(df))


def save_metrics(metrics: dict, path: Path) -> None:
    """Write metrics JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    LOGGER.info("Metrics saved → %s", path)


@torch.no_grad()
def extract_embeddings(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Run model.encode() over *loader* and return ``{pdb_id: embedding}``.

    Each embedding is the pre-MLP fused representation produced by
    ``model.encode()``.  If the model has no ``encode`` method (shouldn't
    happen with our registry) we fall back to the forward pass and return
    the scalar prediction as a 1-D tensor instead.

    Returns
    -------
    dict mapping pdb_id → Tensor [D]  (on CPU, detached)
    """
    if not hasattr(model, "encode"):
        raise AttributeError(
            f"{type(model).__name__} has no encode() method. "
            "Add encode() to the model class before calling extract_embeddings()."
        )
    model.eval()
    result: dict[str, torch.Tensor] = {}

    for batch in loader:
        kwargs: dict[str, torch.Tensor] = {
            "prot_emb": batch["prot_emb"].to(device),
            "lig_emb":  batch["lig_emb"].to(device),
        }
        if "pocket_emb" in batch:
            kwargs["pocket_emb"] = batch["pocket_emb"].to(device)
        if "x" in batch:
            kwargs["x"]          = batch["x"].to(device)
            kwargs["edge_index"] = batch["edge_index"].to(device)
            kwargs["batch"]      = batch["batch"].to(device)
            kwargs["edge_attr"]  = batch["edge_attr"].to(device)

        embs = model.encode(**kwargs)  # [B, D]
        for pdb_id, emb in zip(batch["pdb_id"], embs.cpu()):
            result[pdb_id] = emb

    LOGGER.info("Extracted embeddings for %d complexes (dim=%d).", len(result),
                next(iter(result.values())).shape[0] if result else 0)
    return result

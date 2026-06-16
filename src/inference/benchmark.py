from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _candidate_dirs(output_root: Path, model_name: str) -> list[Path]:
    return [
        output_root / model_name,
        output_root / "random" / model_name,
    ]


def _read_metrics(metrics_path: Path, model_name: str) -> pd.DataFrame:
    if not metrics_path.exists():
        return pd.DataFrame()

    with metrics_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if isinstance(raw, dict) and "external_hard_test" in raw and isinstance(raw["external_hard_test"], dict):
        metrics = raw["external_hard_test"]
    elif (
        isinstance(raw, dict)
        and "metrics" in raw
        and isinstance(raw["metrics"], dict)
        and "external_hard_test" in raw["metrics"]
    ):
        metrics = raw["metrics"]["external_hard_test"]
    elif isinstance(raw, dict):
        metrics = raw
    else:
        metrics = {}

    row = {"model": model_name}
    for key in ("rmse", "mae", "pearson_r", "spearman_r", "within_1", "within_2"):
        if key in metrics:
            row[key] = metrics[key]
    row["metrics_file"] = str(metrics_path)
    return pd.DataFrame([row])


def _read_predictions(pred_path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(pred_path)

    lower_to_original = {c.lower(): c for c in df.columns}
    rename = {}

    if "y_true" not in df.columns:
        for candidate in ("true_delta_g", "true", "experimental", "experiment"):
            if candidate in lower_to_original:
                rename[lower_to_original[candidate]] = "y_true"
                break

    if "y_pred" not in df.columns:
        for candidate in ("predicted_delta_g", "pred", "prediction", "predicted"):
            if candidate in lower_to_original:
                rename[lower_to_original[candidate]] = "y_pred"
                break

    if rename:
        df = df.rename(columns=rename)

    df.insert(0, "model", model_name)

    if "y_true" in df.columns and "y_pred" in df.columns:
        yt = pd.to_numeric(df["y_true"], errors="coerce")
        yp = pd.to_numeric(df["y_pred"], errors="coerce")
        df["abs_error"] = (yt - yp).abs()

    df["predictions_file"] = str(pred_path)

    preferred = ["model", "pdb_id", "split", "y_true", "y_pred", "abs_error", "predictions_file"]
    rest = [c for c in df.columns if c not in preferred]
    return df[[c for c in preferred if c in df.columns] + rest]


def load_one_model_hard_test(
    output_root: Path | str,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    output_root = Path(output_root)
    missing = []

    for model_dir in _candidate_dirs(output_root, model_name):
        pred_path = model_dir / "predictions_external_hard_test.csv"
        metrics_path = model_dir / "metrics.json"

        pred_exists = pred_path.exists()
        metrics_exists = metrics_path.exists()

        if pred_exists or metrics_exists:
            metrics_df = _read_metrics(metrics_path, model_name) if metrics_exists else pd.DataFrame()
            pred_df = _read_predictions(pred_path, model_name) if pred_exists else pd.DataFrame()
            if not pred_exists:
                missing.append(str(pred_path))
            if not metrics_exists:
                missing.append(str(metrics_path))
            return metrics_df, pred_df, missing

        missing.append(str(pred_path))
        missing.append(str(metrics_path))

    return pd.DataFrame(), pd.DataFrame(), missing


def _compute_metrics_from_predictions(df: pd.DataFrame) -> dict:
    yt = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    yp = pd.to_numeric(df["y_pred"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    abs_err = np.abs(yp - yt)

    metrics = {
        "n": len(yt),
        "mae": float(abs_err.mean()) if len(yt) else np.nan,
        "rmse": float(np.sqrt(np.mean((yp - yt) ** 2))) if len(yt) else np.nan,
        "within_1": float((abs_err <= 1.0).mean()) if len(yt) else np.nan,
        "within_2": float((abs_err <= 2.0).mean()) if len(yt) else np.nan,
    }

    if len(yt) > 2 and np.std(yt) > 0 and np.std(yp) > 0:
        metrics["pearson_r"] = float(np.corrcoef(yt, yp)[0, 1])
    else:
        metrics["pearson_r"] = np.nan

    return metrics


def make_truth_vs_pred_figure(df: pd.DataFrame, model_name: str):
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError("Prediction CSV must contain y_true and y_pred columns.")

    yt = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    yp = pd.to_numeric(df["y_pred"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]

    metrics = _compute_metrics_from_predictions(pd.DataFrame({"y_true": yt, "y_pred": yp}))

    lo = float(min(np.min(yt), np.min(yp)) - 0.5)
    hi = float(max(np.max(yt), np.max(yp)) + 0.5)
    diag = np.array([lo, hi])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt, yp, s=22, alpha=0.55, linewidths=0)

    ax.plot(diag, diag, color="black", lw=1.4, label="Ideal")
    ax.plot(diag, diag + 1.0, linestyle="--", lw=1.0, alpha=0.85, label="±1 kcal/mol")
    ax.plot(diag, diag - 1.0, linestyle="--", lw=1.0, alpha=0.85)
    ax.plot(diag, diag + 2.0, linestyle=":", lw=1.2, alpha=0.85, label="±2 kcal/mol")
    ax.plot(diag, diag - 2.0, linestyle=":", lw=1.2, alpha=0.85)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Experimental ΔG (kcal/mol)")
    ax.set_ylabel("Predicted ΔG (kcal/mol)")
    ax.set_title(f"{model_name} — external hard test", fontweight="bold")
    ax.legend(loc="lower right", frameon=True)

    text = (
        f"n={metrics['n']}\n"
        f"MAE={metrics['mae']:.3f}\n"
        f"RMSE={metrics['rmse']:.3f}\n"
        f"r={metrics['pearson_r']:.3f}\n"
        f"≤1={metrics['within_1']:.1%}\n"
        f"≤2={metrics['within_2']:.1%}"
    )
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )

    fig.tight_layout()
    return fig


def find_oracle_scatter(project_root: Path | str = ".") -> Path | None:
    root = Path(project_root)

    candidates = [
        root / "scatteroracle.png",
        root / "scatter_oracle.png",
        root / "outputs" / "meta_router_full_descriptors_v2" / "external_hard_test" / "plots" / "scatter_oracle.png",
        root / "outputs" / "meta_router_full_descriptors_v2" / "plots" / "scatter_oracle.png",
        root / "outputs" / "meta_router_full_descriptors" / "external_hard_test" / "plots" / "scatter_oracle.png",
        root / "outputs" / "meta_router_full_descriptors" / "plots" / "scatter_oracle.png",
    ]

    for p in candidates:
        if p.exists():
            return p

    for name in ("scatteroracle.png", "scatter_oracle.png"):
        hits = list(root.rglob(name))
        if hits:
            return hits[0]

    return None

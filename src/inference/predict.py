from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch

from src.inference.cache import CacheManager, hash_file, hash_text
from src.inference.exact_complex import build_exact_complex_graph_from_pair
from src.inference.docking import DockingBox, dock_smiles_with_vina
from src.inference.featurize import ligand_embeddings, protein_embedding
from src.inference.inputs import REQUIRED_SMILES_COL, resolve_sdf_for_row
from src.inference.model_registry import GRAPH_MODELS, load_cfg, load_trained_model, resolve_device


def _get_or_build_graph(
    cache: CacheManager,
    protein_pdb: Path,
    ligand_sdf: Path,
    complex_id: str,
    pocket_cutoff: float,
    contact_cutoff: float,
) -> dict:
    key = hash_text(f"{hash_file(protein_pdb)}|{hash_file(ligand_sdf)}|{pocket_cutoff}|{contact_cutoff}")
    if cache.exists("graphs", key):
        return cache.load("graphs", key)

    graph = build_exact_complex_graph_from_pair(
        protein_pdb=protein_pdb,
        ligand_sdf=ligand_sdf,
        complex_id=complex_id,
        pocket_cutoff=pocket_cutoff,
        contact_cutoff=contact_cutoff,
    )
    cache.save("graphs", key, graph)
    return graph


def _batch_graphs(items: list[dict]) -> dict:
    xs, edge_attrs, edge_indices, batches = [], [], [], []
    prot_embs, lig_embs, row_indices = [], [], []
    offset = 0

    for batch_idx, item in enumerate(items):
        g = item["graph"]
        x = g["x"]
        xs.append(x)
        edge_attrs.append(g["edge_attr"])
        edge_indices.append(g["edge_index"] + offset)
        batches.append(torch.full((x.shape[0],), batch_idx, dtype=torch.long))
        offset += x.shape[0]
        prot_embs.append(item["prot_emb"])
        lig_embs.append(item["lig_emb"])
        row_indices.append(item["row_index"])

    return {
        "prot_emb": torch.stack(prot_embs),
        "lig_emb": torch.stack(lig_embs),
        "x": torch.cat(xs),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr": torch.cat(edge_attrs),
        "batch": torch.cat(batches),
        "row_indices": row_indices,
    }


def _predict_sequence_rows(model, device, prot_emb, lig_emb_by_smiles, df, batch_size):
    preds = {}
    rows = list(df.iterrows())

    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        ligs = [lig_emb_by_smiles[str(row[REQUIRED_SMILES_COL])] for _, row in chunk]
        prot = prot_emb.unsqueeze(0).repeat(len(chunk), 1)

        with torch.no_grad():
            y = model(
                prot_emb=prot.to(device),
                lig_emb=torch.stack(ligs).to(device),
            ).squeeze(-1).detach().cpu().tolist()

        for (idx, _), pred in zip(chunk, y):
            preds[int(idx)] = float(pred)

    return preds


def _predict_graph_rows(model, device, items, batch_size):
    preds = {}

    for i in range(0, len(items), batch_size):
        b = _batch_graphs(items[i:i + batch_size])

        with torch.no_grad():
            y = model(
                prot_emb=b["prot_emb"].to(device),
                lig_emb=b["lig_emb"].to(device),
                x=b["x"].to(device),
                edge_index=b["edge_index"].to(device),
                edge_attr=b["edge_attr"].to(device),
                batch=b["batch"].to(device),
            ).squeeze(-1).detach().cpu().tolist()

        for row_idx, pred in zip(b["row_indices"], y):
            preds[int(row_idx)] = float(pred)

    return preds


def run_screening(
    ligand_df: pd.DataFrame,
    protein_pdb: Path,
    protein_sequence: str,
    requested_model: str,
    config_path: Path,
    cache: CacheManager,
    sdf_paths_by_name: dict[str, Path],
    device_choice: str = "auto",
    batch_size: int = 64,
    pocket_cutoff: float = 8.0,
    contact_cutoff: float = 6.0,
    progress_cb: Callable[[float, str], None] | None = None,
    vina_enabled: bool = False,
    vina_exe_path: Path | None = None,
    vina_box: DockingBox | None = None,
    vina_exhaustiveness: int = 8,
    vina_num_modes: int = 1,
    vina_cpu: int = 0,
    vina_force_redock: bool = False,
    vina_pose_source_tag: str = "vina_manual_box",
) -> pd.DataFrame:
    def progress(frac, msg):
        if progress_cb:
            progress_cb(frac, msg)

    cfg = load_cfg(config_path)
    device = resolve_device(device_choice)

    out = ligand_df.copy()
    out["requested_model"] = requested_model
    out["model_used"] = requested_model
    out["fallback_used"] = False
    out["fallback_reason"] = ""
    out["graph_status"] = "not_required" if requested_model == "sequence_only" else "pending"
    out["sequence_status"] = "pending"
    out["pose_source"] = "not_required" if requested_model == "sequence_only" else "pending"
    out["ligand_sdf_used"] = ""
    out["docking_status"] = "not_run"
    out["docking_error"] = ""
    out["vina_score_kcal_mol"] = pd.NA
    out["box_center_x"] = vina_box.center_x if vina_box else pd.NA
    out["box_center_y"] = vina_box.center_y if vina_box else pd.NA
    out["box_center_z"] = vina_box.center_z if vina_box else pd.NA
    out["box_size_x"] = vina_box.size_x if vina_box else pd.NA
    out["box_size_y"] = vina_box.size_y if vina_box else pd.NA
    out["box_size_z"] = vina_box.size_z if vina_box else pd.NA
    out["box_source"] = vina_pose_source_tag if vina_box else "none"
    out["predicted_deltaG"] = pd.NA

    progress(0.05, "Encoding/caching protein...")
    prot_emb = protein_embedding(protein_sequence, cache, device)

    progress(0.15, "Encoding/caching ligands...")
    lig_embs = ligand_embeddings(out[REQUIRED_SMILES_COL].astype(str).tolist(), cache, device, batch_size)

    sequence_indices, graph_items = [], []

    if requested_model in GRAPH_MODELS:
        docking_root = cache.root / "vina_docking"
        can_dock = bool(vina_enabled and vina_exe_path is not None and vina_box is not None)

        for n, (idx, row) in enumerate(out.iterrows(), start=1):
            sdf = resolve_sdf_for_row(row, sdf_paths_by_name)
            if sdf is not None:
                out.at[idx, "pose_source"] = "uploaded_exact_sdf"
                out.at[idx, "ligand_sdf_used"] = str(sdf)
                out.at[idx, "docking_status"] = "not_run_exact_sdf_available"
            elif can_dock:
                progress(0.20 + 0.30 * (n / max(1, len(out))), f"Docking ligand {n:,}/{len(out):,} with Vina...")
                dock = dock_smiles_with_vina(
                    smiles=str(row[REQUIRED_SMILES_COL]),
                    ligand_id=str(row.get("ligand_id", idx)),
                    protein_pdb=protein_pdb,
                    vina_exe=vina_exe_path,
                    box=vina_box,
                    docking_root=docking_root,
                    exhaustiveness=vina_exhaustiveness,
                    num_modes=vina_num_modes,
                    cpu=vina_cpu,
                    force=vina_force_redock,
                    pose_source_tag=vina_pose_source_tag,
                )
                out.at[idx, "pose_source"] = dock.pose_source
                out.at[idx, "docking_status"] = dock.status
                out.at[idx, "docking_error"] = dock.error
                out.at[idx, "vina_score_kcal_mol"] = dock.vina_score_kcal_mol if dock.vina_score_kcal_mol is not None else pd.NA
                out.at[idx, "ligand_sdf_used"] = str(dock.ligand_sdf) if dock.ligand_sdf else ""
                sdf = dock.ligand_sdf if dock.success else None
            else:
                out.at[idx, "pose_source"] = "none"
                out.at[idx, "docking_status"] = "not_enabled"

            if sdf is None:
                out.at[idx, "model_used"] = "sequence_only"
                out.at[idx, "fallback_used"] = True
                if can_dock:
                    reason = "vina_docking_failed_or_no_pose"
                    if out.at[idx, "docking_error"]:
                        reason += f": {out.at[idx, 'docking_error']}"
                else:
                    reason = "no_exact_ligand_sdf_pose_and_vina_not_enabled"
                out.at[idx, "fallback_reason"] = reason
                out.at[idx, "graph_status"] = "missing_pose"
                sequence_indices.append(idx)
            else:
                try:
                    graph = _get_or_build_graph(
                        cache=cache,
                        protein_pdb=protein_pdb,
                        ligand_sdf=Path(sdf),
                        complex_id=str(row.get("ligand_id", idx)),
                        pocket_cutoff=pocket_cutoff,
                        contact_cutoff=contact_cutoff,
                    )
                    out.at[idx, "graph_status"] = "ok"
                    graph_items.append({
                        "row_index": idx,
                        "graph": graph,
                        "prot_emb": prot_emb,
                        "lig_emb": lig_embs[str(row[REQUIRED_SMILES_COL])],
                    })
                except Exception as exc:
                    out.at[idx, "model_used"] = "sequence_only"
                    out.at[idx, "fallback_used"] = True
                    out.at[idx, "fallback_reason"] = f"graph_build_failed: {exc}"
                    out.at[idx, "graph_status"] = "failed"
                    sequence_indices.append(idx)
    else:
        sequence_indices = list(out.index)

    if sequence_indices:
        progress(0.55, "Running sequence_only predictions...")
        model = load_trained_model("sequence_only", cfg, device)
        preds = _predict_sequence_rows(model, device, prot_emb, lig_embs, out.loc[sequence_indices].copy(), batch_size)
        for idx, pred in preds.items():
            out.at[idx, "predicted_deltaG"] = pred
            out.at[idx, "sequence_status"] = "ok"

    if graph_items:
        progress(0.75, f"Running {requested_model} predictions...")
        model = load_trained_model(requested_model, cfg, device)
        preds = _predict_graph_rows(model, device, graph_items, batch_size)
        for idx, pred in preds.items():
            out.at[idx, "predicted_deltaG"] = pred
            out.at[idx, "sequence_status"] = "ok"

    out["predicted_deltaG"] = pd.to_numeric(out["predicted_deltaG"], errors="coerce")
    return out.sort_values("predicted_deltaG", ascending=True, na_position="last").reset_index(drop=True)

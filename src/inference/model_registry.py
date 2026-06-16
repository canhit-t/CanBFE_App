from __future__ import annotations

from pathlib import Path

import torch
import yaml

from src.models.fusion import build_model

AVAILABLE_MODELS = ["sequence_only", "graph_only", "concat_fusion", "cross_attention_fusion"]
GRAPH_MODELS = {"graph_only", "concat_fusion", "cross_attention_fusion"}


def resolve_device(device_choice: str) -> torch.device:
    if device_choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_choice)


def checkpoint_path_for(model_name: str, cfg: dict) -> Path:
    p = Path(cfg.get("output_dir", "outputs/phase3")) / model_name / "best_model.pt"
    if p.exists():
        return p

    alt = Path(cfg.get("output_dir", "outputs/phase3")) / "random" / model_name / "best_model.pt"
    if alt.exists():
        return alt

    raise FileNotFoundError(f"Could not find checkpoint for {model_name}. Tried {p} and {alt}")


def load_cfg(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_trained_model(model_name: str, cfg: dict, device: torch.device) -> torch.nn.Module:
    model_cfg = cfg["models"].get(model_name, {})
    model = build_model(model_name, model_cfg).to(device)
    state = torch.load(checkpoint_path_for(model_name, cfg), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

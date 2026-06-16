from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class CacheManager:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, category: str, key: str) -> Path:
        d = self.root / category
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.pt"

    def exists(self, category: str, key: str) -> bool:
        return self.path(category, key).exists()

    def load(self, category: str, key: str) -> Any:
        return torch.load(self.path(category, key), map_location="cpu", weights_only=False)

    def save(self, category: str, key: str, obj: Any) -> Path:
        p = self.path(category, key)
        torch.save(obj, p)
        return p

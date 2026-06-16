from __future__ import annotations

from collections import OrderedDict

import torch

from src.features.encode_ligand_smiles import ChemBERTaEncoder
from src.features.encode_protein_esm import ESM2Encoder
from src.inference.cache import CacheManager, hash_text


def protein_embedding(
    sequence: str,
    cache: CacheManager,
    device: torch.device,
    model_name: str = "facebook/esm2_t12_35M_UR50D",
) -> torch.Tensor:
    key = hash_text(f"{model_name}|{sequence}")
    if cache.exists("proteins", key):
        return cache.load("proteins", key)

    encoder = ESM2Encoder(model_name=model_name, device=str(device))
    result = encoder.encode_batch([sequence], ["target"], window_starts=[0])[0]
    if result.error:
        raise RuntimeError(f"ESM2 encoding failed: {result.error}")
    tensor = result.global_emb.detach().cpu()
    cache.save("proteins", key, tensor)
    return tensor


def ligand_embeddings(
    smiles_list: list[str],
    cache: CacheManager,
    device: torch.device,
    batch_size: int = 64,
    model_name: str = "DeepChem/ChemBERTa-77M-MTR",
) -> dict[str, torch.Tensor]:
    unique = list(OrderedDict((s, None) for s in smiles_list).keys())
    output = {}
    missing = []

    for smiles in unique:
        key = hash_text(f"{model_name}|{smiles}")
        if cache.exists("ligands", key):
            output[smiles] = cache.load("ligands", key)
        else:
            missing.append(smiles)

    if missing:
        encoder = ChemBERTaEncoder(model_name=model_name, device=str(device))
        for i in range(0, len(missing), batch_size):
            batch = missing[i:i + batch_size]
            encoded = encoder.encode_batch(batch, [f"lig_{j}" for j in range(len(batch))])
            for smiles, emb in zip(batch, encoded):
                if emb.error:
                    raise RuntimeError(f"ChemBERTa encoding failed for {smiles}: {emb.error}")
                tensor = emb.global_emb.detach().cpu()
                cache.save("ligands", hash_text(f"{model_name}|{smiles}"), tensor)
                output[smiles] = tensor

    return output

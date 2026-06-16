"""ChemBERTa-based ligand SMILES encoder.

Generates one mean-pooled embedding per SMILES string using a pre-trained
ChemBERTa model from HuggingFace (default: ``DeepChem/ChemBERTa-77M-MTR``).

Mean pooling is computed over all non-padding token positions (attention_mask
== 1), which includes CLS/EOS. This matches the common practice for
ChemBERTa-style models and is consistent with the sentence-BERT approach.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

LOGGER = logging.getLogger(__name__)

CHEMBERTA_MAX_TOKENS: int = 512


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class LigandEmbedding:
    pdb_id: str
    global_emb: Tensor     # shape [hidden_dim]
    was_truncated: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class ChemBERTaEncoder:
    """Mean-pooled SMILES encoder using ChemBERTa.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier,
        e.g. ``"DeepChem/ChemBERTa-77M-MTR"``.
    device:
        ``"cpu"``, ``"cuda"``, or ``None`` (auto-detect).
    max_tokens:
        Maximum token count before truncation (default 512).
    """

    def __init__(
        self,
        model_name: str = "DeepChem/ChemBERTa-77M-MTR",
        device: Optional[str] = None,
        max_tokens: int = CHEMBERTA_MAX_TOKENS,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer  # lazy import

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        LOGGER.info("Loading ChemBERTa tokenizer: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        LOGGER.info("Loading ChemBERTa model: %s → %s", model_name, self.device)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

        self.hidden_dim: int = self.model.config.hidden_size
        LOGGER.info("ChemBERTa hidden_dim=%d  max_tokens=%d", self.hidden_dim, max_tokens)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_batch(
        self,
        smiles_list: list[str],
        pdb_ids: list[str],
    ) -> list[LigandEmbedding]:
        """Encode a batch of SMILES strings.

        Parameters
        ----------
        smiles_list:
            List of canonical SMILES strings.
        pdb_ids:
            Corresponding PDB identifiers (for logging).

        Returns
        -------
        List of :class:`LigandEmbedding` objects, one per input SMILES.
        Failed entries have ``error`` set and ``global_emb`` zeroed.
        """
        assert len(smiles_list) == len(pdb_ids)

        # Detect truncation before the batched tokenization (fast, CPU-only pass)
        was_trunc = self._detect_truncation(smiles_list)

        for pid, trunc in zip(pdb_ids, was_trunc):
            if trunc:
                LOGGER.warning("%s: SMILES exceeds %d tokens; truncating.", pid, self.max_tokens)

        # --- Try full-batch encode first; fall back to per-sample on failure ---
        try:
            return self._encode_smiles(smiles_list, pdb_ids, was_trunc)
        except Exception as batch_exc:  # noqa: BLE001
            LOGGER.warning(
                "Full-batch SMILES encode failed (%s); retrying sample-by-sample.", batch_exc
            )
            results: list[LigandEmbedding] = []
            for smi, pid, trunc in zip(smiles_list, pdb_ids, was_trunc):
                try:
                    results.extend(self._encode_smiles([smi], [pid], [trunc]))
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error("%s: SMILES encoding failed: %s", pid, exc)
                    results.append(
                        LigandEmbedding(
                            pdb_id=pid,
                            global_emb=torch.zeros(self.hidden_dim),
                            was_truncated=trunc,
                            error=str(exc),
                        )
                    )
            return results

    # ------------------------------------------------------------------

    def _detect_truncation(self, smiles_list: list[str]) -> list[bool]:
        """Return per-SMILES boolean indicating whether tokenization exceeds max_tokens."""
        result = []
        for smi in smiles_list:
            try:
                n_tokens = len(
                    self.tokenizer.encode(smi, add_special_tokens=True)
                )
                result.append(n_tokens > self.max_tokens)
            except Exception:  # noqa: BLE001
                result.append(False)
        return result

    def _encode_smiles(
        self,
        smiles_list: list[str],
        pdb_ids: list[str],
        was_trunc: list[bool],
    ) -> list[LigandEmbedding]:
        """Run the actual model forward pass for a list of SMILES strings."""
        enc = self.tokenizer(
            smiles_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        outputs = self.model(**enc)
        hidden = outputs.last_hidden_state  # [B, T, D]

        # Masked mean pool — respect padding (attention_mask == 1)
        attn_mask = enc["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
        summed = (hidden * attn_mask).sum(dim=1)                 # [B, D]
        counts = attn_mask.sum(dim=1).clamp(min=1e-9)            # [B, 1]
        pooled = (summed / counts).cpu()                         # [B, D]

        return [
            LigandEmbedding(
                pdb_id=pid,
                global_emb=pooled[i],
                was_truncated=trunc,
            )
            for i, (pid, trunc) in enumerate(zip(pdb_ids, was_trunc))
        ]

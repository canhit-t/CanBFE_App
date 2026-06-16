"""ESM2-based protein sequence encoder using HuggingFace transformers.

Design notes
------------
* Sequences longer than ``max_residues`` (default 1022 = 1024 − CLS − EOS)
  are encoded using a pocket-aware sliding window.  The caller computes
  ``window_start`` for each sequence and passes it to ``encode_batch``;
  this module just encodes the slice ``seq[window_start : window_start + max_residues]``.

* ESM2 token layout: ``[CLS] aa_1 aa_2 … aa_L [EOS]``
  → ``last_hidden_state[:, 1 : L+1, :]`` contains the per-residue vectors.
  ``residue_emb[i]`` therefore directly corresponds to the windowed
  subsequence position ``i`` (0-based).

* Global protein embedding: mean pool over the encoded window residues only
  (no CLS/EOS), so it always reflects the pocket neighbourhood for long proteins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

LOGGER = logging.getLogger(__name__)

ESM2_MAX_RESIDUES: int = 1022  # 1024 model max − CLS − EOS


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ProteinEmbedding:
    pdb_id: str
    global_emb: Tensor        # shape [hidden_dim]
    residue_emb: Tensor       # shape [seq_len_encoded, hidden_dim]
    seq_len_original: int
    seq_len_encoded: int      # ≤ ESM2_MAX_RESIDUES
    was_truncated: bool
    window_start: int = 0     # start of encoded window in original sequence
    window_end: int = -1      # end of encoded window (exclusive); -1 = not set
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class ESM2Encoder:
    """Encode protein sequences with a pre-trained ESM2 model.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier, e.g. ``"facebook/esm2_t12_35M_UR50D"``.
    device:
        ``"cpu"``, ``"cuda"``, or ``None`` (auto-detect).
    max_residues:
        Maximum residue count before truncation (default 1022).
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t12_35M_UR50D",
        device: Optional[str] = None,
        max_residues: int = ESM2_MAX_RESIDUES,
    ) -> None:
        from transformers import AutoTokenizer, EsmModel  # lazy import

        self.model_name = model_name
        self.max_residues = max_residues
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        LOGGER.info("Loading ESM2 tokenizer: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        LOGGER.info("Loading ESM2 model: %s → %s", model_name, self.device)
        self.model = EsmModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

        self.hidden_dim: int = self.model.config.hidden_size
        LOGGER.info("ESM2 hidden_dim=%d  max_residues=%d", self.hidden_dim, max_residues)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_batch(
        self,
        sequences: list[str],
        pdb_ids: list[str],
        window_starts: list[int] | None = None,
    ) -> list[ProteinEmbedding]:
        """Encode a batch of amino-acid sequences.

        Parameters
        ----------
        sequences:
            List of single-letter amino-acid strings (full, unsliced).
        pdb_ids:
            Corresponding PDB identifiers (used for logging).
        window_starts:
            Optional per-sequence window start positions (0-based) for
            long-protein windowing.  If ``None``, defaults to 0 for all.
            The caller is responsible for computing the pocket-aware window;
            see ``build_sequence_features.compute_window_start``.

        Returns
        -------
        List of :class:`ProteinEmbedding` objects, one per input sequence.
        Sequences that fail individually are returned with ``error`` set.
        """
        assert len(sequences) == len(pdb_ids), "sequences and pdb_ids must be the same length"
        if window_starts is None:
            window_starts = [0] * len(sequences)
        assert len(window_starts) == len(sequences)

        # --- Per-sequence windowing / truncation ---
        sliced: list[str] = []
        was_trunc: list[bool] = []
        orig_lens: list[int] = []
        actual_starts: list[int] = []

        for seq, pid, ws in zip(sequences, pdb_ids, window_starts):
            orig_lens.append(len(seq))
            if len(seq) > self.max_residues:
                # Clamp window_start to a valid range
                ws = max(0, min(ws, len(seq) - self.max_residues))
                sliced.append(seq[ws : ws + self.max_residues])
                was_trunc.append(True)
                actual_starts.append(ws)
                LOGGER.warning(
                    "%s: sequence length %d > max %d; "
                    "encoding window [%d, %d).",
                    pid, len(seq), self.max_residues, ws, ws + self.max_residues,
                )
            else:
                sliced.append(seq)
                was_trunc.append(False)
                actual_starts.append(0)

        # --- Try full-batch encode first; fall back to per-sample on failure ---
        try:
            return self._encode_sequences(
                sliced, pdb_ids, orig_lens, was_trunc, actual_starts
            )
        except Exception as batch_exc:  # noqa: BLE001
            LOGGER.warning(
                "Full-batch encode failed (%s); retrying sample-by-sample.", batch_exc
            )
            results: list[ProteinEmbedding] = []
            for seq, pid, orig_len, trunc, ws in zip(
                sliced, pdb_ids, orig_lens, was_trunc, actual_starts
            ):
                try:
                    results.extend(
                        self._encode_sequences([seq], [pid], [orig_len], [trunc], [ws])
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error("%s: encoding failed: %s", pid, exc)
                    dummy = torch.zeros(self.hidden_dim)
                    results.append(
                        ProteinEmbedding(
                            pdb_id=pid,
                            global_emb=dummy,
                            residue_emb=dummy.unsqueeze(0),
                            seq_len_original=orig_len,
                            seq_len_encoded=0,
                            was_truncated=trunc,
                            window_start=ws,
                            window_end=ws + self.max_residues,
                            error=str(exc),
                        )
                    )
            return results

    # ------------------------------------------------------------------

    def _encode_sequences(
        self,
        sequences: list[str],
        pdb_ids: list[str],
        orig_lens: list[int],
        was_trunc: list[bool],
        window_starts: list[int],
    ) -> list[ProteinEmbedding]:
        """Run the actual forward pass for a list of (already-windowed) sequences."""
        enc = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=False,   # manual windowing already done
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        outputs = self.model(**enc)
        hidden = outputs.last_hidden_state  # [B, T, hidden_dim]

        results: list[ProteinEmbedding] = []
        for i, (pid, seq, orig_len, trunc, ws) in enumerate(
            zip(pdb_ids, sequences, orig_lens, was_trunc, window_starts)
        ):
            seq_len = len(seq)
            # Positions 1..seq_len are residue tokens (0=CLS, seq_len+1=EOS)
            res_emb  = hidden[i, 1 : seq_len + 1, :].cpu()   # [seq_len, D]
            glob_emb = res_emb.mean(dim=0)                    # [D]
            results.append(
                ProteinEmbedding(
                    pdb_id=pid,
                    global_emb=glob_emb,
                    residue_emb=res_emb,
                    seq_len_original=orig_len,
                    seq_len_encoded=seq_len,
                    was_truncated=trunc,
                    window_start=ws,
                    window_end=ws + seq_len,
                )
            )
        return results

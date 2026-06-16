"""Map pocket residues to 0-based positions in the full-protein sequence.

How the mapping works
---------------------
Both the *full-protein PDB* and the *pocket PDB* are parsed with Biopython.
For each file we build an **ordered list** of unique residue keys::

    key = (chain_id: str, seq_num: int, icode: str)   # icode stripped

The full-protein list is traversed in the same order that Biopython's
``structure.get_residues()`` yields them — which is exactly the order used
by ``parse_protein.py`` when building ``protein_sequence``.  Therefore
position *i* in the key list corresponds to ``sequence[i]`` and to
``residue_embeddings[i]`` from ESM2.

To map a pocket residue with key *k*:

1. Look up *k* in a dict ``{full_key → 0-based index}``.
2. If found, record the index.
3. If not found (HETATM-only residue, or missing atoms), log and count as
   unmapped.

The returned sorted, deduplicated index list can directly be used to slice
a residue-embedding tensor::

    pocket_emb = residue_emb[pocket_indices, :]   # [n_pocket, dim]

Edge cases
----------
* No full-protein file → fall back to pocket-only mapping (all pocket
  residues get indices 0, 1, … n-1; ``mapping_status = "no_full_protein"``).
* No pocket file → ``pocket_indices = []``,
  ``mapping_status = "no_pocket"``.
* Window-based truncation (dropping residues outside the ESM2 window) is
  handled in ``build_sequence_features.run_pocket_phase``, not here.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)

# ── Type alias ────────────────────────────────────────────────────────────────
# (chain_id, residue_sequence_number, insertion_code)
_ResKey = tuple[str, int, str]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PocketMapping:
    pdb_id: str
    pocket_indices: list[int]       # 0-based indices into full-protein sequence
    n_full_residues: int
    n_pocket_residues: int
    n_mapped: int                   # pocket residues matched in full protein
    n_unmapped: int                 # pocket residues NOT matched
    n_truncated_dropped: int        # indices dropped outside the ESM2 window (set by build_sequence_features)
    mapping_status: str             # 'ok' | 'partial' | 'no_pocket' |
                                    # 'no_full_protein' | 'failed'
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_residue_keys(pdb_path: Path) -> list[_ResKey]:
    """Return ordered, deduplicated residue keys from a PDB file.

    Only ATOM records (``hetfield == ' '``) are considered.  If none are
    found (e.g. a minimal pocket file where the parser labels everything as
    HETATM), all residues are returned as fallback.

    Parameters
    ----------
    pdb_path:
        Path to a ``.pdb`` file.

    Returns
    -------
    List of ``(chain_id, seq_num, icode)`` tuples in PDB order,
    deduplicated by key.
    """
    try:
        from Bio.PDB import PDBParser  # type: ignore[import-untyped]
        from Bio.PDB.PDBExceptions import PDBConstructionWarning  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("Biopython is required for pocket mapping.") from exc

    parser = PDBParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        structure = parser.get_structure("mol", str(pdb_path))

    seen: set[_ResKey] = set()
    keys: list[_ResKey] = []
    hetatm_keys: list[_ResKey] = []   # fallback bucket

    for residue in structure.get_residues():
        hetfield, seqnum, icode = residue.get_id()
        chain_id = residue.get_parent().get_id()
        key: _ResKey = (chain_id, int(seqnum), icode.strip())

        if key in seen:
            LOGGER.debug(
                "%s: duplicate residue key %s – skipped",
                pdb_path.name, key,
            )
            continue
        seen.add(key)

        if hetfield.strip() == "":   # standard ATOM record
            keys.append(key)
        else:
            hetatm_keys.append(key)

    # Fall back to all residues if no ATOM records found
    return keys if keys else hetatm_keys


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_pocket_residues(
    pdb_id: str,
    full_pdb: Optional[Path],
    pocket_pdb: Optional[Path],
    max_residues: int = 1022,
) -> PocketMapping:
    """Map pocket residues to 0-based positions in the full-protein sequence.

    Parameters
    ----------
    pdb_id:
        4-character PDB identifier (for logging).
    full_pdb:
        Path to the full-protein PDB file, or ``None`` if unavailable.
    pocket_pdb:
        Path to the pocket PDB file, or ``None`` if unavailable.
    max_residues:
        Retained for API compatibility; no longer used for truncation.
        Window-based filtering happens in ``build_sequence_features``.

    Returns
    -------
    :class:`PocketMapping` with sorted, deduplicated pocket indices.
    """
    # --- No pocket file ---
    if pocket_pdb is None or not pocket_pdb.exists():
        return PocketMapping(
            pdb_id=pdb_id,
            pocket_indices=[],
            n_full_residues=0,
            n_pocket_residues=0,
            n_mapped=0,
            n_unmapped=0,
            n_truncated_dropped=0,
            mapping_status="no_pocket",
        )

    # --- Parse pocket ---
    try:
        pocket_keys = _get_residue_keys(pocket_pdb)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("%s: failed to parse pocket PDB %s: %s", pdb_id, pocket_pdb, exc)
        return PocketMapping(
            pdb_id=pdb_id,
            pocket_indices=[],
            n_full_residues=0,
            n_pocket_residues=0,
            n_mapped=0,
            n_unmapped=0,
            n_truncated_dropped=0,
            mapping_status="failed",
            error=str(exc),
        )

    # --- No full protein: treat pocket as the full protein ---
    if full_pdb is None or not full_pdb.exists():
        LOGGER.warning(
            "%s: full protein PDB not found; treating pocket as full protein.", pdb_id
        )
        n = len(pocket_keys)
        indices = list(range(n))
        return PocketMapping(
            pdb_id=pdb_id,
            pocket_indices=indices,
            n_full_residues=n,
            n_pocket_residues=n,
            n_mapped=n,
            n_unmapped=0,
            n_truncated_dropped=0,
            mapping_status="no_full_protein",
        )

    # --- Parse full protein ---
    try:
        full_keys = _get_residue_keys(full_pdb)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("%s: failed to parse full-protein PDB %s: %s", pdb_id, full_pdb, exc)
        return PocketMapping(
            pdb_id=pdb_id,
            pocket_indices=[],
            n_full_residues=0,
            n_pocket_residues=len(pocket_keys),
            n_mapped=0,
            n_unmapped=len(pocket_keys),
            n_truncated_dropped=0,
            mapping_status="failed",
            error=str(exc),
        )

    # --- Build lookup and map ---
    full_lookup: dict[_ResKey, int] = {key: idx for idx, key in enumerate(full_keys)}

    raw_indices: list[int] = []
    n_unmapped = 0

    for key in pocket_keys:
        idx = full_lookup.get(key)
        if idx is None:
            chain, seqnum, icode = key
            LOGGER.debug(
                "%s: pocket residue %s:%d%s not found in full protein.",
                pdb_id, chain, seqnum, icode or "",
            )
            n_unmapped += 1
        else:
            raw_indices.append(idx)

    # Deduplicate and sort — return all mapped indices; window-based
    # truncation is handled later in build_sequence_features.run_pocket_phase.
    dedup = sorted(set(raw_indices))

    # Determine status
    if not dedup and len(pocket_keys) > 0:
        status = "failed"
    elif n_unmapped > 0:
        status = "partial"
    else:
        status = "ok"

    return PocketMapping(
        pdb_id=pdb_id,
        pocket_indices=dedup,
        n_full_residues=len(full_keys),
        n_pocket_residues=len(pocket_keys),
        n_mapped=len(dedup),
        n_unmapped=n_unmapped,
        n_truncated_dropped=0,
        mapping_status=status,
    )

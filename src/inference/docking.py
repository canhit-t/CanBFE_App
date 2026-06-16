from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


from src.inference.cache import hash_file, hash_text


@dataclass
class DockingBox:
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float

    def as_vina_args(self) -> list[str]:
        return [
            "--center_x", str(self.center_x),
            "--center_y", str(self.center_y),
            "--center_z", str(self.center_z),
            "--size_x", str(self.size_x),
            "--size_y", str(self.size_y),
            "--size_z", str(self.size_z),
        ]


def compute_docking_box_from_coordinates(coords: list[tuple[float, float, float]], padding: float = 5.0, min_size: float = 12.0) -> DockingBox:
    """Compute a Vina docking box from existing 3D coordinates.

    This is intended for a known bound reference ligand or a pocket/active-site
    structure. It should not be used on a random/generated screening ligand,
    because that ligand has not been placed in the protein pocket yet.
    """
    if not coords:
        raise ValueError("No atom coordinates were found for auto-boxing.")

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    size_x = max(float(min_size), (max_x - min_x) + 2.0 * float(padding))
    size_y = max(float(min_size), (max_y - min_y) + 2.0 * float(padding))
    size_z = max(float(min_size), (max_z - min_z) + 2.0 * float(padding))

    return DockingBox(
        center_x=(min_x + max_x) / 2.0,
        center_y=(min_y + max_y) / 2.0,
        center_z=(min_z + max_z) / 2.0,
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
    )


def _coords_from_rdkit_mol(mol: Chem.Mol) -> list[tuple[float, float, float]]:
    if mol is None or mol.GetNumConformers() == 0:
        return []
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            # Hydrogens should not dominate the box size.
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append((float(pos.x), float(pos.y), float(pos.z)))
    return coords


def _coords_from_pdb_like_text(text: str) -> list[tuple[float, float, float]]:
    coords = []
    for line in text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            # Standard PDB/PDBQT coordinate columns.
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append((x, y, z))
        except Exception:
            parts = line.split()
            # Fallback for less standard PDBQT/PDB-like records.
            for i in range(len(parts) - 2):
                try:
                    x = float(parts[i])
                    y = float(parts[i + 1])
                    z = float(parts[i + 2])
                    coords.append((x, y, z))
                    break
                except Exception:
                    continue
    return coords


def read_coordinates_for_autobox(path: Path) -> list[tuple[float, float, float]]:
    """Read coordinates from reference ligand or pocket files for auto-boxing.

    Supported best: SDF/MOL/PDB/PDBQT. MOL2 is attempted through RDKit but may
    depend on the exact file formatting.
    """
    suffix = path.suffix.lower()
    coords: list[tuple[float, float, float]] = []

    if suffix == ".sdf":
        suppl = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        mol = next((m for m in suppl if m is not None), None)
        coords = _coords_from_rdkit_mol(mol) if mol is not None else []
    elif suffix == ".mol":
        mol = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=False)
        coords = _coords_from_rdkit_mol(mol) if mol is not None else []
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=False)
        coords = _coords_from_rdkit_mol(mol) if mol is not None else []
    elif suffix in {".pdb", ".pdbqt", ".ent"}:
        coords = _coords_from_pdb_like_text(path.read_text(errors="ignore"))
    else:
        # Try SDF first, then PDB-like text as a forgiving fallback.
        try:
            suppl = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
            mol = next((m for m in suppl if m is not None), None)
            coords = _coords_from_rdkit_mol(mol) if mol is not None else []
        except Exception:
            coords = []
        if not coords:
            coords = _coords_from_pdb_like_text(path.read_text(errors="ignore"))

    if not coords:
        raise ValueError(f"Could not read 3D coordinates from {path.name}. Use a bound reference ligand/pocket file with 3D coordinates.")
    return coords


def compute_docking_box_from_file(path: Path, padding: float = 5.0, min_size: float = 12.0) -> DockingBox:
    coords = read_coordinates_for_autobox(path)
    return compute_docking_box_from_coordinates(coords, padding=padding, min_size=min_size)


@dataclass
class DockingResult:
    success: bool
    ligand_sdf: Path | None = None
    vina_score_kcal_mol: float | None = None
    pose_source: str = "vina_manual_box"
    status: str = "not_run"
    error: str = ""
    work_dir: Path | None = None
    receptor_pdbqt: Path | None = None
    ligand_pdbqt: Path | None = None
    docked_pdbqt: Path | None = None
    stdout: str = ""
    stderr: str = ""


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=True,
    )


def _safe_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return value[:120] or "ligand"


def smiles_to_3d_sdf(smiles: str, out_sdf: Path, name: str = "ligand") -> Path:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    code = AllChem.EmbedMolecule(mol, params)
    if code != 0:
        # Fall back to random coordinates for difficult molecules.
        code = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=0xF00D)
    if code != 0:
        raise ValueError(f"RDKit could not generate a 3D conformer for {name}")

    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=300)
    except Exception:
        # A non-optimized conformer is still better than failing before docking.
        pass

    mol.SetProp("_Name", str(name))
    out_sdf.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(out_sdf))
    writer.write(mol)
    writer.close()
    return out_sdf



def _extract_ambiguous_histidine_templates(stderr: str) -> list[str]:
    """Return Meeko --set_template entries for genuine ambiguous histidines.

    RECEPTOR_PREP_LOOP_V4 marker.

    We only add residues from errors that explicitly say Meeko tied between
    histidine templates such as HIE/HID/HIP. We intentionally do NOT add residues
    from "No template matched" / heavy_miss messages, because those are usually
    incomplete residues and should be handled by -a / --allow_bad_res instead.
    """
    templates: list[str] = []
    text = stderr or ""

    pattern = re.compile(
        r"for residue_key='([^']+)'.{0,300}?tied.{0,150}?HIE\s+HID",
        flags=re.S,
    )

    for match in pattern.finditer(text):
        residue_key = match.group(1).strip()
        context = match.group(0)
        if "No template matched" in context or "heavy_miss" in context:
            continue
        template = f"{residue_key}=HIE"
        if residue_key and template not in templates:
            templates.append(template)

    return templates


def prepare_receptor_pdbqt(
    protein_pdb: Path,
    out_dir: Path,
    box: DockingBox | None = None,
    cache_key_extra: str = "",
) -> Path:
    """Prepare receptor PDBQT with Meeko's CLI module.

    The prepared receptor is cached because it is target-level, not ligand-level.
    This version loops over Meeko's ambiguous-histidine failures. Meeko often
    reports only one ambiguous histidine at a time, so a single retry can get
    past L:38 and then fail at L:49. We keep accumulating genuine HIE/HID/HIP
    ambiguity residues and retry until receptor prep succeeds or no new template
    can be learned.
    """
    key_parts = [hash_file(protein_pdb), "meeko_receptor_loop_v4", cache_key_extra]
    if box is not None:
        key_parts.append(json.dumps(asdict(box), sort_keys=True))
    key = hash_text("|".join(key_parts))[:24]
    target_dir = out_dir / "receptors" / key
    receptor_pdbqt = target_dir / "receptor.pdbqt"
    if receptor_pdbqt.exists() and receptor_pdbqt.stat().st_size > 0:
        return receptor_pdbqt

    target_dir.mkdir(parents=True, exist_ok=True)
    base_cmd = [
        sys.executable,
        "-m", "meeko.cli.mk_prepare_receptor",
        "--read_pdb", str(protein_pdb),
        "-o", str(target_dir / "receptor"),
        "-p", str(receptor_pdbqt),
        "-a",  # delete bad/missing-atom residues instead of hard failing when possible
    ]
    if box is not None:
        base_cmd += [
            "--box_center", str(box.center_x), str(box.center_y), str(box.center_z),
            "--box_size", str(box.size_x), str(box.size_y), str(box.size_z),
        ]

    receptor_templates_used: list[str] = []
    last_stdout = ""
    last_stderr = ""

    max_retries = 25
    for attempt in range(max_retries):
        cmd = list(base_cmd)
        if receptor_templates_used:
            cmd += ["-n", ",".join(receptor_templates_used)]

        try:
            _run(cmd)
            break
        except subprocess.CalledProcessError as exc:
            last_stdout = exc.stdout or ""
            last_stderr = exc.stderr or ""

            new_templates = _extract_ambiguous_histidine_templates(last_stderr)
            added = False
            for tmpl in new_templates:
                if tmpl not in receptor_templates_used:
                    receptor_templates_used.append(tmpl)
                    added = True

            if added:
                continue

            raise RuntimeError(
                "Meeko receptor preparation failed. "
                f"Automatic histidine templates used so far: {','.join(receptor_templates_used) or 'none'}. "
                f"STDOUT: {last_stdout}\nSTDERR: {last_stderr}"
            ) from exc
    else:
        raise RuntimeError(
            "Meeko receptor preparation failed after repeated ambiguous-histidine retries. "
            f"Templates attempted: {','.join(receptor_templates_used) or 'none'}. "
            f"Last STDOUT: {last_stdout}\nLast STDERR: {last_stderr}"
        )

    if receptor_templates_used:
        try:
            (target_dir / "receptor_prep_notes.json").write_text(
                json.dumps(
                    {
                        "automatic_set_template": receptor_templates_used,
                        "strategy": "ambiguous histidines defaulted to HIE; bad residues handled by -a",
                        "marker": "RECEPTOR_PREP_LOOP_V4",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    if receptor_pdbqt.exists():
        return receptor_pdbqt

    candidates = sorted(target_dir.glob("*.pdbqt"))
    if candidates:
        shutil.copyfile(candidates[0], receptor_pdbqt)
        return receptor_pdbqt

    raise RuntimeError("Meeko receptor preparation finished but no receptor PDBQT was produced.")

def prepare_ligand_pdbqt(ligand_sdf: Path, ligand_pdbqt: Path) -> Path:
    ligand_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m", "meeko.cli.mk_prepare_ligand",
        "-i", str(ligand_sdf),
        "-o", str(ligand_pdbqt),
    ]
    try:
        _run(cmd)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Meeko ligand preparation failed. "
            f"STDOUT: {exc.stdout}\nSTDERR: {exc.stderr}"
        ) from exc
    if not ligand_pdbqt.exists():
        raise RuntimeError("Meeko ligand preparation finished but no ligand PDBQT was produced.")
    return ligand_pdbqt


def export_vina_pdbqt_to_sdf(docked_pdbqt: Path, out_sdf: Path) -> Path:
    # MEEKO_EXPORT_S_FLAG_V1: this Meeko version uses -s for SDF output, not -o.
    out_sdf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m", "meeko.cli.mk_export",
        str(docked_pdbqt),
        "-s", str(out_sdf),
    ]
    try:
        _run(cmd)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Meeko export of docked PDBQT to SDF failed. "
            f"STDOUT: {exc.stdout}\nSTDERR: {exc.stderr}"
        ) from exc
    if not out_sdf.exists():
        raise RuntimeError("Meeko export finished but no docked SDF was produced.")
    return out_sdf


def parse_vina_score(stdout: str, docked_pdbqt: Path | None = None) -> float | None:
    texts = [stdout or ""]
    if docked_pdbqt is not None and docked_pdbqt.exists():
        try:
            texts.append(docked_pdbqt.read_text(errors="ignore"))
        except Exception:
            pass

    for text in texts:
        m = re.search(r"REMARK\s+VINA\s+RESULT:\s*([-+]?\d+(?:\.\d+)?)", text)
        if m:
            return float(m.group(1))

    for line in (stdout or "").splitlines():
        m = re.match(r"\s*1\s+([-+]?\d+(?:\.\d+)?)\s+", line)
        if m:
            return float(m.group(1))
    return None


def dock_smiles_with_vina(
    smiles: str,
    ligand_id: str,
    protein_pdb: Path,
    vina_exe: Path,
    box: DockingBox,
    docking_root: Path,
    exhaustiveness: int = 8,
    num_modes: int = 1,
    cpu: int = 0,
    force: bool = False,
    pose_source_tag: str = "vina_manual_box",
) -> DockingResult:
    """Generate a ligand conformer, dock it with Vina, and export top pose as SDF.

    The docking box is protein/target-level. It is intentionally reused across ligands.
    """
    if not vina_exe.exists():
        return DockingResult(success=False, status="vina_exe_missing", error=f"Vina executable not found: {vina_exe}")

    ligand_id_safe = _safe_id(ligand_id)
    key = hash_text(
        "|".join([
            hash_file(protein_pdb),
            str(smiles),
            json.dumps(asdict(box), sort_keys=True),
            str(vina_exe),
            str(exhaustiveness),
            str(num_modes),
            pose_source_tag,
            "vina_box_v2",
        ])
    )[:24]
    work_dir = docking_root / "ligands" / f"{ligand_id_safe}_{key}"
    work_dir.mkdir(parents=True, exist_ok=True)

    input_sdf = work_dir / "input_3d.sdf"
    ligand_pdbqt = work_dir / "ligand.pdbqt"
    docked_pdbqt = work_dir / "docked.pdbqt"
    docked_sdf = work_dir / "docked.sdf"
    metadata_json = work_dir / "docking_metadata.json"

    if docked_sdf.exists() and metadata_json.exists() and not force:
        try:
            meta = json.loads(metadata_json.read_text())
            return DockingResult(
                success=True,
                ligand_sdf=docked_sdf,
                vina_score_kcal_mol=meta.get("vina_score_kcal_mol"),
                pose_source=meta.get("pose_source", pose_source_tag),
                status="cached",
                work_dir=work_dir,
                receptor_pdbqt=Path(meta["receptor_pdbqt"]) if meta.get("receptor_pdbqt") else None,
                ligand_pdbqt=ligand_pdbqt,
                docked_pdbqt=docked_pdbqt,
            )
        except Exception:
            pass

    try:
        receptor_pdbqt = prepare_receptor_pdbqt(protein_pdb, docking_root, box=box)
        smiles_to_3d_sdf(smiles, input_sdf, name=ligand_id_safe)
        prepare_ligand_pdbqt(input_sdf, ligand_pdbqt)

        cmd = [
            str(vina_exe.resolve()),
            "--receptor", str(receptor_pdbqt.resolve()),
            "--ligand", str(ligand_pdbqt.resolve()),
            *box.as_vina_args(),
            "--exhaustiveness", str(int(exhaustiveness)),
            "--num_modes", str(int(num_modes)),
            "--cpu", str(int(cpu)),
            "--out", str(docked_pdbqt.resolve()),
        ]
        # Use absolute paths and do not change cwd. Vina on Windows often fails
        # with exit status 1 when relative paths are interpreted from the
        # ligand work directory rather than the project root.
        try:
            completed = _run(cmd)
        except subprocess.CalledProcessError as vina_exc:
            raise RuntimeError(
                "Vina docking command failed. "
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT: {vina_exc.stdout}\n"
                f"STDERR: {vina_exc.stderr}"
            ) from vina_exc
        score = parse_vina_score(completed.stdout, docked_pdbqt)
        export_vina_pdbqt_to_sdf(docked_pdbqt, docked_sdf)

        metadata = {
            "ligand_id": ligand_id,
            "smiles": smiles,
            "pose_source": pose_source_tag,
            "vina_score_kcal_mol": score,
            "box": asdict(box),
            "vina_exe": str(vina_exe),
            "receptor_pdbqt": str(receptor_pdbqt),
            "ligand_pdbqt": str(ligand_pdbqt),
            "docked_pdbqt": str(docked_pdbqt),
            "docked_sdf": str(docked_sdf),
            "exhaustiveness": int(exhaustiveness),
            "num_modes": int(num_modes),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return DockingResult(
            success=True,
            ligand_sdf=docked_sdf,
            vina_score_kcal_mol=score,
            pose_source=pose_source_tag,
            status="success",
            work_dir=work_dir,
            receptor_pdbqt=receptor_pdbqt,
            ligand_pdbqt=ligand_pdbqt,
            docked_pdbqt=docked_pdbqt,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except Exception as exc:
        return DockingResult(
            success=False,
            status="failed",
            error=str(exc),
            work_dir=work_dir,
            ligand_pdbqt=ligand_pdbqt if ligand_pdbqt.exists() else None,
            docked_pdbqt=docked_pdbqt if docked_pdbqt.exists() else None,
        )

# CanBFE

CanBFE is a protein–ligand binding free energy prediction project for estimating binding affinity / ΔG from protein, ligand, and structure-derived representations.

The repository contains the application code, model definitions, inference utilities, training/analysis scripts, configuration files, documentation, and app tooling. Large licensed datasets and generated cached features are intentionally **not included** in this GitHub repository.

---

## Repository contents

Expected repository structure:

```text
CanBFE/
├── app.py
├── configs/
├── docs/
├── scripts/
├── src/
├── tools/
├── README.md
└── .gitignore
```

### Main folders

```text
app.py
```

Streamlit application for ligand screening, AutoDock Vina docking, model inference, and benchmark/result viewing when required files are present.

```text
configs/
```

YAML configuration files for training, inference, and model setup.

```text
src/
```

Core source code, including data processing, feature generation, model definitions, training utilities, evaluation, and inference utilities.

```text
scripts/
```

Analysis scripts used during model evaluation and manuscript development. These include oracle analysis, router/meta-model diagnostics, error analysis, hard-test diagnostics, and plotting scripts.

```text
tools/
```

External tools required by the app when available. For the current Windows app workflow, this includes the AutoDock Vina executable.

```text
docs/
```

Important project documentation, including installation notes, methodology, pipeline documentation, architecture notes, and environment/package information.

---

## Important documentation files

The `docs/` folder contains project notes that should be read before training, reproducing experiments, or modifying model architecture.

Recommended files:

```text
docs/installation.txt
```

Environment setup commands and core package installation instructions.

```text
docs/pipfreeze.txt
```

Known working package versions from a development environment.

```text
docs/methodology.md
```

Scientific methodology, dataset description, feature-generation logic, model architectures, training protocol, oracle/router analysis, and evaluation design.

```text
docs/pipeline_docs.txt
```

Detailed pipeline documentation covering metadata extraction, ESM2/ChemBERTa feature generation, graph construction, train/validation splits, model training, evaluation, and analysis scripts.

```text
docs/cross_attention_fusion_architecture.md
```

Detailed explanation and diagram of the `CrossAttentionFusionModel` architecture.

The documentation notes that the model receives sequence-level ESM2/ChemBERTa embeddings and structure-level graph inputs, with cross-attention between global sequence tokens and local graph/node embeddings.

---

# What is not included in this repository

The following folders are intentionally excluded from Git:

data/
pdbbind_v2018/
hard_test/

These folders are excluded because they are large, generated, licensed, or managed separately through the CanHitt OneDrive storage repository. See the section below for instructions on how to access the required data.

# Required data access

Training, benchmark reproduction, and some app views require datasets, cached features, and generated analysis outputs that are not included in this GitHub repository.

These files are stored separately in the internal CanHitt OneDrive storage. To access them, ask the project supervisor, Dr. AG, for access to the CanHitt shared repository and look in:

AI_Handle/

The external data storage should contain the following zipped folders:

File	Purpose
data.zip	Cached featurized data used by the app, training, development, and visualization workflows
pdbbind_v2018.zip	PDBbind v2018 raw dataset used for model training
hard_test.zip	External hard-test dataset used for evaluation, benchmarking, and visualization

Without these files, users can inspect the code and run limited app functionality, but they will not be able to reproduce model training, hard-test benchmarking, or all analysis views.

The zipped folders provide access to some or all of the following project assets:

PDBbind v2018 raw dataset
external hard-test dataset
processed metadata tables
train/validation/external split files
cached ESM2 protein embeddings
cached ChemBERTa ligand embeddings
cached pocket ESM2 embeddings
cached structural graph features
trained model checkpoints, if not regenerated locally
benchmark prediction CSVs
oracle/router/error-analysis outputs

Do not commit these zipped folders, extracted datasets, cached features, or generated model outputs to GitHub. They are intentionally stored outside version control because of file size and data-sharing restrictions.

---

## Dataset requirements

The full training pipeline assumes PDBBind-style complex folders containing files such as:

```text
<pdb_id>_protein.pdb
<pdb_id>_pocket.pdb
<pdb_id>_ligand.sdf
<pdb_id>_ligand.mol2
```

The methodology uses PDBBind v2018 for training/validation and an external temporal hard-test set from later PDB entries. Affinities are converted to binding free energy using:

```text
ΔG = RT ln K
```

where `K` is Ki or Kd in molar units and final ΔG units are kcal mol⁻¹.

---

## Installation

Python 3.11 is recommended.

The installation notes in `docs/installation.txt` use a Python 3.11 virtual environment, upgrade `pip`, `setuptools`, and `wheel`, install a CUDA 12.8 PyTorch nightly build, and then install the scientific, ML, graph, docking, and Streamlit dependencies.

### 1. Create and activate environment

On Windows PowerShell:

```powershell
py -3.11 -m venv bindfusion
.\bindfusion\Scripts\activate
```

On Linux/macOS:

```bash
python3.11 -m venv bindfusion
source bindfusion/bin/activate
```

Upgrade packaging tools:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### 2. Install PyTorch

For RTX 50-series GPUs or other newer NVIDIA GPUs, CUDA 12.8 may be required.

A tested development command was:

```powershell
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

For older CUDA environments, use the PyTorch installation command appropriate for your CUDA version from the official PyTorch selector.

Verify PyTorch/CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If CUDA fails, the app can still be run on CPU, but ESM2/ChemBERTa embedding generation will be slower.

### 3. Install core project dependencies

On Windows PowerShell, use backticks for multi-line commands:

```powershell
pip install `
  numpy pandas scipy scikit-learn matplotlib tqdm pyyaml `
  biopython gemmi MDAnalysis `
  rdkit `
  transformers accelerate datasets tokenizers safetensors sentencepiece einops `
  torch_geometric `
  prolif meeko openbabel-wheel `
  spyrmsd posebusters `
  wandb hydra-core omegaconf rich loguru `
  streamlit
```

The development package list includes packages such as `rdkit`, `meeko`, `openbabel-wheel`, `MDAnalysis`, `torch-geometric`, `transformers`, `streamlit`, `wandb`, `hydra-core`, `omegaconf`, and other scientific dependencies.

### 4. Optional: install exact versions from pip freeze

For strict reproduction, compare against:

```text
docs/pipfreeze.txt
```

This file records a known development environment. It includes, among many others:

```text
torch
torchvision
torchaudio
torch-geometric
transformers
rdkit
meeko
openbabel-wheel
MDAnalysis
gemmi
biopython
prolif
posebusters
spyrmsd
streamlit
hydra-core
omegaconf
wandb
```

Exact version pinning may need adjustment depending on CUDA, GPU, Python version, and operating system.

---

## AutoDock Vina setup

The app can optionally generate docked ligand poses using AutoDock Vina.

On Windows, the recommended setup is to use the standalone Vina executable rather than installing the Python `vina` package.

Expected executable path:

```text
tools/vina/vina_1.2.7_win.exe
```

Test from the repository root:

```powershell
.\tools\vina\vina_1.2.7_win.exe --help
```

The app uses the following docking workflow:

```text
SMILES
→ RDKit 3D ligand conformer
→ Meeko ligand PDBQT preparation
→ Meeko receptor PDBQT preparation
→ AutoDock Vina docking
→ Meeko export of docked PDBQT to SDF
→ graph/fusion model inference
```

The docking box is target-level, not ligand-level. One protein target uses one binding-site box, and all screened ligands are docked into that same site.

For auto-boxing, use:

```text
known bound reference ligand SDF
or pocket PDB
```

Do not auto-box from the full protein PDB unless you intentionally want a very large whole-protein search box.

---

## Running the app

From the repository root:

```powershell
streamlit run app.py
```

The app opens locally at:

```text
http://localhost:8501
```

---

## App workflow

For prospective ligand screening:

```text
1. Upload a protein PDB.
2. Upload a ligand CSV containing a SMILES column.
3. Select a model.
4. If using graph/fusion models without exact ligand SDFs, enable Vina docking.
5. Upload a known bound reference ligand or pocket file for auto-boxing.
6. Click “Automatically determine box.”
7. Run screening.
8. Download the results table.
```

Minimum ligand CSV:

```csv
SMILES
CCO
CC(=O)O
```

Recommended ligand CSV:

```csv
ligand_id,SMILES,name,notes
ligand_001,CCO,ethanol,example only
ligand_002,CC(=O)O,acetic acid,example only
```

The app preserves input metadata columns.

---

## App output columns

Typical output columns include:

```text
ligand_id
SMILES
model_requested
model_used
fallback_used
fallback_reason
pose_source
ligand_sdf_used
docking_status
docking_error
vina_score_kcal_mol
box_center_x
box_center_y
box_center_z
box_size_x
box_size_y
box_size_z
predicted_deltaG
```

Important distinction:

```text
vina_score_kcal_mol
```

is the Vina docking score.

```text
predicted_deltaG
```

is the CanBFE model prediction.

These are not the same quantity and should not be interpreted interchangeably.

---

## Deployable model modes

The currently deployable app modes are:

```text
sequence_only
graph_only
concat_fusion
```

The `cross_attention_fusion` mode requires architecture-matched checkpoints. If the checkpoint was trained with an older model class, loading may fail with missing keys such as:

```text
local_layernorm.*
global_layernorm.*
local_pool_proj.*
global_pool_proj.*
```

The cross-attention architecture documentation explains that the model uses ESM2 protein embeddings, ChemBERTa ligand embeddings, graph node features, bidirectional cross-attention, pooling, and an MLP head.

If the checkpoint does not match the current implementation, either:

```text
retrain cross_attention_fusion with the current code
```

or:

```text
restore the exact model class used to produce the checkpoint
```

Do not use `strict=False` to bypass missing keys, because that would randomly initialize missing layers.

---

## Model/checkpoint compatibility

Checkpoints must match the architecture used during training.

For example, if an older graph checkpoint was trained without RBF distance modulation:

```yaml
rbf_dim: 0
```

then inference should also use:

```yaml
rbf_dim: 0
```

If the current config expects RBF layers but the checkpoint does not contain them, PyTorch may report missing keys such as:

```text
gnn.rbf_encoder.centers
gnn.rbf_encoder.sigma
gnn.conv_layers.0.lin_edge.weight
gnn.conv_layers.1.lin_edge.weight
```

The methodology notes that graph edges can carry distances and that RBF distance encoding may be used to inject distance information into message-passing layers.

Again: do not use `strict=False` unless intentionally testing an untrained/randomly initialized architectural component.

---

## Protein, pocket, and graph inputs

For sequence-only prediction:

```text
protein sequence → ESM2 embedding
SMILES → ChemBERTa embedding
```

For graph and fusion prediction:

```text
protein PDB + ligand pose SDF → local pocket/complex graph
```

For prospective screening with Vina:

```text
protein.pdb
→ full sequence extraction / ESM2 embedding
→ receptor preparation for Vina
→ source structure for local pocket graph extraction

docked ligand SDF
→ ligand pose used to build the local complex graph
```

A `pocket.pdb` is useful for benchmarking or auto-boxing. However, for prospective screening, a fixed pocket file should not be blindly reused as the graph for every docked ligand unless that is explicitly intended. The preferred screening design is:

```text
protein.pdb + docked ligand SDF
→ extract local pocket around docked ligand
→ build graph
```

The pipeline documentation describes graph construction as a heterogeneous graph combining ligand atoms and pocket residues, with ligand covalent edges, ligand–pocket contacts, and pocket–pocket contacts.

---

## Training and development pipeline

Full training/development requires external data and cached features.

The high-level pipeline is:

```text
1. Build metadata from PDBBind/hard-test structure folders.
2. Parse Ki/Kd affinities and convert to ΔG.
3. Extract protein sequences and ligand SMILES.
4. Encode protein sequences using ESM2.
5. Encode ligands using ChemBERTa.
6. Map pocket residues.
7. Build pocket-contextual ESM2 features when needed.
8. Build structural complex graphs.
9. Generate train/validation/external hard-test splits.
10. Train selected model architectures.
11. Evaluate on train/validation/external hard-test splits.
12. Run analysis scripts for oracle/router/error analysis.
```

The detailed pipeline documentation describes metadata extraction, sequence and pocket featurization, structural graph construction, split strategies, model architectures, training, evaluation, and downstream analysis scripts.

---

## Expected external data layout

A local data setup may look like:

```text
data/
├── pdbbind_v2018/
│   └── <pdb_id>/
│       ├── <pdb_id>_protein.pdb
│       ├── <pdb_id>_pocket.pdb
│       ├── <pdb_id>_ligand.sdf
│       └── <pdb_id>_ligand.mol2
├── hard_test/
├── processed/
├── splits/
└── features/
    ├── protein_global_embeddings.pt
    ├── protein_residue_embeddings/
    ├── pocket_residue_indices.pt
    ├── pocket_esm2_embeddings.pt
    ├── ligand_smiles_embeddings.pt
    └── graphs/
```

Actual paths should be configured in the YAML files under:

```text
configs/
```

---

## Analysis scripts

The `scripts/` folder contains scripts used for model evaluation and manuscript analyses, including:

```text
oracle analysis
router/meta-model diagnostics
error analysis
hard-test failure analysis
descriptor analysis
benchmark visualization
prediction plots
training curve plots
```

These scripts may require:

```text
prediction CSVs
metadata tables
feature index files
graph index files
descriptor tables
oracle/router outputs
trained checkpoint outputs
```

If these files are missing, obtain them from the internal Teams repository or regenerate them locally.

---

## Known limitations

```text
1. Large datasets and cached features are not included in Git.
2. Some app views require benchmark prediction CSVs or cached outputs.
3. Cross-attention checkpoints must match the exact architecture.
4. Vina docking requires a meaningful target binding box.
5. Vina-generated poses are predicted poses, not experimental bound poses.
6. Predictions from docked poses are not equivalent to benchmark predictions using experimental co-crystal ligand poses.
7. CPU-only ESM2/ChemBERTa embedding generation can be slow.
8. PDB/receptor preparation can fail for problematic structures unless cleaned or manually prepared.
9. Model results should always be reported with the exact checkpoint/config/features used.
```

---

## Git hygiene

Do not commit:

```text
virtual environments
PDBbind/raw datasets
hard-test datasets
cached features
screening cache
large checkpoints
temporary outputs
private/customer files
```

Recommended `.gitignore` entries:

```gitignore
# environments
bindfusion/
bindfusion311/
venv/
.env/

# Python/cache
__pycache__/
*.pyc
.cache/

# data and generated outputs
data/
outputs/
screening_cache/
*.pt
*.pth
*.ckpt
*.pkl
*.joblib

# OS/editor
.DS_Store
.vscode/
.idea/

# archives/logs
*.zip
*.log
```

---

## Reproducibility note

When reporting results, always distinguish between:

```text
experimental bound ligand poses
Vina-docked ligand poses
sequence-only predictions
graph/fusion predictions
oracle/router retrospective analyses
prospective deployable screening workflows
```

Benchmark claims should be tied to the exact dataset, checkpoint, YAML config, and feature-generation pipeline used for the reported experiment.

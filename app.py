from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference.benchmark import (
    find_oracle_scatter,
    load_one_model_hard_test,
    make_truth_vs_pred_figure,
)
from src.inference.cache import CacheManager
from src.inference.docking import DockingBox, compute_docking_box_from_file
from src.inference.inputs import load_ligand_table, save_uploaded_file, standardize_protein_from_pdb
from src.inference.model_registry import AVAILABLE_MODELS, GRAPH_MODELS
from src.inference.predict import run_screening


st.set_page_config(page_title="BindFusion screening", page_icon="🧬", layout="wide")

st.title("BindFusion screening")
st.caption("Predict protein–ligand binding free energy, ΔG, in kcal/mol.")

screen_tab, benchmark_tab = st.tabs(["Screen ligands", "Hard-test benchmark"])

for _key, _default in {
    "vina_box_center_x": 0.0,
    "vina_box_center_y": 0.0,
    "vina_box_center_z": 0.0,
    "vina_box_size_x": 20.0,
    "vina_box_size_y": 20.0,
    "vina_box_size_z": 20.0,
    "vina_box_source": "manual",
}.items():
    st.session_state.setdefault(_key, _default)

with st.sidebar:
    st.header("Model")
    requested_model = st.selectbox(
        "Choose screening model",
        AVAILABLE_MODELS,
        index=0,
        help=(
            "Graph/fusion models require exact protein–ligand structures. "
            "Without matching ligand SDF poses, the app falls back to sequence_only."
        ),
    )

    st.header("Settings")
    config_path = Path(st.text_input("Config path", "configs/phase3.yaml"))
    cache_dir = Path(st.text_input("Cache directory", "screening_cache"))
    output_dir = Path(st.text_input("Output directory", "outputs/screening_runs"))
    device_choice = st.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
    batch_size = int(st.number_input("Batch size", min_value=1, max_value=512, value=64, step=1))

    st.header("Vina docking (optional)")
    vina_enabled = st.checkbox(
        "Dock missing ligand poses with AutoDock Vina",
        value=False,
        help="Use one target-level docking box for all ligands that do not have uploaded exact SDF poses.",
    )
    vina_exe_path = Path(st.text_input("Vina executable path", "tools/vina/vina_1.2.7_win.exe"))
    st.caption("The docking box is for the protein target/pocket and is reused for every ligand in the CSV.")

    st.markdown("**Auto-box, optional**")
    autobox_upload = st.file_uploader(
        "Known bound reference ligand or pocket file",
        type=["sdf", "mol", "mol2", "pdb", "pdbqt", "ent"],
        accept_multiple_files=False,
        help=(
            "Use a known bound ligand or a pocket/active-site file already in the protein coordinate frame. "
            "Do not use the new screening ligand SMILES for this."
        ),
    )
    auto_col1, auto_col2 = st.columns(2)
    with auto_col1:
        autobox_padding = float(st.number_input("Auto-box padding Å", min_value=0.0, max_value=20.0, value=5.0, step=0.5, format="%.2f"))
    with auto_col2:
        autobox_min_size = float(st.number_input("Minimum box size Å", min_value=4.0, max_value=40.0, value=12.0, step=0.5, format="%.2f"))

    if st.button("Automatically determine box", disabled=autobox_upload is None):
        autobox_dir = cache_dir / "_autobox_inputs"
        autobox_dir.mkdir(parents=True, exist_ok=True)
        autobox_path = autobox_dir / autobox_upload.name
        autobox_path.write_bytes(autobox_upload.getvalue())
        try:
            auto_box = compute_docking_box_from_file(autobox_path, padding=autobox_padding, min_size=autobox_min_size)
            st.session_state["vina_box_center_x"] = float(auto_box.center_x)
            st.session_state["vina_box_center_y"] = float(auto_box.center_y)
            st.session_state["vina_box_center_z"] = float(auto_box.center_z)
            st.session_state["vina_box_size_x"] = float(auto_box.size_x)
            st.session_state["vina_box_size_y"] = float(auto_box.size_y)
            st.session_state["vina_box_size_z"] = float(auto_box.size_z)
            st.session_state["box_center_x_input"] = float(auto_box.center_x)
            st.session_state["box_center_y_input"] = float(auto_box.center_y)
            st.session_state["box_center_z_input"] = float(auto_box.center_z)
            st.session_state["box_size_x_input"] = float(auto_box.size_x)
            st.session_state["box_size_y_input"] = float(auto_box.size_y)
            st.session_state["box_size_z_input"] = float(auto_box.size_z)
            st.session_state["vina_box_source"] = f"vina_auto_box:{autobox_upload.name}"
            st.success(
                "Auto-box set: "
                f"center=({auto_box.center_x:.3f}, {auto_box.center_y:.3f}, {auto_box.center_z:.3f}), "
                f"size=({auto_box.size_x:.3f}, {auto_box.size_y:.3f}, {auto_box.size_z:.3f})"
            )
        except Exception as exc:
            st.error(f"Could not determine box automatically: {exc}")

    st.markdown("**Docking box**")
    st.caption(f"Current box source: `{st.session_state.get('vina_box_source', 'manual')}`")
    c1, c2, c3 = st.columns(3)
    with c1:
        box_center_x = float(st.number_input("Box center X", value=float(st.session_state["vina_box_center_x"]), step=0.5, format="%.3f", key="box_center_x_input"))
        box_size_x = float(st.number_input("Box size X", min_value=1.0, value=float(st.session_state["vina_box_size_x"]), step=0.5, format="%.3f", key="box_size_x_input"))
    with c2:
        box_center_y = float(st.number_input("Box center Y", value=float(st.session_state["vina_box_center_y"]), step=0.5, format="%.3f", key="box_center_y_input"))
        box_size_y = float(st.number_input("Box size Y", min_value=1.0, value=float(st.session_state["vina_box_size_y"]), step=0.5, format="%.3f", key="box_size_y_input"))
    with c3:
        box_center_z = float(st.number_input("Box center Z", value=float(st.session_state["vina_box_center_z"]), step=0.5, format="%.3f", key="box_center_z_input"))
        box_size_z = float(st.number_input("Box size Z", min_value=1.0, value=float(st.session_state["vina_box_size_z"]), step=0.5, format="%.3f", key="box_size_z_input"))

    if st.button("Use current numbers as manual box"):
        st.session_state["vina_box_source"] = "vina_manual_box"
        st.info("Current numeric box will be recorded as manual box.")
    vina_exhaustiveness = int(st.number_input("Vina exhaustiveness", min_value=1, max_value=64, value=8, step=1))
    vina_num_modes = int(st.number_input("Vina num modes", min_value=1, max_value=20, value=1, step=1))
    vina_cpu = int(st.number_input("Vina CPU count, 0 = auto", min_value=0, max_value=64, value=0, step=1))
    vina_force_redock = st.checkbox("Force redock cached Vina poses", value=False)

    st.header("Exact complex graph settings")
    pocket_cutoff = float(
        st.number_input("Pocket selection cutoff (Å)", min_value=3.0, max_value=20.0, value=8.0, step=0.5)
    )
    contact_cutoff = float(
        st.number_input("Ligand–pocket contact cutoff (Å)", min_value=3.0, max_value=12.0, value=6.0, step=0.5)
    )


with screen_tab:
    st.subheader("1. Upload target protein")
    protein_upload = st.file_uploader("Protein PDB", type=["pdb"], accept_multiple_files=False)

    st.subheader("2. Upload ligand CSV")
    csv_upload = st.file_uploader("CSV with a SMILES column", type=["csv"], accept_multiple_files=False)
    st.caption("All extra CSV columns are preserved in the prediction output.")

    st.subheader("3. Optional exact ligand SDF poses")
    sdf_uploads = st.file_uploader(
        "Upload bound ligand SDF files only if they are exact protein-coordinate poses",
        type=["sdf"],
        accept_multiple_files=True,
    )
    st.caption("If graph/fusion is selected and a ligand SDF is missing, Vina can dock that ligand automatically when enabled.")

    run = st.button("Run screening", type="primary")

    if run:
        if protein_upload is None:
            st.error("Please upload a protein PDB.")
            st.stop()
        if csv_upload is None:
            st.error("Please upload a CSV with a SMILES column.")
            st.stop()
        if not config_path.exists():
            st.error(f"Config file not found: {config_path}")
            st.stop()

        cache = CacheManager(cache_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            protein_path = save_uploaded_file(protein_upload, tmpdir / protein_upload.name)
            protein_info = standardize_protein_from_pdb(protein_path)

            if not protein_info.sequence:
                st.error(f"Could not extract a protein sequence. {protein_info.warning or ''}")
                st.stop()

            st.success(
                f"Extracted standardized protein sequence: {protein_info.num_residues:,} residues "
                f"({protein_info.parser_used or 'parser unknown'})."
            )

            ligand_df = load_ligand_table(csv_upload)
            st.info(f"Loaded {len(ligand_df):,} ligands.")

            sdf_dir = tmpdir / "sdf"
            sdf_dir.mkdir(parents=True, exist_ok=True)
            sdf_paths_by_name = {}
            for up in sdf_uploads or []:
                p = save_uploaded_file(up, sdf_dir / up.name)
                sdf_paths_by_name[up.name] = p
                sdf_paths_by_name[Path(up.name).stem] = p

            if requested_model in GRAPH_MODELS and not sdf_paths_by_name and not vina_enabled:
                st.warning(
                    "You selected a graph-based/fusion model, but no exact ligand SDF poses were uploaded and Vina is disabled. "
                    "The app will fall back to sequence_only."
                )
            if requested_model in GRAPH_MODELS and vina_enabled and not vina_exe_path.exists():
                st.error(f"Vina executable not found: {vina_exe_path}")
                st.stop()

            progress = st.progress(0.0, text="Starting...")

            def progress_cb(frac: float, message: str) -> None:
                progress.progress(max(0.0, min(1.0, frac)), text=message)

            try:
                result_df = run_screening(
                    ligand_df=ligand_df,
                    protein_pdb=protein_path,
                    protein_sequence=protein_info.sequence,
                    requested_model=requested_model,
                    config_path=config_path,
                    cache=cache,
                    sdf_paths_by_name=sdf_paths_by_name,
                    device_choice=device_choice,
                    batch_size=batch_size,
                    pocket_cutoff=pocket_cutoff,
                    contact_cutoff=contact_cutoff,
                    progress_cb=progress_cb,
                    vina_enabled=vina_enabled and requested_model in GRAPH_MODELS,
                    vina_exe_path=vina_exe_path if vina_enabled else None,
                    vina_box=DockingBox(
                        center_x=box_center_x,
                        center_y=box_center_y,
                        center_z=box_center_z,
                        size_x=box_size_x,
                        size_y=box_size_y,
                        size_z=box_size_z,
                    ) if vina_enabled else None,
                    vina_exhaustiveness=vina_exhaustiveness,
                    vina_num_modes=vina_num_modes,
                    vina_cpu=vina_cpu,
                    vina_force_redock=vina_force_redock,
                    vina_pose_source_tag=st.session_state.get("vina_box_source", "vina_manual_box"),
                )
            except Exception as exc:
                st.exception(exc)
                st.stop()

            progress.progress(1.0, text="Done.")
            run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            out_path = output_dir / f"bindfusion_screening_{run_id}.csv"
            result_df.to_csv(out_path, index=False)

            st.subheader("Predictions")
            st.dataframe(result_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download predictions CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name=out_path.name,
                mime="text/csv",
            )
            st.caption(f"Saved local copy to `{out_path}`.")


with benchmark_tab:
    st.subheader("Hard-test benchmark")
    st.caption("Select one model at a time to view hard-test performance against the oracle reference image.")

    col0, col1 = st.columns([1, 2])
    with col0:
        benchmark_model = st.selectbox(
            "Model to view",
            AVAILABLE_MODELS,
            index=0,
            key="benchmark_model_select",
        )
    with col1:
        benchmark_root = Path(
            st.text_input(
                "Benchmark output root",
                "outputs/phase3",
                help="Folder containing model subfolders such as sequence_only, graph_only, concat_fusion, cross_attention_fusion.",
            )
        )

    oracle_path_text = st.text_input(
        "Oracle scatter image path, optional",
        "",
        help="Leave blank to auto-detect scatteroracle.png or scatter_oracle.png.",
    )
    oracle_path = Path(oracle_path_text) if oracle_path_text.strip() else None

    if st.button("Load selected benchmark"):
        metrics_df, predictions_df, missing = load_one_model_hard_test(
            output_root=benchmark_root,
            model_name=benchmark_model,
        )

        if missing:
            with st.expander("Missing/unchecked files", expanded=False):
                for item in missing:
                    st.write(f"- `{item}`")

        if predictions_df.empty:
            st.warning(f"No prediction CSV was found for `{benchmark_model}`.")
            st.stop()

        if not metrics_df.empty:
            st.markdown("### Summary metrics")
            display_metrics = metrics_df.copy()
            for c in ["rmse", "mae", "pearson_r", "spearman_r", "within_1", "within_2"]:
                if c in display_metrics.columns:
                    display_metrics[c] = pd.to_numeric(display_metrics[c], errors="coerce").round(4)
            st.dataframe(display_metrics, use_container_width=True, hide_index=True)

        st.markdown("### Truth vs predicted ΔG")

        left, right = st.columns(2)
        with left:
            st.caption(f"Selected model: `{benchmark_model}`")
            fig = make_truth_vs_pred_figure(predictions_df, benchmark_model)
            st.pyplot(fig, use_container_width=True)

        with right:
            resolved_oracle = oracle_path if oracle_path and oracle_path.exists() else find_oracle_scatter(Path("."))
            st.caption("Oracle reference")
            if resolved_oracle and resolved_oracle.exists():
                st.image(str(resolved_oracle), use_container_width=True)
                st.caption(f"`{resolved_oracle}`")
            else:
                st.info(
                    "Oracle scatter image was not found. Put `scatteroracle.png` or "
                    "`scatter_oracle.png` in the project root, or paste its path above."
                )

        st.markdown("### Per-complex predictions")
        display_preds = predictions_df.copy()
        for col in ["y_true", "y_pred", "abs_error"]:
            if col in display_preds.columns:
                display_preds[col] = pd.to_numeric(display_preds[col], errors="coerce").round(4)
        st.dataframe(display_preds, use_container_width=True, hide_index=True)

        st.download_button(
            "Download selected model hard-test predictions",
            data=predictions_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{benchmark_model}_hard_test_predictions.csv",
            mime="text/csv",
        )

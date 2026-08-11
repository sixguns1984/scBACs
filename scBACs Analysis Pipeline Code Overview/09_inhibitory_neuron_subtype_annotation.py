"""
CellTypist-based inhibitory-neuron subtype annotation for the AD/APOE4 analyses.

Final-analysis scope
--------------------
1. Train (or load) an Inh subtype classifier from the ROSMAP.MIT reference.
2. Transfer the classifier to independent replication datasets.
3. Retain both raw high-resolution predictions and an optional harmonized label.
4. Evaluate transfer using annotation-overlap scores when an independent reference
   label is available, and export canonical-marker dot plots.

Exploratory HVG/PCA/UMAP re-computation from the working script is omitted because
it is not required by the final Methods. Existing source-data UMAPs may still be
used for visualization outside this script.
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import scanpy as sc
import celltypist
from celltypist import models

# =============================================================================
# 1. CONFIGURATION — edit this section for your project
# =============================================================================

PROJECT_ROOT = Path(os.environ.get("SCBACS_PROJECT_ROOT", ".")).resolve()
DATA_DIR = PROJECT_ROOT / "dataset"
RESULT_DIR = PROJECT_ROOT / "results" / "Inh_subtype_annotation"
FIGURE_DIR = PROJECT_ROOT / "figures" / "Inh_subtype_annotation"
MODEL_DIR = PROJECT_ROOT / "models" / "Inh_subtype_annotation"
for p in [RESULT_DIR, FIGURE_DIR, MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

TRAIN_CLASSIFIER = False
REFERENCE_H5AD = DATA_DIR / "ROSMAP-MIT_Prefrontal-cortex.h5ad"
REFERENCE_CELLTYPE_COL = "major_cell_type"
REFERENCE_INH_VALUE = "Inh"
REFERENCE_LABEL_COL = "cell_type_high_resolution"
MODEL_PATH = MODEL_DIR / "Inh_Subclass_model.pkl"

# Transfer resources. Add GSE237718/GSE157827 files here if they are stored
# separately; the same classifier is applied without retraining.
TRANSFER_RESOURCES = [
    {
        "name": "ROSMAP_PFC_discovery",
        "h5ad": DATA_DIR / "sce_test.h5ad",
        "metadata": DATA_DIR / "meta_human_cortex_scrna_atlas_CT_NDDs.csv",
        "subset": {"celltype": "Inh", "dataset": "ROSMAP.MIT", "sub_tissue": "Prefrontal cortex"},
        "reference_label_col": "sub_celltype",
    },
    {
        "name": "SEAAD_PFC_replication",
        "h5ad": DATA_DIR / "SEAAD_A9_RNAseq_final-nuclei.2024-02-13.h5ad",
        "metadata": DATA_DIR / "meta_SEAAD_PFC_replication.csv",
        "subset": {},
        "reference_label_col": "Subclass",
    },
    {
        "name": "GSE263468_PFC_replication",
        "h5ad": DATA_DIR / "GSE263468_PFC_processed_CellXGene.h5ad",
        "metadata": DATA_DIR / "meta_GSE263468_PFC_AD_replication.csv",
        "subset": {"celltype": "Inh"},
        "reference_label_col": None,
    },
    {
        "name": "APOE4_replication",
        "h5ad": DATA_DIR / "sce_APOE4_aging_replication.h5ad",
        "metadata": DATA_DIR / "meta_APOE4_Temporal_cortex_replication.csv",
        "subset": {"celltype": "Inh"},
        "reference_label_col": None,
    },
]

# This mapping is retained only as an explicit coarse harmonization used in the
# working analysis for overlap checks. The raw CellTypist prediction is always
# preserved and should be used when the final 13 high-resolution subtype labels
# are required. Do not silently collapse the raw labels downstream.
COARSE_ROSMAP_MAP = {
    "Inh CUX2 MSR1": "Inh SST",
    "Inh ENOX2 SPHKAP": "Inh SST",
    "Inh L1 PAX6 CA4": "Inh PAX6",
    "Inh L1-2 PAX6 SCGN": "Inh PAX6",
    "Inh L3-5 SST MAFB": "Inh SST",
    "Inh L5-6 PVALB STON2": "Inh PVALB",
    "Inh LAMP5 NRG1 (Rosehip)": "Inh LAMP5",
    "Inh LAMP5 RELN": "Inh LAMP5",
    "Inh PTPRK FAM19A1": "Inh LAMP5",
    "Inh PVALB HTR4": "Inh PVALB",
    "Inh PVALB SULF1": "Inh PVALB",
    "Inh SORCS1 TTN": "Inh LAMP5",
    "Inh VIP ABI3BP": "Inh VIP",
    "Inh VIP CLSTN2": "Inh VIP",
    "Inh VIP THSD7B": "Inh VIP",
    "Inh VIP TSHZ2": "Inh VIP",
}
SEAAD_COARSE_MAP = {
    "Sst": "Inh SST", "Sst Chodl": "Inh L6 SST NPY", "Pvalb": "Inh PVALB",
    "Lamp5": "Inh LAMP5", "Lamp5 Lhx6": "Inh L1-6 LAMP5 CA13",
    "Chandelier": "Inh PVALB CA8 (Chandelier)", "Vip": "Inh VIP",
    "Pax6": "Inh PAX6", "Sncg": "Inh LAMP5",
}
MARKER_GENES = [
    "ALCAM", "TRPM3", "CUX2", "MSR1", "ENOX2", "SPHKAP", "FBN2",
    "EPB41L4A", "GPC5", "RIT2", "PAX6", "CA4", "SCGN", "LAMP5", "CA13",
    "SST", "MAFB", "PVALB", "STON2", "TH", "NPY", "NRG1", "RELN", "PTPRK",
    "FAM19A1", "CA8", "HTR4", "SULF1", "RYR3", "TSHZ2", "SGCD", "PDE3A",
    "SORCS1", "TTN", "VIP", "ABI3BP", "CLSTN2", "THSD7B",
]

# =============================================================================
# 2. HELPERS
# =============================================================================

def normalize_for_celltypist(adata):
    out = adata.copy()
    out.X = out.X.astype(np.float32)
    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)
    return out


def apply_obs_subset(adata, rules):
    mask = np.ones(adata.n_obs, dtype=bool)
    for col, value in rules.items():
        if col not in adata.obs.columns:
            continue
        mask &= adata.obs[col].astype(str).eq(str(value)).to_numpy()
    return adata[mask].copy()


def align_metadata(adata, metadata_path):
    if metadata_path is None or not Path(metadata_path).exists():
        return adata
    meta = pd.read_csv(metadata_path, index_col=0)
    ids = adata.obs_names.intersection(meta.index.astype(str))
    out = adata[ids].copy()
    meta = meta.loc[ids]
    for col in meta.columns:
        out.obs[col] = meta[col].values
    return out


def annotation_overlap(predicted, reference):
    tab = pd.crosstab(predicted, reference, normalize="index")
    rows = []
    for label in tab.index:
        score = tab.loc[label, label] if label in tab.columns else 0.0
        rows.append({"label": label, "overlap_score": float(score), "n_predicted": int((predicted == label).sum())})
    return pd.DataFrame(rows), tab

# =============================================================================
# 3. TRAIN OR LOAD CLASSIFIER
# =============================================================================

if TRAIN_CLASSIFIER:
    reference = sc.read_h5ad(REFERENCE_H5AD)
    if REFERENCE_CELLTYPE_COL in reference.obs.columns:
        reference = reference[reference.obs[REFERENCE_CELLTYPE_COL].astype(str) == REFERENCE_INH_VALUE].copy()
    reference = normalize_for_celltypist(reference)
    classifier = celltypist.train(
        reference,
        labels=REFERENCE_LABEL_COL,
        check_expression=False,
        feature_selection=True,
    )
    classifier.write(MODEL_PATH)
else:
    classifier = models.Model.load(MODEL_PATH)

# =============================================================================
# 4. TRANSFER TO DISCOVERY / REPLICATION RESOURCES
# =============================================================================

all_overlap = []
for cfg in TRANSFER_RESOURCES:
    if not Path(cfg["h5ad"]).exists():
        print("Skipping missing resource:", cfg["name"], cfg["h5ad"])
        continue

    print("\nAnnotating", cfg["name"])
    adata = sc.read_h5ad(cfg["h5ad"])
    adata = align_metadata(adata, cfg.get("metadata"))
    adata = apply_obs_subset(adata, cfg.get("subset", {}))
    if adata.n_obs == 0:
        print("  no eligible Inh cells; skipped")
        continue

    adata_norm = normalize_for_celltypist(adata)
    pred = celltypist.annotate(adata_norm, model=classifier, majority_voting=False).to_adata()
    pred.obs["sub_celltype_celltypist_raw"] = pred.obs["predicted_labels"].astype(str)
    pred.obs["sub_celltype_celltypist_coarse"] = pred.obs["sub_celltype_celltypist_raw"].replace(COARSE_ROSMAP_MAP)

    # Reattach original metadata fields that CellTypist may not copy identically.
    for col in adata.obs.columns:
        pred.obs[col] = adata.obs.loc[pred.obs_names, col].values

    ref_col = cfg.get("reference_label_col")
    if ref_col and ref_col in pred.obs.columns:
        reference_labels = pred.obs[ref_col].astype(str)
        if cfg["name"].startswith("SEAAD"):
            reference_labels = reference_labels.replace(SEAAD_COARSE_MAP)
        else:
            reference_labels = reference_labels.replace(COARSE_ROSMAP_MAP)
        scores, matrix = annotation_overlap(pred.obs["sub_celltype_celltypist_coarse"], reference_labels)
        scores["resource"] = cfg["name"]
        all_overlap.append(scores)
        matrix.to_csv(RESULT_DIR / f"{cfg['name']}_annotation_overlap_matrix.csv")

    out_meta = pred.obs.copy()
    out_meta.to_csv(RESULT_DIR / f"{cfg['name']}_Inh_subtype_metadata.csv")

    present_markers = [g for g in MARKER_GENES if g in pred.var_names]
    if present_markers:
        sc.settings.figdir = FIGURE_DIR
        sc.pl.dotplot(
            pred,
            var_names=present_markers,
            groupby="sub_celltype_celltypist_raw",
            standard_scale="var",
            use_raw=False,
            show=False,
            save=f"_{cfg['name']}_canonical_markers.pdf",
        )

if all_overlap:
    pd.concat(all_overlap, ignore_index=True).to_csv(
        RESULT_DIR / "Inh_subtype_annotation_overlap_scores.csv", index=False
    )

print("\nDone. Raw high-resolution CellTypist labels are preserved in sub_celltype_celltypist_raw.")

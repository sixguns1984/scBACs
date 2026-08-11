"""
Human Cortex Single-Cell Transcriptomic Atlas Cell Type Annotation Pipeline
===========================================================================

Purpose:
- Re-annotate major CNS cell types using scANVI, CellTypist, and CellAssign
- Retain cells with concordant predictions across all three methods
- Refine neurons, vascular mural cells, and CNS-resident macrophages using scANVI

Reference dataset:
- ROSMAP.MIT

Final major-cell-type scANVI model:
- n_latent = 10
- n_layers = 4

Python version:
- Python 3.9.20

Usage:
- Edit paths in Section 2 and run the script section by section
- Command-line execution is also supported; examples are provided at the end
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scanpy as sc
import scvi
import celltypist

from celltypist import models
from scvi.external import CellAssign


# ============================================================================
# 2. PATHS AND BASIC SETTINGS
# ============================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "celltype_annotation"
)

MODEL_DIR = os.path.join(
    OUTPUT_DIR,
    "models"
)

METRIC_DIR = os.path.join(
    OUTPUT_DIR,
    "metrics"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    METRIC_DIR,
    exist_ok=True
)

INPUT_H5AD_FILES = [
    os.path.join(DATA_DIR, "sce_train.h5ad"),
    os.path.join(DATA_DIR, "sce_test1_0.h5ad"),
    os.path.join(DATA_DIR, "sce_test1_1.h5ad"),
    os.path.join(DATA_DIR, "sce_test2.h5ad"),
    os.path.join(DATA_DIR, "sce_test3.h5ad"),
]

MARKER_FILE = os.path.join(
    DATA_DIR,
    "cellassign_human_brain_vascular_AD_cell_types_markers.csv"
)

REFERENCE_DATASET = "ROSMAP.MIT"

DATASET_COL = "dataset"
DONOR_COL = "donor_id"
CELLTYPE_COL = "celltype"

N_HVG = 2000

# Final major-cell-type scANVI parameters used in the manuscript
MAJOR_N_LATENT = 10
MAJOR_N_LAYERS = 4

# Set True only if you want to repeat the exploratory grid search
RUN_HYPERPARAMETER_TUNING = False

# The final atlas retains only three-method concordant major annotations
KEEP_DISCORDANT_CELLS = False

scvi.settings.seed = 0
scvi.settings.num_threads = 4
np.random.seed(0)

sc.set_figure_params(
    figsize=(6, 6),
    frameon=False
)


# ============================================================================
# OPTIONAL COMMAND-LINE PATH OVERRIDE
# ============================================================================

if __name__ == "__main__" and len(sys.argv) > 1:
    parser = argparse.ArgumentParser(
        description="Consensus cell-type annotation for the scBACs atlas."
    )

    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input h5ad files contributing to the integrated atlas."
    )

    parser.add_argument(
        "--marker-file",
        required=True,
        help="CellAssign marker matrix CSV."
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory."
    )

    args = parser.parse_args()

    INPUT_H5AD_FILES = args.input
    MARKER_FILE = args.marker_file
    OUTPUT_DIR = os.path.abspath(
        args.output_dir
    )

    MODEL_DIR = os.path.join(
        OUTPUT_DIR,
        "models"
    )

    METRIC_DIR = os.path.join(
        OUTPUT_DIR,
        "metrics"
    )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        METRIC_DIR,
        exist_ok=True
    )


# ============================================================================
# 3. LOAD AND MERGE HUMAN CORTEX DATASETS
# ============================================================================

print("\n" + "=" * 70)
print("Loading human cortex single-cell datasets")
print("=" * 70)

adata_list = []

for file_path in INPUT_H5AD_FILES:

    print(
        "Loading:",
        file_path
    )

    temp_adata = sc.read_h5ad(
        file_path
    )

    adata_list.append(
        temp_adata
    )

adata_raw = sc.concat(
    adata_list,
    axis=0,
    join="inner",
    merge="same",
    index_unique=None
)

adata_raw.var_names_make_unique()

print(
    "Merged data shape:",
    adata_raw.shape
)

print(
    "\nDataset distribution:"
)

print(
    adata_raw.obs[
        DATASET_COL
    ].value_counts()
)

# Keep the merged raw-count atlas for later methods
adata_raw.write_h5ad(
    os.path.join(
        OUTPUT_DIR,
        "human_cortex_integrated_before_annotation.h5ad"
    ),
    compression="gzip"
)


# ============================================================================
# 4. PREPARE MAJOR CELL-TYPE REFERENCE LABELS
# ============================================================================

print("\n" + "=" * 70)
print("Preparing major cell-type reference labels")
print("=" * 70)

adata_raw.obs[
    "celltype_original"
] = (
    adata_raw.obs[
        CELLTYPE_COL
    ]
    .astype(str)
)

adata_raw.obs[
    "celltype_major_ref"
] = (
    adata_raw.obs[
        CELLTYPE_COL
    ]
    .astype(str)
)

adata_raw.obs[
    "celltype_major_ref"
] = adata_raw.obs[
    "celltype_major_ref"
].replace(
    {
        "Exc":
            "Neuron",

        "Inh":
            "Neuron",

        "Mic":
            "CNS-macrophage",

        "CAM":
            "CNS-macrophage",

        "Per":
            "Per/SMC/Mural",

        "SMC":
            "Per/SMC/Mural",

        "Mural":
            "Per/SMC/Mural",

        "T cell":
            "T_cell",

        "T cells":
            "T_cell"
    }
)

print(
    adata_raw.obs[
        "celltype_major_ref"
    ].value_counts()
)


# ============================================================================
# 5. HIGHLY VARIABLE GENE SELECTION FOR MAJOR SCANVI
# ============================================================================

print("\n" + "=" * 70)
print("Selecting highly variable genes")
print("=" * 70)

# Select HVGs on a normalized copy.
# The scVI/scANVI model itself is trained on the corresponding raw counts.
adata_hvg = adata_raw.copy()

sc.pp.normalize_total(
    adata_hvg
)

sc.pp.log1p(
    adata_hvg
)

sc.pp.highly_variable_genes(
    adata_hvg,
    n_top_genes=N_HVG,
    batch_key=DONOR_COL
)

hvg_genes = (
    adata_hvg.var_names[
        adata_hvg.var[
            "highly_variable"
        ]
    ]
    .tolist()
)

print(
    "Selected HVGs:",
    len(hvg_genes)
)

adata_scanvi = (
    adata_raw[
        :,
        hvg_genes
    ]
    .copy()
)

# ROSMAP.MIT is the reference dataset.
# Cells from all other datasets are treated as unlabeled.
adata_scanvi.obs[
    "celltype_scanvi_train"
] = adata_scanvi.obs[
    "celltype_major_ref"
].copy()

adata_scanvi.obs.loc[
    adata_scanvi.obs[
        DATASET_COL
    ] != REFERENCE_DATASET,
    "celltype_scanvi_train"
] = "Unknown"

adata_scanvi.obs[
    "celltype_scanvi_train"
] = adata_scanvi.obs[
    "celltype_scanvi_train"
].astype(
    "category"
)

print(
    adata_scanvi.obs[
        "celltype_scanvi_train"
    ].value_counts()
)


# ============================================================================
# 6. OPTIONAL SCANVI HYPERPARAMETER TESTING
# ============================================================================

if RUN_HYPERPARAMETER_TUNING:

    print("\n" + "=" * 70)
    print("Testing major-cell-type scANVI parameters")
    print("=" * 70)

    n_latent_list = [
        10,
        20,
        30,
        40,
        50
    ]

    n_layers_list = [
        1,
        2,
        3,
        4,
        5
    ]

    tuning_results = []

    for n_latent in n_latent_list:

        for n_layers in n_layers_list:

            print(
                "Testing n_latent={}, n_layers={}".format(
                    n_latent,
                    n_layers
                )
            )

            adata_temp = adata_scanvi.copy()

            scvi.model.SCVI.setup_anndata(
                adata_temp,
                batch_key=DONOR_COL
            )

            scvi_model_temp = scvi.model.SCVI(
                adata_temp,
                n_latent=n_latent,
                n_layers=n_layers,
                gene_likelihood="nb"
            )

            scvi_model_temp.train(
                5,
                train_size=5 / 6,
                early_stopping=True,
                early_stopping_patience=5,
                check_val_every_n_epoch=2
            )

            scanvi_model_temp = (
                scvi.model.SCANVI
                .from_scvi_model(
                    scvi_model_temp,
                    labels_key="celltype_scanvi_train",
                    unlabeled_category="Unknown"
                )
            )

            try:

                scanvi_model_temp.train(
                    20,
                    train_size=5 / 6,
                    early_stopping=True,
                    early_stopping_patience=5,
                    check_val_every_n_epoch=2,
                    plan_kwargs={
                        "lr":
                            0.0001,

                        "reduce_lr_on_plateau":
                            True,

                        "lr_factor":
                            0.1,

                        "lr_patience":
                            8
                    }
                )

                validation_elbo = (
                    scanvi_model_temp.history[
                        "elbo_validation"
                    ]
                    .min()
                    .iloc[0]
                )

                if "validation_accuracy" in scanvi_model_temp.history:
                    validation_accuracy = (
                        scanvi_model_temp.history[
                            "validation_accuracy"
                        ]
                        .max()
                        .iloc[0]
                    )
                else:
                    validation_accuracy = np.nan

            except Exception as error:

                print(
                    "Training failed:",
                    error
                )

                validation_elbo = np.nan
                validation_accuracy = np.nan

            tuning_results.append(
                {
                    "n_latent":
                        n_latent,

                    "n_layers":
                        n_layers,

                    "validation_accuracy":
                        validation_accuracy,

                    "validation_elbo":
                        validation_elbo
                }
            )

    tuning_results = pd.DataFrame(
        tuning_results
    )

    tuning_results.to_csv(
        os.path.join(
            METRIC_DIR,
            "major_scanvi_hyperparameter_testing.csv"
        ),
        index=False
    )

    print(
        tuning_results
    )


# ============================================================================
# 7. FINAL MAJOR CELL-TYPE SCANVI MODEL
# ============================================================================

print("\n" + "=" * 70)
print("Training final major-cell-type scANVI model")
print("=" * 70)

print(
    "Final parameters: n_latent={}, n_layers={}".format(
        MAJOR_N_LATENT,
        MAJOR_N_LAYERS
    )
)

scvi.model.SCVI.setup_anndata(
    adata_scanvi,
    batch_key=DONOR_COL
)

scvi_model = scvi.model.SCVI(
    adata_scanvi,
    n_latent=MAJOR_N_LATENT,
    n_layers=MAJOR_N_LAYERS,
    gene_likelihood="nb"
)

scvi_model.train(
    5,
    train_size=5 / 6,
    early_stopping=True,
    early_stopping_patience=5,
    check_val_every_n_epoch=2
)

scvi_model.save(
    os.path.join(
        MODEL_DIR,
        "major_scvi_latent10_layer4"
    ),
    save_anndata=False,
    overwrite=True
)

scanvi_model = (
    scvi.model.SCANVI
    .from_scvi_model(
        scvi_model,
        labels_key="celltype_scanvi_train",
        unlabeled_category="Unknown"
    )
)

scanvi_model.train(
    100,
    train_size=5 / 6,
    early_stopping=True,
    early_stopping_patience=5,
    check_val_every_n_epoch=2,
    plan_kwargs={
        "lr":
            0.0001,

        "reduce_lr_on_plateau":
            True,

        "lr_factor":
            0.1,

        "lr_patience":
            8
    }
)

scanvi_model.save(
    os.path.join(
        MODEL_DIR,
        "major_scanvi_latent10_layer4"
    ),
    save_anndata=False,
    overwrite=True
)

adata_raw.obs[
    "scanvi_major"
] = scanvi_model.predict(
    adata_scanvi
)

adata_raw.obsm[
    "X_scANVI_major"
] = scanvi_model.get_latent_representation(
    adata_scanvi
)

print(
    adata_raw.obs[
        "scanvi_major"
    ].value_counts()
)

scanvi_model.history[
    "elbo_validation"
].plot()

plt.title(
    "Major cell-type scANVI training"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Validation ELBO"
)

plt.tight_layout()
plt.show()


# ============================================================================
# 8. CELLTYPIST MAJOR CELL-TYPE ANNOTATION
# ============================================================================

print("\n" + "=" * 70)
print("Major cell-type annotation using CellTypist")
print("=" * 70)

adata_celltypist = adata_raw.copy()

sc.pp.normalize_total(
    adata_celltypist
)

sc.pp.log1p(
    adata_celltypist
)

# Train the classifier using ROSMAP.MIT reference cells
adata_celltypist_train = adata_celltypist[
    adata_celltypist.obs[
        DATASET_COL
    ] == REFERENCE_DATASET,
    :
].copy()

print(
    "CellTypist reference cells:",
    adata_celltypist_train.n_obs
)

celltypist_model = celltypist.train(
    adata_celltypist_train,
    labels="celltype_major_ref",
    check_expression=False,
    feature_selection=True
)

celltypist_model_file = os.path.join(
    MODEL_DIR,
    "ROSMAP_MIT_major_celltype_CellTypist.pkl"
)

celltypist_model.write(
    celltypist_model_file
)

celltypist_model_loaded = models.Model.load(
    celltypist_model_file
)

celltypist_result = celltypist.annotate(
    adata_celltypist,
    model=celltypist_model_loaded,
    majority_voting=False
)

celltypist_adata = (
    celltypist_result
    .to_adata()
)

adata_raw.obs[
    "celltypist_major"
] = (
    celltypist_adata.obs[
        "predicted_labels"
    ]
    .astype(str)
    .values
)

print(
    adata_raw.obs[
        "celltypist_major"
    ].value_counts()
)


# ============================================================================
# 9. CELLASSIGN MAJOR CELL-TYPE ANNOTATION
# ============================================================================

print("\n" + "=" * 70)
print("Major cell-type annotation using CellAssign")
print("=" * 70)

marker = pd.read_csv(
    MARKER_FILE,
    index_col=0
)

common_marker_genes = np.intersect1d(
    marker.index,
    adata_raw.var_names
)

print(
    "CellAssign marker genes found:",
    len(common_marker_genes)
)

adata_cellassign = (
    adata_raw[
        :,
        common_marker_genes
    ]
    .copy()
)

marker = marker.loc[
    common_marker_genes,
    :
].copy()

library_size = np.asarray(
    adata_cellassign.X.sum(
        axis=1
    )
).ravel()

adata_cellassign.obs[
    "size_factor"
] = (
    library_size
    / np.mean(
        library_size
    )
)

CellAssign.setup_anndata(
    adata_cellassign,
    size_factor_key="size_factor"
)

cellassign_model = CellAssign(
    adata_cellassign,
    marker
)

cellassign_model.train(
    max_epochs=100,
    train_size=5 / 6,
    early_stopping=True,
    early_stopping_patience=5,
    check_val_every_n_epoch=2,
    plan_kwargs={
        "lr":
            0.003,

        "reduce_lr_on_plateau":
            True,

        "lr_factor":
            0.1,

        "lr_patience":
            8
    }
)

cellassign_probability = (
    cellassign_model
    .predict()
)

adata_raw.obs[
    "cellassign_major"
] = (
    cellassign_probability
    .idxmax(
        axis=1
    )
    .astype(str)
    .values
)

print(
    adata_raw.obs[
        "cellassign_major"
    ].value_counts()
)


# ============================================================================
# 10. HARMONIZE PREDICTION NAMES
# ============================================================================

prediction_name_map = {
    "Astrocyte":
        "Ast",

    "Astrocytes":
        "Ast",

    "Oligodendrocyte":
        "Oli",

    "Oligodendrocytes":
        "Oli",

    "Oligodendrocyte precursor cell":
        "OPC",

    "Oligodendrocyte precursor cells":
        "OPC",

    "Microglia":
        "CNS-macrophage",

    "Macrophage":
        "CNS-macrophage",

    "CNS macrophage":
        "CNS-macrophage",

    "Endothelial":
        "End",

    "Endothelial cell":
        "End",

    "Fibroblast":
        "Fib",

    "T cell":
        "T_cell",

    "T cells":
        "T_cell",

    "Pericyte/SMC/Mural":
        "Per/SMC/Mural"
}

for col in [
    "scanvi_major",
    "celltypist_major",
    "cellassign_major"
]:
    adata_raw.obs[
        col
    ] = (
        adata_raw.obs[
            col
        ]
        .astype(str)
        .replace(
            prediction_name_map
        )
    )


# ============================================================================
# 11. MODEL EVALUATION
# ============================================================================

print("\n" + "=" * 70)
print("Major cell-type annotation accuracy")
print("=" * 70)

evaluation_mask = (
    adata_raw.obs[
        "celltype_major_ref"
    ]
    .notna()
)

annotation_accuracy = []

for prediction_col in [
    "scanvi_major",
    "celltypist_major",
    "cellassign_major"
]:

    evaluation_df = adata_raw.obs.loc[
        evaluation_mask,
        [
            "celltype_major_ref",
            prediction_col
        ]
    ].copy()

    overall_accuracy = np.mean(
        evaluation_df[
            "celltype_major_ref"
        ].values
        ==
        evaluation_df[
            prediction_col
        ].values
    )

    annotation_accuracy.append(
        {
            "method":
                prediction_col,

            "accuracy":
                overall_accuracy
        }
    )

    confusion_matrix = pd.crosstab(
        evaluation_df[
            "celltype_major_ref"
        ],
        evaluation_df[
            prediction_col
        ],
        normalize="index"
    )

    confusion_matrix.to_csv(
        os.path.join(
            METRIC_DIR,
            "{}_confusion_matrix.csv".format(
                prediction_col
            )
        )
    )

    print(
        "\n{} accuracy: {:.4f}".format(
            prediction_col,
            overall_accuracy
        )
    )

annotation_accuracy = pd.DataFrame(
    annotation_accuracy
)

annotation_accuracy.to_csv(
    os.path.join(
        METRIC_DIR,
        "major_celltype_annotation_accuracy.csv"
    ),
    index=False
)


# ============================================================================
# 12. THREE-METHOD CONSENSUS ANNOTATION
# ============================================================================

print("\n" + "=" * 70)
print("Three-method consensus annotation")
print("=" * 70)

prediction_cols = [
    "scanvi_major",
    "celltypist_major",
    "cellassign_major"
]

consensus_mask = (
    adata_raw.obs[
        prediction_cols
    ]
    .nunique(
        axis=1
    )
    ==
    1
)

adata_raw.obs[
    "major_celltype_consensus"
] = np.nan

adata_raw.obs.loc[
    consensus_mask,
    "major_celltype_consensus"
] = (
    adata_raw.obs.loc[
        consensus_mask,
        "scanvi_major"
    ]
)

print(
    "Concordant cells: {}/{} ({:.2f}%)".format(
        int(
            consensus_mask.sum()
        ),
        adata_raw.n_obs,
        consensus_mask.mean() * 100
    )
)

print(
    adata_raw.obs[
        "major_celltype_consensus"
    ].value_counts()
)

adata_raw.obs[
    "celltype_consensus"
] = (
    adata_raw.obs[
        "major_celltype_consensus"
    ]
)


# ============================================================================
# 13. NEURON REFINEMENT: NEURON -> EXC / INH
# ============================================================================

print("\n" + "=" * 70)
print("Refining neurons into Exc and Inh")
print("=" * 70)

neuron = adata_raw[
    adata_raw.obs[
        "major_celltype_consensus"
    ] == "Neuron",
    :
].copy()

neuron_hvg = neuron.copy()

sc.pp.normalize_total(
    neuron_hvg
)

sc.pp.log1p(
    neuron_hvg
)

sc.pp.highly_variable_genes(
    neuron_hvg,
    n_top_genes=min(
        N_HVG,
        neuron_hvg.n_vars
    ),
    batch_key=DONOR_COL
)

neuron_genes = neuron_hvg.var_names[
    neuron_hvg.var[
        "highly_variable"
    ]
].tolist()

neuron = neuron[
    :,
    neuron_genes
].copy()

neuron.obs[
    "celltype_scanvi_train"
] = neuron.obs[
    "celltype_original"
].copy()

neuron.obs.loc[
    neuron.obs[
        DATASET_COL
    ] != REFERENCE_DATASET,
    "celltype_scanvi_train"
] = "Unknown"

neuron.obs.loc[
    ~neuron.obs[
        "celltype_scanvi_train"
    ].isin(
        [
            "Exc",
            "Inh",
            "Unknown"
        ]
    ),
    "celltype_scanvi_train"
] = "Unknown"

neuron.obs[
    "celltype_scanvi_train"
] = neuron.obs[
    "celltype_scanvi_train"
].astype(
    "category"
)

scvi.model.SCVI.setup_anndata(
    neuron,
    batch_key=DONOR_COL
)

neuron_scvi = scvi.model.SCVI(
    neuron,
    n_latent=10,
    n_layers=4,
    gene_likelihood="nb"
)

neuron_scvi.train(
    5,
    early_stopping=True,
    early_stopping_patience=5,
    check_val_every_n_epoch=2
)

neuron_scanvi = (
    scvi.model.SCANVI
    .from_scvi_model(
        neuron_scvi,
        labels_key="celltype_scanvi_train",
        unlabeled_category="Unknown"
    )
)

neuron_scanvi.train(
    100,
    early_stopping=True,
    early_stopping_patience=5,
    check_val_every_n_epoch=2,
    plan_kwargs={
        "lr":
            0.001,

        "reduce_lr_on_plateau":
            True,

        "lr_factor":
            0.1,

        "lr_patience":
            8
    }
)

neuron.obs[
    "celltype_refined"
] = neuron_scanvi.predict(
    neuron
)

adata_raw.obs.loc[
    neuron.obs_names,
    "celltype_consensus"
] = neuron.obs[
    "celltype_refined"
].values

print(
    neuron.obs[
        "celltype_refined"
    ].value_counts()
)


# ============================================================================
# 14. VASCULAR REFINEMENT: PER / SMC / MURAL
# ============================================================================

print("\n" + "=" * 70)
print("Refining Per/SMC/Mural cells")
print("=" * 70)

vascular = adata_raw[
    adata_raw.obs[
        "major_celltype_consensus"
    ] == "Per/SMC/Mural",
    :
].copy()

vascular_hvg = vascular.copy()

sc.pp.normalize_total(
    vascular_hvg
)

sc.pp.log1p(
    vascular_hvg
)

sc.pp.highly_variable_genes(
    vascular_hvg,
    n_top_genes=min(
        N_HVG,
        vascular_hvg.n_vars
    ),
    batch_key=DONOR_COL
)

vascular_genes = vascular_hvg.var_names[
    vascular_hvg.var[
        "highly_variable"
    ]
].tolist()

vascular = vascular[
    :,
    vascular_genes
].copy()

vascular.obs[
    "celltype_scanvi_train"
] = vascular.obs[
    "celltype_original"
].copy()

vascular.obs.loc[
    vascular.obs[
        DATASET_COL
    ] != REFERENCE_DATASET,
    "celltype_scanvi_train"
] = "Unknown"

vascular.obs.loc[
    ~vascular.obs[
        "celltype_scanvi_train"
    ].isin(
        [
            "Per",
            "SMC",
            "Mural",
            "Unknown"
        ]
    ),
    "celltype_scanvi_train"
] = "Unknown"

vascular.obs[
    "celltype_scanvi_train"
] = vascular.obs[
    "celltype_scanvi_train"
].astype(
    "category"
)

scvi.model.SCVI.setup_anndata(
    vascular,
    batch_key=DONOR_COL
)

vascular_scvi = scvi.model.SCVI(
    vascular,
    n_latent=50,
    n_layers=2,
    gene_likelihood="nb"
)

vascular_scvi.train(
    5,
    early_stopping=True,
    early_stopping_patience=5,
    check_val_every_n_epoch=2
)

vascular_scanvi = (
    scvi.model.SCANVI
    .from_scvi_model(
        vascular_scvi,
        labels_key="celltype_scanvi_train",
        unlabeled_category="Unknown"
    )
)

vascular_scanvi.train(
    100,
    early_stopping=True,
    early_stopping_patience=5,
    check_val_every_n_epoch=2,
    plan_kwargs={
        "lr":
            0.001,

        "reduce_lr_on_plateau":
            True,

        "lr_factor":
            0.1,

        "lr_patience":
            8
    }
)

vascular.obs[
    "celltype_refined"
] = vascular_scanvi.predict(
    vascular
)

adata_raw.obs.loc[
    vascular.obs_names,
    "celltype_consensus"
] = vascular.obs[
    "celltype_refined"
].values

print(
    vascular.obs[
        "celltype_refined"
    ].value_counts()
)


# ============================================================================
# 15. MACROPHAGE REFINEMENT: CNS-MACROPHAGE -> MIC / CAM
# ============================================================================

print("\n" + "=" * 70)
print("Refining CNS-resident macrophages into Mic and CAM")
print("=" * 70)

macrophage = adata_raw[
    adata_raw.obs[
        "major_celltype_consensus"
    ] == "CNS-macrophage",
    :
].copy()

macrophage_hvg = macrophage.copy()

sc.pp.normalize_total(
    macrophage_hvg
)

sc.pp.log1p(
    macrophage_hvg
)

sc.pp.highly_variable_genes(
    macrophage_hvg,
    n_top_genes=min(
        N_HVG,
        macrophage_hvg.n_vars
    ),
    batch_key=DONOR_COL
)

macrophage_genes = macrophage_hvg.var_names[
    macrophage_hvg.var[
        "highly_variable"
    ]
].tolist()

macrophage = macrophage[
    :,
    macrophage_genes
].copy()

macrophage.obs[
    "celltype_scanvi_train"
] = macrophage.obs[
    "celltype_original"
].copy()

macrophage.obs.loc[
    macrophage.obs[
        DATASET_COL
    ] != REFERENCE_DATASET,
    "celltype_scanvi_train"
] = "Unknown"

macrophage.obs.loc[
    ~macrophage.obs[
        "celltype_scanvi_train"
    ].isin(
        [
            "Mic",
            "CAM",
            "Unknown"
        ]
    ),
    "celltype_scanvi_train"
] = "Unknown"

macrophage.obs[
    "celltype_scanvi_train"
] = macrophage.obs[
    "celltype_scanvi_train"
].astype(
    "category"
)

scvi.model.SCVI.setup_anndata(
    macrophage,
    batch_key=DONOR_COL
)

macrophage_scvi = scvi.model.SCVI(
    macrophage,
    n_latent=20,
    n_layers=2,
    gene_likelihood="nb"
)

macrophage_scvi.train(
    5,
    early_stopping=True,
    early_stopping_patience=5,
    check_val_every_n_epoch=2
)

macrophage_scanvi = (
    scvi.model.SCANVI
    .from_scvi_model(
        macrophage_scvi,
        labels_key="celltype_scanvi_train",
        unlabeled_category="Unknown"
    )
)

macrophage_scanvi.train(
    100,
    early_stopping=True,
    early_stopping_patience=5,
    check_val_every_n_epoch=2,
    plan_kwargs={
        "lr":
            0.001,

        "reduce_lr_on_plateau":
            True,

        "lr_factor":
            0.1,

        "lr_patience":
            8
    }
)

macrophage.obs[
    "celltype_refined"
] = macrophage_scanvi.predict(
    macrophage
)

adata_raw.obs.loc[
    macrophage.obs_names,
    "celltype_consensus"
] = macrophage.obs[
    "celltype_refined"
].values

print(
    macrophage.obs[
        "celltype_refined"
    ].value_counts()
)


# ============================================================================
# 16. FINAL CELL-TYPE ANNOTATION
# ============================================================================

print("\n" + "=" * 70)
print("Final consensus annotation")
print("=" * 70)

print(
    adata_raw.obs[
        "celltype_consensus"
    ].value_counts(
        dropna=False
    )
)

if KEEP_DISCORDANT_CELLS:
    adata_final = adata_raw.copy()
else:
    adata_final = adata_raw[
        consensus_mask,
        :
    ].copy()

print(
    "Final atlas shape:",
    adata_final.shape
)


# ============================================================================
# 17. SAVE FINAL ANNOTATED ATLAS
# ============================================================================

OUTPUT_H5AD = os.path.join(
    OUTPUT_DIR,
    "human_cortex_atlas_consensus_annotated.h5ad"
)

adata_final.write_h5ad(
    OUTPUT_H5AD,
    compression="gzip"
)

print(
    "Saved final annotated atlas:",
    OUTPUT_H5AD
)


# ============================================================================
# 18. COMMAND-LINE USAGE
# ============================================================================

"""
Example:

python 02_consensus_celltype_annotation.py \
    --input \
        /path/to/sce_train.h5ad \
        /path/to/sce_test1_0.h5ad \
        /path/to/sce_test1_1.h5ad \
        /path/to/sce_test2.h5ad \
        /path/to/sce_test3.h5ad \
    --marker-file \
        /path/to/cellassign_human_brain_vascular_AD_cell_types_markers.csv \
    --output-dir \
        /path/to/results/celltype_annotation
"""

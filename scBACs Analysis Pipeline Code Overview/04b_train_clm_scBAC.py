"""
Train the neural cumulative-link model (CLM) component of scBACs.

Important stage settings from the manuscript:
- development: age <= 18 years, 2-year ordered age intervals
- adult: age > 18 years, 5-year ordered age intervals
- full: all ages, 5-year intervals; comparison model only

The script can be run section by section in an IDE.
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

import os
import sys
import argparse

import pandas as pd
import scanpy as sc
import torch

import clm_clock_v3 as clm_clock


# ============================================================================
# 2. PATHS AND ANALYSIS SETTINGS
# ============================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "retrained_scBACs",
    "full_genes_predictions_20260415"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

TRAIN_H5AD = os.path.join(
    DATA_DIR,
    "sce_train_CT.h5ad"
)

TRAIN_H5AD_OLD = os.path.join(
    DATA_DIR,
    "sce_train.h5ad"
)

NEW_2025_TRAINING_H5AD = os.path.join(
    DATA_DIR,
    "sce_train_added.h5ad"
)

CELLTYPES = [
    "Exc",
    "Inh",
    "Ast",
    "Mic",
    "Oli",
    "OPC"
]

STAGES_TO_TRAIN = [
    "development",
    "adult",
    "full"
]

AGE_CUT = 18
N_FOLDS = 5
RANDOM_SEED = 42

HIDDEN_DIMS = (
    128,
    64
)

DROPOUT = 0.5
BATCH_SIZE = 256
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# OPTIONAL COMMAND-LINE PATH OVERRIDE
# ============================================================================

if __name__ == "__main__" and len(sys.argv) > 1:
    parser = argparse.ArgumentParser(
        description="Train CLM scBAC models using original public model filenames."
    )

    parser.add_argument(
        "--training-h5ad",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        required=True
    )

    parser.add_argument(
        "--device",
        default=DEVICE
    )

    args = parser.parse_args()

    TRAIN_H5AD = args.training_h5ad
    MODEL_DIR = os.path.abspath(
        args.output_dir
    )
    DEVICE = args.device

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )


# ============================================================================
# 3. LOAD THE REVISED TRAINING RESOURCE
# ============================================================================

print("\n" + "=" * 70)
print("Loading scBAC training resource")
print("=" * 70)

if os.path.exists(TRAIN_H5AD):
    sce_train = sc.read_h5ad(
        TRAIN_H5AD
    )
else:
    sce_train_old = sc.read_h5ad(
        TRAIN_H5AD_OLD
    )

    sce_train_new = sc.read_h5ad(
        NEW_2025_TRAINING_H5AD
    )

    sce_train = sc.concat(
        [
            sce_train_old,
            sce_train_new
        ],
        axis=0,
        join="inner",
        merge="same",
        index_unique=None
    )

    if "status" in sce_train.obs.columns:
        sce_train = sce_train[
            sce_train.obs["status"] == "CT",
            :
        ].copy()

print(
    "Training data shape:",
    sce_train.shape
)

print(
    "Device:",
    DEVICE
)


# ============================================================================
# 4. TRAIN CLM MODELS
# ============================================================================

clm_training_summary = []

for stage in STAGES_TO_TRAIN:

    print("\n" + "#" * 80)
    print(
        "CLM stage: {} | age interval: {} years".format(
            stage,
            clm_clock.get_clm_bin_width(stage)
        )
    )
    print("#" * 80)

    for celltype in CELLTYPES:

        metadata = clm_clock.train_clm_clock(
            adata=sce_train,
            celltype=celltype,
            stage=stage,
            output_dir=MODEL_DIR,
            age_cut=AGE_CUT,
            n_folds=N_FOLDS,
            hidden_dims=HIDDEN_DIMS,
            dropout=DROPOUT,
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            learning_rate=LEARNING_RATE,
            random_seed=RANDOM_SEED,
            device=DEVICE
        )

        clm_training_summary.append(
            {
                "celltype": celltype,
                "stage": stage,
                "bin_width": metadata["bin_width"],
                "n_cells": metadata["n_cells"],
                "n_donors": metadata["n_donors"],
                "n_features": metadata["n_features"],
                "cell_spearman_r": metadata["oof_cell_metrics"]["spearman_r"],
                "cell_mae": metadata["oof_cell_metrics"]["mae"],
                "donor_spearman_r": metadata["oof_donor_metrics"]["spearman_r"],
                "donor_mae": metadata["oof_donor_metrics"]["mae"]
            }
        )


# ============================================================================
# 5. SAVE TRAINING SUMMARY
# ============================================================================

clm_training_summary = pd.DataFrame(
    clm_training_summary
)

clm_training_summary.to_csv(
    os.path.join(
        MODEL_DIR,
        "CLM_training_summary.csv"
    ),
    index=False
)

print("\nCLM training complete.")
print(
    clm_training_summary
)


# ============================================================================
# 6. COMMAND-LINE USAGE
# ============================================================================

"""
python 04b_train_clm_scBAC_v3.py \
    --training-h5ad /path/to/sce_train_CT.h5ad \
    --output-dir /path/to/models/retrained_scBACs/full_genes_predictions_20260415 \
    --device cuda
"""

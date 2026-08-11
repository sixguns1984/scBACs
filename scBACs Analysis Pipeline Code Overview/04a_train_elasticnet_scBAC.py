"""
Train the Elastic Net component of stage-specific scBACs.

The script is designed to be run section by section in an IDE. Edit the paths
and analysis settings below, then execute from top to bottom.

Primary clocks:
- development: age <= 18 years
- adult: age > 18 years

The optional full-lifespan model is retained for manuscript comparison analyses.
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

import os
import sys
import argparse

import pandas as pd
import scanpy as sc

import elasticnet_clock_v3 as elasticnet_clock


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

# Preferred public training file: revised 13-study CT training resource
TRAIN_H5AD = os.path.join(
    DATA_DIR,
    "sce_train_CT.h5ad"
)

# Fallback files for the original local working-data organization
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

# Full is included because it is used as a comparison model in the manuscript.
STAGES_TO_TRAIN = [
    "development",
    "adult",
    "full"
]

AGE_CUT = 18
N_FOLDS = 5
RANDOM_SEED = 42

ALPHA = 0.01
L1_RATIO_CANDIDATES = (
    0.10,
    0.50,
    0.70,
    0.90,
    0.95
)


# ============================================================================
# OPTIONAL COMMAND-LINE PATH OVERRIDE
# ============================================================================

if __name__ == "__main__" and len(sys.argv) > 1:
    parser = argparse.ArgumentParser(
        description="Train Elastic Net scBAC models."
    )

    parser.add_argument(
        "--training-h5ad",
        required=True,
        help="Revised 13-study CT training h5ad."
    )

    parser.add_argument(
        "--output-dir",
        required=True
    )

    args = parser.parse_args()

    TRAIN_H5AD = args.training_h5ad
    MODEL_DIR = os.path.abspath(
        args.output_dir
    )

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

    print(
        "sce_train_CT.h5ad was not found. Reconstructing the revised training "
        "resource from sce_train.h5ad + GSE291605_PFC.h5ad."
    )

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
    "Training studies:",
    sce_train.obs["dataset"].nunique()
)

print(
    "Training donors:",
    (
        sce_train.obs["dataset"].astype(str)
        + "::"
        + sce_train.obs["donor_id"].astype(str)
    ).nunique()
)


# ============================================================================
# 4. TRAIN DEVELOPMENTAL, ADULT, AND FULL-LIFESPAN ELASTIC NET MODELS
# ============================================================================

elasticnet_training_summary = []

for stage in STAGES_TO_TRAIN:

    print("\n" + "#" * 80)
    print("Elastic Net stage:", stage)
    print("#" * 80)

    for celltype in CELLTYPES:

        metadata = elasticnet_clock.train_elasticnet_clock(
            adata=sce_train,
            celltype=celltype,
            stage=stage,
            output_dir=MODEL_DIR,
            age_cut=AGE_CUT,
            alpha=ALPHA,
            l1_ratio_candidates=L1_RATIO_CANDIDATES,
            n_folds=N_FOLDS,
            max_iter=1000,
            random_seed=RANDOM_SEED,
            n_jobs=-1
        )

        elasticnet_training_summary.append(
            {
                "celltype": celltype,
                "stage": stage,
                "n_cells": metadata["n_cells"],
                "n_donors": metadata["n_donors"],
                "n_features": metadata["n_features"],
                "best_l1_ratio": metadata["best_l1_ratio"],
                "cell_spearman_r": metadata["oof_cell_metrics"]["spearman_r"],
                "cell_mae": metadata["oof_cell_metrics"]["mae"],
                "donor_spearman_r": metadata["oof_donor_metrics"]["spearman_r"],
                "donor_mae": metadata["oof_donor_metrics"]["mae"]
            }
        )


# ============================================================================
# 5. SAVE TRAINING SUMMARY
# ============================================================================

elasticnet_training_summary = pd.DataFrame(
    elasticnet_training_summary
)

elasticnet_training_summary.to_csv(
    os.path.join(
        MODEL_DIR,
        "ElasticNet_training_summary.csv"
    ),
    index=False
)

print("\nElastic Net training complete.")
print(
    elasticnet_training_summary
)


# ============================================================================
# 6. COMMAND-LINE USAGE
# ============================================================================

"""
python 04a_train_elasticnet_scBAC_v3.py \
    --training-h5ad /path/to/sce_train_CT.h5ad \
    --output-dir /path/to/models/retrained_scBACs/full_genes_predictions_20260415
"""

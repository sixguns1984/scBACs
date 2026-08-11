"""
Train the Transformer component of stage-specific scBACs.

Primary clocks:
- development: age <= 18 years
- adult: age > 18 years

The full-lifespan Transformer is trained only for comparison analyses.
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

import transformer_clock_v3 as transformer_clock


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

NUM_TOKENS = 20
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 1
DROPOUT_PROB = 0.5

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
MAX_EPOCHS = 100
LR_PATIENCE = 5
LR_FACTOR = 0.5
EARLY_STOPPING_PATIENCE = 10
GRADIENT_CLIP_NORM = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# OPTIONAL COMMAND-LINE PATH OVERRIDE
# ============================================================================

if __name__ == "__main__" and len(sys.argv) > 1:
    parser = argparse.ArgumentParser(
        description="Train Transformer scBAC models using original public model filenames."
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
# 4. TRAIN TRANSFORMER MODELS
# ============================================================================

transformer_training_summary = []

for stage in STAGES_TO_TRAIN:

    print("\n" + "#" * 80)
    print("Transformer stage:", stage)
    print("#" * 80)

    for celltype in CELLTYPES:

        metadata = transformer_clock.train_transformer_clock(
            adata=sce_train,
            celltype=celltype,
            stage=stage,
            output_dir=MODEL_DIR,
            age_cut=AGE_CUT,
            n_folds=N_FOLDS,
            num_tokens=NUM_TOKENS,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            dropout_prob=DROPOUT_PROB,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            lr_patience=LR_PATIENCE,
            lr_factor=LR_FACTOR,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            gradient_clip_norm=GRADIENT_CLIP_NORM,
            random_seed=RANDOM_SEED,
            device=DEVICE
        )

        transformer_training_summary.append(
            {
                "celltype": celltype,
                "stage": stage,
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

transformer_training_summary = pd.DataFrame(
    transformer_training_summary
)

transformer_training_summary.to_csv(
    os.path.join(
        MODEL_DIR,
        "Transformer_training_summary.csv"
    ),
    index=False
)

print("\nTransformer training complete.")
print(
    transformer_training_summary
)


# ============================================================================
# 6. COMMAND-LINE USAGE
# ============================================================================

"""
python 04c_train_transformer_scBAC_v3.py \
    --training-h5ad /path/to/sce_train_CT.h5ad \
    --output-dir /path/to/models/retrained_scBACs/full_genes_predictions_20260415 \
    --device cuda
"""

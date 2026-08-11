"""
Leave-one-study-out (LOSO) validation of the three adult scBAC components.

In each iteration:
1. one complete study from the 13-study training resource is withheld;
2. Elastic Net, CLM, and Transformer models are trained on the remaining studies;
3. each component uses donor-grouped five-fold training internally;
4. the five fold-specific models are averaged to predict the withheld study;
5. performance is evaluated at both cell and donor levels.

The primary LOSO analysis uses adult cells only (age > 18 years). MAE is the
primary cross-study metric because many individual studies contain few donors
and restricted chronological-age ranges.

No ComBat correction is used in this primary pipeline. The ComBat analysis in
the response letter was an exploratory sensitivity analysis only.
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

import os
import sys
import argparse

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error

import elasticnet_clock
import clm_clock
import transformer_clock


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

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "LOSO_validation"
)

MODEL_DIR = os.path.join(
    OUTPUT_DIR,
    "models"
)

PREDICTION_DIR = os.path.join(
    OUTPUT_DIR,
    "predictions"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    PREDICTION_DIR,
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

ALGORITHMS = [
    "ElasticNet",
    "CLM",
    "Transformer"
]

AGE_CUT = 18
N_FOLDS = 5
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optional convenience setting for testing only a subset while developing code.
# Leave as None for the complete 13-study LOSO analysis.
STUDIES_TO_RUN = None


# ============================================================================
# OPTIONAL COMMAND-LINE PATH OVERRIDE
# ============================================================================

if __name__ == "__main__" and len(sys.argv) > 1:
    parser = argparse.ArgumentParser(
        description="Run adult-stage LOSO validation for all scBAC components."
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
    OUTPUT_DIR = os.path.abspath(
        args.output_dir
    )
    DEVICE = args.device

    MODEL_DIR = os.path.join(
        OUTPUT_DIR,
        "models"
    )

    PREDICTION_DIR = os.path.join(
        OUTPUT_DIR,
        "predictions"
    )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        PREDICTION_DIR,
        exist_ok=True
    )


# ============================================================================
# 3. SIMPLE PERFORMANCE FUNCTIONS
# ============================================================================


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "spearman_r": np.nan,
            "pearson_r": np.nan
        }

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    if (
        len(y_true) >= 2
        and np.std(y_true) > 0
        and np.std(y_pred) > 0
    ):
        spearman_r = spearmanr(
            y_true,
            y_pred
        )[0]

        pearson_r = pearsonr(
            y_true,
            y_pred
        )[0]
    else:
        spearman_r = np.nan
        pearson_r = np.nan

    return {
        "n": int(len(y_true)),
        "mae": float(mae),
        "spearman_r": float(spearman_r),
        "pearson_r": float(pearson_r)
    }


def donor_median_table(cell_df):
    """Aggregate withheld-study cell predictions using donor median."""

    return (
        cell_df.groupby(
            [
                "dataset",
                "donor_id"
            ],
            observed=True
        )
        .agg(
            true_age=("true_age", "first"),
            predicted_age=("predicted_age", "median"),
            n_cells=("predicted_age", "size")
        )
        .reset_index()
    )


# ============================================================================
# 4. LOAD THE REVISED 13-STUDY TRAINING RESOURCE
# ============================================================================

print("\n" + "=" * 70)
print("Loading the revised training resource for LOSO")
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

# Primary LOSO analysis is adult-stage only: age > 18
sce_train = sce_train[
    pd.to_numeric(
        sce_train.obs["Age_at_death"],
        errors="coerce"
    ) > AGE_CUT,
    :
].copy()

all_studies = sorted(
    sce_train.obs["dataset"].astype(str).unique()
)

if STUDIES_TO_RUN is not None:
    all_studies = [
        study
        for study in all_studies
        if study in STUDIES_TO_RUN
    ]

print(
    "Number of adult training studies:",
    len(all_studies)
)

print(
    "Studies:",
    all_studies
)


# ============================================================================
# 5. RUN LOSO FOR EACH CELL TYPE AND EACH COMPONENT MODEL
# ============================================================================

all_cell_predictions = []
all_donor_predictions = []
all_metrics = []

for celltype in CELLTYPES:

    print("\n" + "#" * 90)
    print("LOSO cell type:", celltype)
    print("#" * 90)

    sce_celltype = sce_train[
        sce_train.obs["celltype"].astype(str) == celltype,
        :
    ].copy()

    for left_out_study in all_studies:

        print("\n" + "-" * 90)
        print("Held-out study:", left_out_study)
        print("-" * 90)

        train_mask = (
            sce_celltype.obs["dataset"].astype(str)
            != left_out_study
        )

        test_mask = (
            sce_celltype.obs["dataset"].astype(str)
            == left_out_study
        )

        sce_loso_train = sce_celltype[
            train_mask,
            :
        ].copy()

        sce_loso_test = sce_celltype[
            test_mask,
            :
        ].copy()

        if sce_loso_test.n_obs == 0:
            continue

        print(
            "Training cells:",
            sce_loso_train.n_obs,
            "| Test cells:",
            sce_loso_test.n_obs
        )

        # --------------------------------------------------------------------
        # 5.1 Elastic Net
        # --------------------------------------------------------------------

        elastic_root = os.path.join(
            MODEL_DIR,
            "ElasticNet",
            "left_out_{}".format(left_out_study)
        )

        elasticnet_clock.train_elasticnet_clock(
            adata=sce_loso_train,
            celltype=celltype,
            stage="adult",
            output_dir=elastic_root,
            age_cut=AGE_CUT,
            n_folds=N_FOLDS,
            random_seed=RANDOM_SEED
        )

        elastic_prediction = elasticnet_clock.predict_elasticnet_ensemble(
            sce_loso_test,
            os.path.join(
                elastic_root,
                "adult",
                celltype
            ),
            normalize=True
        )

        # --------------------------------------------------------------------
        # 5.2 CLM -- adult bin width is automatically fixed at 5 years
        # --------------------------------------------------------------------

        clm_root = os.path.join(
            MODEL_DIR,
            "CLM",
            "left_out_{}".format(left_out_study)
        )

        clm_clock.train_clm_clock(
            adata=sce_loso_train,
            celltype=celltype,
            stage="adult",
            output_dir=clm_root,
            age_cut=AGE_CUT,
            n_folds=N_FOLDS,
            random_seed=RANDOM_SEED,
            device=DEVICE
        )

        clm_prediction = clm_clock.predict_clm_ensemble(
            sce_loso_test,
            os.path.join(
                clm_root,
                "adult",
                celltype
            ),
            normalize=True,
            device=DEVICE
        )

        # --------------------------------------------------------------------
        # 5.3 Transformer
        # --------------------------------------------------------------------

        transformer_root = os.path.join(
            MODEL_DIR,
            "Transformer",
            "left_out_{}".format(left_out_study)
        )

        transformer_clock.train_transformer_clock(
            adata=sce_loso_train,
            celltype=celltype,
            stage="adult",
            output_dir=transformer_root,
            age_cut=AGE_CUT,
            n_folds=N_FOLDS,
            random_seed=RANDOM_SEED,
            device=DEVICE
        )

        transformer_prediction = transformer_clock.predict_transformer_ensemble(
            sce_loso_test,
            os.path.join(
                transformer_root,
                "adult",
                celltype
            ),
            normalize=True,
            device=DEVICE
        )

        # --------------------------------------------------------------------
        # 5.4 Store cell-level predictions
        # --------------------------------------------------------------------

        prediction_matrix = {
            "ElasticNet": elastic_prediction,
            "CLM": clm_prediction,
            "Transformer": transformer_prediction,
            "Ensemble": np.mean(
                np.column_stack(
                    [
                        elastic_prediction,
                        clm_prediction,
                        transformer_prediction
                    ]
                ),
                axis=1
            )
        }

        for algorithm, prediction in prediction_matrix.items():

            cell_df = pd.DataFrame(
                {
                    "cell_id": sce_loso_test.obs_names.astype(str),
                    "dataset": left_out_study,
                    "donor_id": sce_loso_test.obs["donor_id"].astype(str).values,
                    "celltype": celltype,
                    "algorithm": algorithm,
                    "true_age": pd.to_numeric(
                        sce_loso_test.obs["Age_at_death"],
                        errors="coerce"
                    ).values,
                    "predicted_age": prediction
                }
            )

            all_cell_predictions.append(
                cell_df
            )

            donor_df = donor_median_table(
                cell_df
            )

            donor_df["celltype"] = celltype
            donor_df["algorithm"] = algorithm

            all_donor_predictions.append(
                donor_df
            )

            cell_metrics = calculate_metrics(
                cell_df["true_age"],
                cell_df["predicted_age"]
            )

            donor_metrics = calculate_metrics(
                donor_df["true_age"],
                donor_df["predicted_age"]
            )

            all_metrics.append(
                {
                    "dataset": left_out_study,
                    "celltype": celltype,
                    "algorithm": algorithm,
                    "level": "Cell",
                    "n": cell_metrics["n"],
                    "age_min": cell_df["true_age"].min(),
                    "age_max": cell_df["true_age"].max(),
                    "spearman_r": cell_metrics["spearman_r"],
                    "pearson_r": cell_metrics["pearson_r"],
                    "mae": cell_metrics["mae"]
                }
            )

            all_metrics.append(
                {
                    "dataset": left_out_study,
                    "celltype": celltype,
                    "algorithm": algorithm,
                    "level": "Donor",
                    "n": donor_metrics["n"],
                    "age_min": donor_df["true_age"].min(),
                    "age_max": donor_df["true_age"].max(),
                    "spearman_r": donor_metrics["spearman_r"],
                    "pearson_r": donor_metrics["pearson_r"],
                    "mae": donor_metrics["mae"]
                }
            )


# ============================================================================
# 6. SAVE COMPLETE LOSO PREDICTIONS
# ============================================================================

loso_cell_predictions = pd.concat(
    all_cell_predictions,
    axis=0,
    ignore_index=True
)

loso_donor_predictions = pd.concat(
    all_donor_predictions,
    axis=0,
    ignore_index=True
)

loso_metrics = pd.DataFrame(
    all_metrics
)

loso_cell_predictions.to_csv(
    os.path.join(
        PREDICTION_DIR,
        "LOSO_cell_level_predictions.csv"
    ),
    index=False
)

loso_donor_predictions.to_csv(
    os.path.join(
        PREDICTION_DIR,
        "LOSO_donor_median_predictions.csv"
    ),
    index=False
)

loso_metrics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "LOSO_per_study_performance.csv"
    ),
    index=False
)


# ============================================================================
# 7. SUMMARIZE LOSO MAE ACROSS STUDIES
# ============================================================================

# The manuscript uses MAE as the primary LOSO metric.
loso_summary = (
    loso_metrics.groupby(
        [
            "celltype",
            "algorithm",
            "level"
        ],
        observed=True
    )
    .agg(
        mean_mae=("mae", "mean"),
        sem_mae=("mae", "sem"),
        median_mae=("mae", "median"),
        mean_spearman=("spearman_r", "mean"),
        n_studies=("dataset", "nunique")
    )
    .reset_index()
)

loso_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "LOSO_mean_performance_across_studies.csv"
    ),
    index=False
)

print("\n" + "=" * 90)
print("LOSO summary")
print("=" * 90)
print(
    loso_summary
)


# ============================================================================
# 8. COMMAND-LINE USAGE
# ============================================================================

"""
python 04d_LOSO_scBAC_validation.py \
    --training-h5ad /path/to/sce_train_CT.h5ad \
    --output-dir /path/to/results/LOSO_validation \
    --device cuda
"""

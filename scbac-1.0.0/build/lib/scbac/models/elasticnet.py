"""
Elastic Net functions for the scBAC pipeline.

This file contains only the model functions used by the public analysis scripts.
It is intentionally kept simple and is compatible with Python 3.9.20.

Stage definitions
-----------------
development : Age_at_death <= 18
adult       : Age_at_death > 18
full        : all ages, used only for the full-lifespan comparison model

Model settings
--------------
- alpha = 0.01
- l1_ratio candidates = 0.10, 0.50, 0.70, 0.90, 0.95
- donor-grouped five-fold cross-validation
- maximum 1,000 iterations
- random coordinate selection
- independent predictions are averaged across the five fold-specific models
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import scanpy as sc

from scipy import sparse
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================================
# 2. BASIC FUNCTIONS
# ============================================================================


def normalize_adata(adata, copy=True):
    """Library-size normalize and log1p transform an AnnData object."""

    adata_use = adata.copy() if copy else adata
    adata_use.X = adata_use.X.astype("float32")
    sc.pp.normalize_per_cell(adata_use)
    sc.pp.log1p(adata_use)
    return adata_use


def get_stage_mask(ages, stage, age_cut=18):
    """Return the cell mask for development, adult, or full-lifespan models."""

    ages = np.asarray(ages, dtype=float)

    if stage == "development":
        return ages <= age_cut
    elif stage == "adult":
        return ages > age_cut
    elif stage == "full":
        return np.ones(len(ages), dtype=bool)
    else:
        raise ValueError("stage must be 'development', 'adult', or 'full'")


def make_unique_donor_id(obs, donor_col="donor_id", dataset_col="dataset"):
    """Create study-specific donor IDs when a dataset column is available."""

    if dataset_col in obs.columns:
        return (
            obs[dataset_col].astype(str)
            + "::"
            + obs[donor_col].astype(str)
        ).values

    return obs[donor_col].astype(str).values


def prepare_feature_matrix(adata, feature_genes):
    """
    Reindex expression to the exact training feature order.

    Genes absent from the new dataset are represented as zero.
    The input AnnData object should already be normalized/log-transformed.
    """

    feature_genes = [str(gene) for gene in feature_genes]
    var_index = pd.Index(adata.var_names.astype(str))
    positions = var_index.get_indexer(feature_genes)

    present_model_pos = np.where(positions >= 0)[0]
    present_data_pos = positions[present_model_pos]

    X = np.zeros(
        (adata.n_obs, len(feature_genes)),
        dtype=np.float32
    )

    if len(present_data_pos) > 0:
        block = adata[:, present_data_pos].X

        if sparse.issparse(block):
            block = block.toarray()

        X[:, present_model_pos] = np.asarray(
            block,
            dtype=np.float32
        )

    n_missing = int(np.sum(positions < 0))
    return X, n_missing


def regression_metrics(y_true, y_pred):
    """Calculate Spearman correlation, Pearson correlation, MAE, and R2."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return {
            "n": 0,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "mae": np.nan,
            "r2": np.nan
        }

    if (
        len(y_true) >= 2
        and np.unique(y_true).size > 1
        and np.unique(y_pred).size > 1
    ):
        spearman_r, spearman_p = spearmanr(y_true, y_pred)
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
    else:
        spearman_r = np.nan
        spearman_p = np.nan
        pearson_r = np.nan
        pearson_p = np.nan

    return {
        "n": int(len(y_true)),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan
    }


def donor_median_predictions(obs, predictions, donor_col="donor_id", age_col="Age_at_death", dataset_col="dataset"):
    """Summarize cell-level predictions by donor median."""

    df = obs.copy()
    df["predicted_age"] = np.asarray(predictions, dtype=float)

    if dataset_col in df.columns:
        group_cols = [dataset_col, donor_col]
    else:
        group_cols = [donor_col]

    donor_df = (
        df.groupby(group_cols, observed=True)
        .agg(
            predicted_age=("predicted_age", "median"),
            chronological_age=(age_col, "first"),
            n_cells=("predicted_age", "size")
        )
        .reset_index()
    )

    return donor_df


# ============================================================================
# 3. MODEL TRAINING
# ============================================================================


def train_elasticnet_clock(
    adata,
    celltype,
    stage,
    output_dir,
    age_cut=18,
    celltype_col="celltype",
    donor_col="donor_id",
    dataset_col="dataset",
    age_col="Age_at_death",
    status_col="status",
    control_label="CT",
    alpha=0.01,
    l1_ratio_candidates=(0.10, 0.50, 0.70, 0.90, 0.95),
    n_folds=5,
    max_iter=1000,
    random_seed=42,
    n_jobs=-1,
    feature_genes=None
):
    """Train one cell-type- and stage-specific five-fold Elastic Net ensemble."""

    print("\n" + "=" * 70)
    print("Elastic Net | {} | {}".format(celltype, stage))
    print("=" * 70)

    # Keep only control cells of the requested lineage
    adata_ct = adata[
        (adata.obs[status_col].astype(str) == control_label)
        & (adata.obs[celltype_col].astype(str) == celltype),
        :
    ].copy()

    ages = pd.to_numeric(
        adata_ct.obs[age_col],
        errors="coerce"
    ).values

    stage_mask = get_stage_mask(
        ages,
        stage,
        age_cut=age_cut
    ) & np.isfinite(ages)

    adata_ct = adata_ct[stage_mask, :].copy()

    if adata_ct.n_obs == 0:
        raise ValueError("No cells available for {} / {}".format(celltype, stage))

    # Independent normalization of the training resource
    adata_ct = normalize_adata(
        adata_ct,
        copy=False
    )

    # No age-correlation feature preselection.
    # If a feature list is supplied, preserve its exact order.
    if feature_genes is None:
        feature_genes = adata_ct.var_names.astype(str).tolist()
    else:
        feature_genes = [str(gene) for gene in feature_genes]

    X, n_missing = prepare_feature_matrix(
        adata_ct,
        feature_genes
    )

    if n_missing > 0:
        print("Training feature list contains {} genes absent from the input matrix.".format(n_missing))

    y = pd.to_numeric(
        adata_ct.obs[age_col],
        errors="raise"
    ).values.astype(float)

    groups = make_unique_donor_id(
        adata_ct.obs,
        donor_col=donor_col,
        dataset_col=dataset_col
    )

    if len(np.unique(groups)) < n_folds:
        raise ValueError("Not enough donors for {}-fold CV".format(n_folds))

    print("Cells:", adata_ct.n_obs)
    print("Donors:", len(np.unique(groups)))
    print("Features:", len(feature_genes))
    print("Age range: {:.1f} - {:.1f}".format(y.min(), y.max()))

    stage_label = {
        "development": "Development",
        "adult": "Adult",
        "full": "Full"
    }[stage]
    age_cut_filename = 0 if stage == "full" else int(age_cut)
    public_stage_dir = os.path.join(
        output_dir,
        "ElasticNet_{}".format(stage_label)
    )
    model_dir = os.path.join(
        public_stage_dir,
        "_training_work",
        celltype
    )
    feature_dir = os.path.join(
        public_stage_dir,
        "model_features"
    )
    validation_dir = os.path.join(
        public_stage_dir,
        "validation_results"
    )
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    group_kfold = GroupKFold(
        n_splits=n_folds
    )

    cv_splits = list(
        group_kfold.split(
            X,
            y,
            groups
        )
    )

    # Select the L1 mixing ratio using donor-grouped CV.
    # The regularization strength alpha remains fixed at 0.01.
    selector = ElasticNetCV(
        l1_ratio=list(l1_ratio_candidates),
        alphas=[alpha],
        cv=cv_splits,
        n_jobs=n_jobs,
        max_iter=max_iter,
        random_state=random_seed,
        selection="random",
        verbose=0
    )

    selector.fit(
        X,
        y
    )

    best_l1_ratio = float(
        selector.l1_ratio_
    )

    print("Selected l1_ratio:", best_l1_ratio)

    fold_models = []
    fold_results = []

    oof_prediction = np.full(
        adata_ct.n_obs,
        np.nan
    )

    oof_fold = np.full(
        adata_ct.n_obs,
        -1,
        dtype=int
    )

    for fold_idx, (train_idx, val_idx) in enumerate(
        cv_splits,
        start=1
    ):

        print("\nFold {}/{}".format(fold_idx, n_folds))

        model = ElasticNet(
            alpha=alpha,
            l1_ratio=best_l1_ratio,
            max_iter=max_iter,
            random_state=random_seed + fold_idx - 1,
            selection="random"
        )

        model.fit(
            X[train_idx],
            y[train_idx]
        )

        val_pred = model.predict(
            X[val_idx]
        )

        oof_prediction[val_idx] = val_pred
        oof_fold[val_idx] = fold_idx

        cell_metrics = regression_metrics(
            y[val_idx],
            val_pred
        )

        donor_df = donor_median_predictions(
            adata_ct.obs.iloc[val_idx],
            val_pred,
            donor_col=donor_col,
            age_col=age_col,
            dataset_col=dataset_col
        )

        donor_metrics = regression_metrics(
            donor_df["chronological_age"],
            donor_df["predicted_age"]
        )

        fold_results.append(
            {
                "fold": fold_idx,
                "n_train_cells": len(train_idx),
                "n_validation_cells": len(val_idx),
                "n_train_donors": len(np.unique(groups[train_idx])),
                "n_validation_donors": len(np.unique(groups[val_idx])),
                "cell_spearman_r": cell_metrics["spearman_r"],
                "cell_mae": cell_metrics["mae"],
                "donor_spearman_r": donor_metrics["spearman_r"],
                "donor_mae": donor_metrics["mae"],
                "n_nonzero_coefficients": int(np.count_nonzero(model.coef_))
            }
        )

        model_file = os.path.join(
            model_dir,
            "fold_{}.joblib".format(fold_idx)
        )

        joblib.dump(
            model,
            model_file
        )

        fold_models.append(model)

        print(
            "Cell MAE={:.2f}, donor MAE={:.2f}".format(
                cell_metrics["mae"],
                donor_metrics["mae"]
            )
        )

    # Save the public five-fold ensemble using the original scBAC filename.
    public_model_file = os.path.join(
        public_stage_dir,
        "{}_elasticnet_5fold_ensemble_age_cut{}_n_features12779.pkl".format(
            celltype,
            age_cut_filename
        )
    )
    joblib.dump(
        fold_models,
        public_model_file
    )
    public_feature_file = os.path.join(
        feature_dir,
        "{}_elasticnet_features_age_cut{}_n_features12779.csv".format(
            celltype,
            age_cut_filename
        )
    )
    pd.DataFrame(
        {
            "genename": feature_genes,
            "celltype": celltype
        }
    ).to_csv(
        public_feature_file,
        index=False
    )
    pd.DataFrame(
        {
            "fold": list(range(1, n_folds + 1)),
            "l1_ratio": [best_l1_ratio] * n_folds
        }
    ).to_csv(
        os.path.join(
            public_stage_dir,
            "{}_elasticnet_l1_ratio_age_cut{}_n_features12779.csv".format(
                celltype,
                age_cut_filename
            )
        ),
        index=False
    )
    pd.DataFrame(
        fold_results
    ).to_csv(
        os.path.join(
            validation_dir,
            "{}_elasticnet_5fold_validation_summary_age_cut{}_n_features12779.csv".format(
                celltype,
                age_cut_filename
            )
        ),
        index=False
    )

    # Auxiliary work files are retained for diagnostics and local reloads.
    pd.DataFrame(
        {
            "gene": feature_genes
        }
    ).to_csv(
        os.path.join(
            model_dir,
            "features.csv"
        ),
        index=False
    )

    # Save fold metrics
    pd.DataFrame(
        fold_results
    ).to_csv(
        os.path.join(
            model_dir,
            "fold_validation_metrics.csv"
        ),
        index=False
    )

    # Save OOF predictions
    oof_cell = adata_ct.obs.copy()
    oof_cell["true_age"] = y
    oof_cell["elasticnet_oof_prediction"] = oof_prediction
    oof_cell["cv_fold"] = oof_fold
    oof_cell.index.name = "cell_id"

    oof_cell.to_csv(
        os.path.join(
            model_dir,
            "oof_cell_predictions.csv"
        )
    )

    oof_donor = donor_median_predictions(
        adata_ct.obs,
        oof_prediction,
        donor_col=donor_col,
        age_col=age_col,
        dataset_col=dataset_col
    )

    oof_donor.to_csv(
        os.path.join(
            model_dir,
            "oof_donor_predictions.csv"
        ),
        index=False
    )

    oof_cell_metrics = regression_metrics(
        y,
        oof_prediction
    )

    oof_donor_metrics = regression_metrics(
        oof_donor["chronological_age"],
        oof_donor["predicted_age"]
    )

    metadata = {
        "model_type": "elasticnet",
        "celltype": celltype,
        "stage": stage,
        "age_cut": float(age_cut),
        "alpha": float(alpha),
        "best_l1_ratio": float(best_l1_ratio),
        "l1_ratio_candidates": list(l1_ratio_candidates),
        "n_folds": int(n_folds),
        "max_iter": int(max_iter),
        "random_seed": int(random_seed),
        "n_cells": int(adata_ct.n_obs),
        "n_donors": int(len(np.unique(groups))),
        "n_features": int(len(feature_genes)),
        "public_age_cut_filename": int(age_cut_filename),
        "public_model_file": public_model_file,
        "public_feature_file": public_feature_file,
        "normalization": "scanpy.pp.normalize_per_cell + scanpy.pp.log1p",
        "oof_cell_metrics": oof_cell_metrics,
        "oof_donor_metrics": oof_donor_metrics
    }

    with open(
        os.path.join(
            model_dir,
            "metadata.json"
        ),
        "w"
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2
        )

    # Convenience file containing the five fitted models
    joblib.dump(
        fold_models,
        os.path.join(
            model_dir,
            "ensemble_models.joblib"
        )
    )

    print("\nOOF cell MAE: {:.2f}".format(oof_cell_metrics["mae"]))
    print("OOF donor MAE: {:.2f}".format(oof_donor_metrics["mae"]))
    print("Public model saved to:", public_model_file)
    print("Auxiliary training files saved to:", model_dir)

    return metadata


# ============================================================================
# 4. LOAD AND PREDICT
# ============================================================================


def load_elasticnet_ensemble(model_dir):
    """Load the five fold-specific Elastic Net models and feature list."""

    with open(
        os.path.join(
            model_dir,
            "metadata.json"
        ),
        "r"
    ) as handle:
        metadata = json.load(handle)

    feature_genes = pd.read_csv(
        os.path.join(
            model_dir,
            "features.csv"
        )
    )["gene"].astype(str).tolist()

    models = []

    for fold_idx in range(
        1,
        int(metadata["n_folds"]) + 1
    ):
        models.append(
            joblib.load(
                os.path.join(
                    model_dir,
                    "fold_{}.joblib".format(fold_idx)
                )
            )
        )

    return models, feature_genes, metadata


def predict_elasticnet_ensemble(
    adata,
    model_dir,
    normalize=True,
    chunk_size=10000,
    return_fold_predictions=False
):
    """Predict ages using the mean of the five Elastic Net fold models."""

    models, feature_genes, metadata = load_elasticnet_ensemble(
        model_dir
    )

    if normalize:
        adata_use = normalize_adata(
            adata,
            copy=True
        )
    else:
        adata_use = adata.copy()

    mean_predictions = []
    fold_predictions_all = []

    for start in range(
        0,
        adata_use.n_obs,
        chunk_size
    ):
        stop = min(
            start + chunk_size,
            adata_use.n_obs
        )

        X, n_missing = prepare_feature_matrix(
            adata_use[start:stop],
            feature_genes
        )

        fold_predictions = np.column_stack(
            [
                model.predict(X)
                for model in models
            ]
        )

        mean_predictions.append(
            fold_predictions.mean(axis=1)
        )

        if return_fold_predictions:
            fold_predictions_all.append(
                fold_predictions
            )

    mean_predictions = np.concatenate(
        mean_predictions
    )

    if return_fold_predictions:
        return (
            mean_predictions,
            np.vstack(fold_predictions_all)
        )

    return mean_predictions

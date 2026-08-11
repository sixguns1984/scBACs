"""
Neural Cumulative Link Model (CLM) functions for the scBAC pipeline.

Python version: 3.9.20

Stage definitions
-----------------
development : Age_at_death <= 18, 2-year ordered age intervals
adult       : Age_at_death > 18, 5-year ordered age intervals
full        : all ages, 5-year intervals; comparison model only

Architecture
------------
- hidden layers: 128 and 64
- Group Normalization + ReLU + dropout 0.5
- ordered cumulative-link thresholds
- continuous age = probability-weighted mean of age-bin centers
- loss = 0.70 * ordinal NLL + 0.30 * MAE + 0.15 * rank loss
- Adam, learning rate 1e-3
- batch size 256
- maximum 100 epochs

All outer cross-validation folds are donor-disjoint. For epoch selection, an
inner donor-level validation split is created from the outer-training donors.
The final fold model is then refit on all outer-training cells for the selected
number of epochs before evaluating the untouched outer validation fold.
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

import os
import json
import copy
import joblib
import random

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim

from scipy import sparse
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# 2. BASIC SETTINGS AND FUNCTIONS
# ============================================================================


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device=None):
    if device is None:
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device_use = torch.device(device)

    if device_use.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")

    return device_use


def normalize_adata(adata, copy=True):
    adata_use = adata.copy() if copy else adata
    adata_use.X = adata_use.X.astype("float32")
    sc.pp.normalize_per_cell(adata_use)
    sc.pp.log1p(adata_use)
    return adata_use


def get_stage_mask(ages, stage, age_cut=18):
    ages = np.asarray(ages, dtype=float)

    if stage == "development":
        return ages <= age_cut
    elif stage == "adult":
        return ages > age_cut
    elif stage == "full":
        return np.ones(len(ages), dtype=bool)
    else:
        raise ValueError("stage must be 'development', 'adult', or 'full'")


def get_clm_bin_width(stage):
    """Return the manuscript CLM interval for each clock stage."""

    if stage == "development":
        return 2.0
    elif stage == "adult":
        return 5.0
    elif stage == "full":
        return 5.0
    else:
        raise ValueError("stage must be 'development', 'adult', or 'full'")


def make_unique_donor_id(obs, donor_col="donor_id", dataset_col="dataset"):
    if dataset_col in obs.columns:
        return (
            obs[dataset_col].astype(str)
            + "::"
            + obs[donor_col].astype(str)
        ).values

    return obs[donor_col].astype(str).values


def prepare_feature_matrix(adata, feature_genes):
    """Align expression to the exact training feature order and zero-fill missing genes."""

    feature_genes = [str(gene) for gene in feature_genes]
    positions = pd.Index(
        adata.var_names.astype(str)
    ).get_indexer(
        feature_genes
    )

    present_model_pos = np.where(
        positions >= 0
    )[0]

    present_data_pos = positions[
        present_model_pos
    ]

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

    return X, int(np.sum(positions < 0))


def regression_metrics(y_true, y_pred):
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
    df = obs.copy()
    df["predicted_age"] = np.asarray(predictions, dtype=float)

    if dataset_col in df.columns:
        group_cols = [dataset_col, donor_col]
    else:
        group_cols = [donor_col]

    return (
        df.groupby(group_cols, observed=True)
        .agg(
            predicted_age=("predicted_age", "median"),
            chronological_age=(age_col, "first"),
            n_cells=("predicted_age", "size")
        )
        .reset_index()
    )


# ============================================================================
# 3. AGE BINS
# ============================================================================


def age_to_bins(ages, bin_width):
    """
    Convert chronological ages to occupied ordered age bins.

    Empty age intervals are removed and class labels are remapped to 0..K-1.
    """

    ages = np.asarray(ages, dtype=float)

    if len(ages) == 0:
        raise ValueError("No ages supplied")

    if np.any(~np.isfinite(ages)):
        raise ValueError("Ages contain non-finite values")

    max_edge = (
        np.ceil(
            np.max(ages) / bin_width
        )
        * bin_width
        + bin_width
    )

    edges = np.arange(
        0.0,
        max_edge + 1e-8,
        bin_width
    )

    original_class = np.digitize(
        ages,
        edges,
        right=False
    ) - 1

    original_class = np.clip(
        original_class,
        0,
        len(edges) - 2
    )

    all_centers = (
        edges[:-1]
        + edges[1:]
    ) / 2.0

    occupied = np.unique(
        original_class
    )

    class_map = {
        old_class: new_class
        for new_class, old_class in enumerate(occupied)
    }

    remapped_class = np.array(
        [
            class_map[value]
            for value in original_class
        ],
        dtype=int
    )

    bin_centers = all_centers[
        occupied
    ]

    retained_edges = np.array(
        [
            edges[index]
            for index in occupied
        ]
        + [
            edges[occupied[-1] + 1]
        ],
        dtype=float
    )

    return remapped_class, retained_edges, bin_centers


# ============================================================================
# 4. NEURAL CUMULATIVE LINK MODEL
# ============================================================================


class NeuralCumulativeLink(nn.Module):
    """Neural cumulative-link ordinal regression model."""

    def __init__(
        self,
        input_dim,
        n_classes,
        hidden_dims=(128, 64),
        dropout=0.5
    ):
        super(NeuralCumulativeLink, self).__init__()

        if n_classes < 2:
            raise ValueError("CLM requires at least two occupied age bins")

        self.n_classes = int(n_classes)
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = float(dropout)

        layers = []
        previous_dim = input_dim

        for hidden_dim in self.hidden_dims:
            n_groups = min(
                32,
                max(1, hidden_dim // 8)
            )

            while hidden_dim % n_groups != 0 and n_groups > 1:
                n_groups -= 1

            layers.extend(
                [
                    nn.Linear(
                        previous_dim,
                        hidden_dim
                    ),
                    nn.GroupNorm(
                        num_groups=n_groups,
                        num_channels=hidden_dim
                    ),
                    nn.ReLU(),
                    nn.Dropout(
                        self.dropout
                    )
                ]
            )

            previous_dim = hidden_dim

        layers.append(
            nn.Linear(
                previous_dim,
                1
            )
        )

        self.feature_extractor = nn.Sequential(
            *layers
        )

        self.threshold_raw = nn.Parameter(
            torch.randn(
                self.n_classes - 1
            )
        )

    def get_thresholds(self):
        return torch.sort(
            self.threshold_raw
        )[0]

    def forward(self, x):
        eta = self.feature_extractor(
            x
        ).squeeze(-1)

        thresholds = self.get_thresholds()

        cumulative = torch.sigmoid(
            thresholds.unsqueeze(0)
            - eta.unsqueeze(1)
        )

        probs = torch.empty(
            (
                x.shape[0],
                self.n_classes
            ),
            dtype=x.dtype,
            device=x.device
        )

        probs[:, 0] = cumulative[:, 0]

        if self.n_classes > 2:
            probs[:, 1:-1] = (
                cumulative[:, 1:]
                - cumulative[:, :-1]
            )

        probs[:, -1] = 1.0 - cumulative[:, -1]

        return probs, eta, thresholds

    def predict_age(self, x, bin_centers):
        probs, _, _ = self.forward(
            x
        )

        centers = torch.as_tensor(
            bin_centers,
            dtype=probs.dtype,
            device=x.device
        )

        return (
            probs
            * centers.unsqueeze(0)
        ).sum(
            dim=1
        )


def clm_loss(
    probs,
    targets,
    bin_centers,
    nll_weight=0.70,
    mae_weight=0.30,
    rank_weight=0.15
):
    """Composite ordinal NLL + MAE + rank-consistency loss."""

    row_index = torch.arange(
        probs.shape[0],
        device=probs.device
    )

    ordinal_nll = -torch.log(
        probs[
            row_index,
            targets
        ].clamp_min(1e-8)
    ).mean()

    expected_age = (
        probs
        * bin_centers.unsqueeze(0)
    ).sum(
        dim=1
    )

    observed_bin_age = bin_centers[
        targets
    ]

    mae = torch.abs(
        expected_age
        - observed_bin_age
    ).mean()

    if probs.shape[0] > 1:
        pred_diff = (
            expected_age[:, None]
            - expected_age[None, :]
        )

        true_diff = (
            observed_bin_age[:, None]
            - observed_bin_age[None, :]
        )

        mask = true_diff.ne(0)

        if mask.any():
            rank_loss = torch.relu(
                -true_diff[mask]
                * pred_diff[mask]
            ).mean()
        else:
            rank_loss = torch.zeros(
                (),
                device=probs.device
            )
    else:
        rank_loss = torch.zeros(
            (),
            device=probs.device
        )

    return (
        nll_weight * ordinal_nll
        + mae_weight * mae
        + rank_weight * rank_loss
    )


# ============================================================================
# 5. TRAINING HELPERS
# ============================================================================


def make_loader(X, y, batch_size, shuffle, seed):
    dataset = TensorDataset(
        torch.from_numpy(
            X.astype(
                np.float32,
                copy=False
            )
        ),
        torch.from_numpy(
            y.astype(
                np.int64,
                copy=False
            )
        )
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=min(
            batch_size,
            max(1, len(dataset))
        ),
        shuffle=shuffle,
        generator=generator if shuffle else None,
        drop_last=False
    )


def split_inner_donors(donor_ids, validation_fraction=0.20, random_seed=42):
    unique_donors = np.unique(
        np.asarray(donor_ids).astype(str)
    )

    if len(unique_donors) < 2:
        return unique_donors, np.array([], dtype=str)

    rng = np.random.RandomState(
        random_seed
    )

    rng.shuffle(
        unique_donors
    )

    n_validation = max(
        1,
        int(
            round(
                len(unique_donors)
                * validation_fraction
            )
        )
    )

    n_validation = min(
        n_validation,
        len(unique_donors) - 1
    )

    validation_donors = unique_donors[
        :n_validation
    ]

    training_donors = unique_donors[
        n_validation:
    ]

    return training_donors, validation_donors


def train_epochs(
    model,
    train_loader,
    bin_centers,
    device,
    max_epochs,
    learning_rate=1e-3,
    validation_loader=None,
    lr_patience=10,
    lr_factor=0.5,
    early_stopping_patience=15
):
    """Train a CLM and optionally select the best epoch on an inner validation set."""

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    scheduler = None

    if validation_loader is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=lr_patience,
            factor=lr_factor
        )

    centers_tensor = torch.as_tensor(
        bin_centers,
        dtype=torch.float32,
        device=device
    )

    best_state = None
    best_epoch = max_epochs
    best_validation_loss = np.inf
    without_improvement = 0
    history = []

    for epoch in range(
        1,
        max_epochs + 1
    ):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            probs, _, _ = model(
                batch_X
            )

            loss = clm_loss(
                probs,
                batch_y,
                centers_tensor
            )

            loss.backward()
            optimizer.step()

            train_loss_sum += (
                loss.item()
                * len(batch_X)
            )

            train_n += len(batch_X)

        train_loss = (
            train_loss_sum
            / max(train_n, 1)
        )

        if validation_loader is None:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": np.nan
                }
            )
            continue

        model.eval()
        val_loss_sum = 0.0
        val_n = 0

        with torch.no_grad():
            for batch_X, batch_y in validation_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                probs, _, _ = model(
                    batch_X
                )

                loss = clm_loss(
                    probs,
                    batch_y,
                    centers_tensor
                )

                val_loss_sum += (
                    loss.item()
                    * len(batch_X)
                )

                val_n += len(batch_X)

        validation_loss = (
            val_loss_sum
            / max(val_n, 1)
        )

        scheduler.step(
            validation_loss
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            without_improvement = 0
        else:
            without_improvement += 1

        if without_improvement >= early_stopping_patience:
            break

    if validation_loader is not None and best_state is not None:
        model.load_state_dict(
            best_state
        )

    return model, int(best_epoch), pd.DataFrame(history)


def train_one_outer_fold(
    X_train,
    ages_train,
    donor_ids_train,
    bin_width,
    device,
    fold_idx,
    hidden_dims=(128, 64),
    dropout=0.5,
    batch_size=256,
    max_epochs=100,
    learning_rate=1e-3,
    random_seed=42
):
    """Select an epoch using inner donors, then refit on all outer-training cells."""

    set_random_seed(
        random_seed + fold_idx
    )

    y_bins, bin_edges, bin_centers = age_to_bins(
        ages_train,
        bin_width=bin_width
    )

    inner_train_donors, inner_val_donors = split_inner_donors(
        donor_ids_train,
        validation_fraction=0.20,
        random_seed=random_seed + fold_idx
    )

    inner_train_mask = np.isin(
        donor_ids_train.astype(str),
        inner_train_donors
    )

    inner_val_mask = np.isin(
        donor_ids_train.astype(str),
        inner_val_donors
    )

    selector_model = NeuralCumulativeLink(
        X_train.shape[1],
        len(bin_centers),
        hidden_dims=hidden_dims,
        dropout=dropout
    ).to(device)

    inner_train_loader = make_loader(
        X_train[inner_train_mask],
        y_bins[inner_train_mask],
        batch_size=batch_size,
        shuffle=True,
        seed=random_seed + fold_idx
    )

    if inner_val_mask.any():
        inner_val_loader = make_loader(
            X_train[inner_val_mask],
            y_bins[inner_val_mask],
            batch_size=batch_size,
            shuffle=False,
            seed=random_seed + fold_idx
        )
    else:
        inner_val_loader = None

    selector_model, best_epoch, history = train_epochs(
        selector_model,
        inner_train_loader,
        bin_centers,
        device,
        max_epochs,
        learning_rate=learning_rate,
        validation_loader=inner_val_loader
    )

    # Refit a fresh model on all outer-training cells for the selected epoch count
    set_random_seed(
        random_seed + fold_idx
    )

    final_model = NeuralCumulativeLink(
        X_train.shape[1],
        len(bin_centers),
        hidden_dims=hidden_dims,
        dropout=dropout
    ).to(device)

    full_loader = make_loader(
        X_train,
        y_bins,
        batch_size=batch_size,
        shuffle=True,
        seed=random_seed + fold_idx
    )

    final_model, _, _ = train_epochs(
        final_model,
        full_loader,
        bin_centers,
        device,
        max(1, best_epoch),
        learning_rate=learning_rate,
        validation_loader=None
    )

    final_model.eval()

    return (
        final_model,
        bin_centers,
        bin_edges,
        best_epoch,
        history
    )


def predict_one_model(model, X, bin_centers, device, batch_size=256):
    predictions = []

    model.eval()

    with torch.no_grad():
        for start in range(
            0,
            len(X),
            batch_size
        ):
            stop = min(
                start + batch_size,
                len(X)
            )

            batch_X = torch.from_numpy(
                X[start:stop].astype(
                    np.float32,
                    copy=False
                )
            ).to(device)

            batch_pred = model.predict_age(
                batch_X,
                bin_centers
            )

            predictions.append(
                batch_pred.cpu().numpy()
            )

    if len(predictions) == 0:
        return np.empty(0)

    return np.concatenate(
        predictions
    )


# ============================================================================
# 6. TRAIN FIVE-FOLD CLM CLOCK
# ============================================================================


def train_clm_clock(
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
    n_folds=5,
    hidden_dims=(128, 64),
    dropout=0.5,
    batch_size=256,
    max_epochs=100,
    learning_rate=1e-3,
    random_seed=42,
    device=None,
    feature_genes=None
):
    """Train a cell-type-specific CLM ensemble for development, adult, or full life."""

    set_random_seed(random_seed)
    device_use = get_device(device)
    bin_width = get_clm_bin_width(stage)

    print("\n" + "=" * 70)
    print("CLM | {} | {}".format(celltype, stage))
    print("=" * 70)
    print("Device:", device_use)
    print("Age interval: {} years".format(bin_width))

    adata_ct = adata[
        (adata.obs[status_col].astype(str) == control_label)
        & (adata.obs[celltype_col].astype(str) == celltype),
        :
    ].copy()

    ages_all = pd.to_numeric(
        adata_ct.obs[age_col],
        errors="coerce"
    ).values

    stage_mask = get_stage_mask(
        ages_all,
        stage,
        age_cut=age_cut
    ) & np.isfinite(ages_all)

    adata_ct = adata_ct[
        stage_mask,
        :
    ].copy()

    if adata_ct.n_obs == 0:
        raise ValueError("No cells available for {} / {}".format(celltype, stage))

    adata_ct = normalize_adata(
        adata_ct,
        copy=False
    )

    if feature_genes is None:
        feature_genes = adata_ct.var_names.astype(str).tolist()
    else:
        feature_genes = [str(gene) for gene in feature_genes]

    X, n_missing = prepare_feature_matrix(
        adata_ct,
        feature_genes
    )

    if n_missing > 0:
        print("Training feature list contains {} absent genes.".format(n_missing))

    ages = pd.to_numeric(
        adata_ct.obs[age_col],
        errors="raise"
    ).values.astype(float)

    donors = make_unique_donor_id(
        adata_ct.obs,
        donor_col=donor_col,
        dataset_col=dataset_col
    )

    if len(np.unique(donors)) < n_folds:
        raise ValueError("Not enough donors for {}-fold CV".format(n_folds))

    print("Cells:", adata_ct.n_obs)
    print("Donors:", len(np.unique(donors)))
    print("Features:", len(feature_genes))
    print("Age range: {:.1f} - {:.1f}".format(ages.min(), ages.max()))

    stage_label = {
        "development": "Development",
        "adult": "Adult",
        "full": "Full"
    }[stage]
    age_cut_filename = 0 if stage == "full" else int(age_cut)
    public_stage_dir = os.path.join(
        output_dir,
        "CLM_{}".format(stage_label)
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

    splitter = GroupKFold(
        n_splits=n_folds
    )

    oof_prediction = np.full(
        adata_ct.n_obs,
        np.nan
    )

    oof_fold = np.full(
        adata_ct.n_obs,
        -1,
        dtype=int
    )

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        splitter.split(
            X,
            ages,
            donors
        ),
        start=1
    ):

        print("\nFold {}/{}".format(fold_idx, n_folds))

        model, bin_centers, bin_edges, best_epoch, history = train_one_outer_fold(
            X[train_idx],
            ages[train_idx],
            donors[train_idx],
            bin_width=bin_width,
            device=device_use,
            fold_idx=fold_idx,
            hidden_dims=hidden_dims,
            dropout=dropout,
            batch_size=batch_size,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            random_seed=random_seed
        )

        val_pred = predict_one_model(
            model,
            X[val_idx],
            bin_centers,
            device_use,
            batch_size=batch_size
        )

        oof_prediction[val_idx] = val_pred
        oof_fold[val_idx] = fold_idx

        cell_metrics = regression_metrics(
            ages[val_idx],
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
                "n_classes": len(bin_centers),
                "best_epoch": best_epoch,
                "n_train_cells": len(train_idx),
                "n_validation_cells": len(val_idx),
                "n_train_donors": len(np.unique(donors[train_idx])),
                "n_validation_donors": len(np.unique(donors[val_idx])),
                "cell_spearman_r": cell_metrics["spearman_r"],
                "cell_mae": cell_metrics["mae"],
                "donor_spearman_r": donor_metrics["spearman_r"],
                "donor_mae": donor_metrics["mae"]
            }
        )

        checkpoint = {
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "model_config": {
                "input_dim": int(X.shape[1]),
                "n_classes": int(len(bin_centers)),
                "hidden_dims": list(hidden_dims),
                "dropout": float(dropout)
            },
            "input_dim": int(X.shape[1]),
            "n_classes": int(len(bin_centers)),
            "hidden_dims": list(hidden_dims),
            "dropout": float(dropout),
            "bin_centers": np.asarray(bin_centers),
            "bin_edges": np.asarray(bin_edges),
            "best_epoch": int(best_epoch),
            "fold": int(fold_idx),
            "clm_bin_width": float(bin_width)
        }
        auxiliary_checkpoint_file = os.path.join(
            model_dir,
            "fold_{}.pt".format(fold_idx)
        )
        public_checkpoint_file = os.path.join(
            public_stage_dir,
            "{}_clm_5fold_model_fold{}_age_cut{}_n_features12779.pt".format(
                celltype,
                fold_idx,
                age_cut_filename
            )
        )
        torch.save(
            checkpoint,
            auxiliary_checkpoint_file
        )
        torch.save(
            checkpoint,
            public_checkpoint_file
        )

        history.to_csv(
            os.path.join(
                model_dir,
                "fold_{}_epoch_selection_history.csv".format(fold_idx)
            ),
            index=False
        )

        print(
            "Cell MAE={:.2f}, donor MAE={:.2f}, best epoch={}".format(
                cell_metrics["mae"],
                donor_metrics["mae"],
                best_epoch
            )
        )

        del model

        if device_use.type == "cuda":
            torch.cuda.empty_cache()

    public_feature_file = os.path.join(
        feature_dir,
        "{}_clm_5fold_features_age_cut{}_n_features12779.csv".format(
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
    public_model_paths = [
        os.path.join(
            public_stage_dir,
            "{}_clm_5fold_model_fold{}_age_cut{}_n_features12779.pt".format(
                celltype,
                fold_idx,
                age_cut_filename
            )
        )
        for fold_idx in range(1, n_folds + 1)
    ]
    public_ensemble_info = {
        "model_paths": public_model_paths,
        "features": list(feature_genes),
        "fold_val_results": fold_results,
        "n_models": int(n_folds),
        "clm_bin_width": float(bin_width),
        "age_cut": int(age_cut_filename),
        "celltype": celltype
    }
    public_metadata_file = os.path.join(
        public_stage_dir,
        "{}_clm_5fold_ensemble_metadata_age_cut{}_n_features12779.pkl".format(
            celltype,
            age_cut_filename
        )
    )
    joblib.dump(
        public_ensemble_info,
        public_metadata_file
    )
    joblib.dump(
        {
            "model_paths": public_model_paths,
            "metadata_path": public_metadata_file,
            "best_model_path": public_model_paths[0]
        },
        os.path.join(
            public_stage_dir,
            "{}_clm_5fold_ensemble_paths_age_cut{}_n_features12779.pkl".format(
                celltype,
                age_cut_filename
            )
        )
    )
    pd.DataFrame(
        fold_results
    ).to_csv(
        os.path.join(
            validation_dir,
            "{}_clm_5fold_validation_summary_age_cut{}_n_features12779.csv".format(
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

    pd.DataFrame(
        fold_results
    ).to_csv(
        os.path.join(
            model_dir,
            "fold_validation_metrics.csv"
        ),
        index=False
    )

    oof_cell = adata_ct.obs.copy()
    oof_cell["true_age"] = ages
    oof_cell["clm_oof_prediction"] = oof_prediction
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
        ages,
        oof_prediction
    )

    oof_donor_metrics = regression_metrics(
        oof_donor["chronological_age"],
        oof_donor["predicted_age"]
    )

    metadata = {
        "model_type": "clm",
        "celltype": celltype,
        "stage": stage,
        "age_cut": float(age_cut),
        "bin_width": float(bin_width),
        "hidden_dims": list(hidden_dims),
        "dropout": float(dropout),
        "batch_size": int(batch_size),
        "max_epochs": int(max_epochs),
        "learning_rate": float(learning_rate),
        "n_folds": int(n_folds),
        "random_seed": int(random_seed),
        "n_cells": int(adata_ct.n_obs),
        "n_donors": int(len(np.unique(donors))),
        "n_features": int(len(feature_genes)),
        "public_age_cut_filename": int(age_cut_filename),
        "public_feature_file": public_feature_file,
        "public_ensemble_metadata_file": public_metadata_file,
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

    print("\nOOF cell MAE: {:.2f}".format(oof_cell_metrics["mae"]))
    print("OOF donor MAE: {:.2f}".format(oof_donor_metrics["mae"]))
    print("Public CLM stage directory:", public_stage_dir)
    print("Auxiliary training files saved to:", model_dir)

    return metadata


# ============================================================================
# 7. LOAD AND PREDICT
# ============================================================================


def load_clm_ensemble(model_dir, device=None):
    device_use = get_device(device)

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
    bin_centers_list = []

    for fold_idx in range(
        1,
        int(metadata["n_folds"]) + 1
    ):
        checkpoint = torch.load(
            os.path.join(
                model_dir,
                "fold_{}.pt".format(fold_idx)
            ),
            map_location=device_use
        )

        model = NeuralCumulativeLink(
            checkpoint["input_dim"],
            checkpoint["n_classes"],
            hidden_dims=tuple(checkpoint["hidden_dims"]),
            dropout=checkpoint["dropout"]
        ).to(device_use)

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.eval()

        models.append(model)
        bin_centers_list.append(
            np.asarray(
                checkpoint["bin_centers"],
                dtype=float
            )
        )

    return (
        models,
        bin_centers_list,
        feature_genes,
        metadata,
        device_use
    )


def predict_clm_ensemble(
    adata,
    model_dir,
    normalize=True,
    chunk_size=5000,
    device=None,
    return_fold_predictions=False
):
    """Predict ages using the mean of five fold-specific CLM models."""

    models, centers_list, feature_genes, metadata, device_use = load_clm_ensemble(
        model_dir,
        device=device
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

        fold_predictions = []

        for model, centers in zip(
            models,
            centers_list
        ):
            fold_predictions.append(
                predict_one_model(
                    model,
                    X,
                    centers,
                    device_use,
                    batch_size=int(metadata["batch_size"])
                )
            )

        fold_predictions = np.column_stack(
            fold_predictions
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

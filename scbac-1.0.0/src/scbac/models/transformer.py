"""
Transformer functions for the scBAC pipeline.

Python version: 3.9.20

Stage definitions
-----------------
development : Age_at_death <= 18
adult       : Age_at_death > 18
full        : all ages, comparison model only

Architecture
------------
- ordered genes divided consecutively into 20 equally sized tokens
- complete expression vector in each token projected to 128 dimensions
- learnable positional embeddings
- one Transformer encoder layer
- four attention heads
- 256-dimensional feed-forward layer
- GELU activation and dropout 0.5
- global average pooling
- LayerNorm -> Linear(128,64) -> GELU -> Dropout -> Linear(64,1)

Training
--------
- MSE loss
- AdamW, learning rate 1e-3, weight decay 1e-4
- batch size 32
- maximum 100 epochs
- ReduceLROnPlateau factor 0.5, patience 5
- early stopping patience 10
- gradient clipping at 1.0
- donor-grouped five-fold cross-validation
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
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# 2. BASIC FUNCTIONS
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


def make_unique_donor_id(obs, donor_col="donor_id", dataset_col="dataset"):
    if dataset_col in obs.columns:
        return (
            obs[dataset_col].astype(str)
            + "::"
            + obs[donor_col].astype(str)
        ).values

    return obs[donor_col].astype(str).values


def prepare_transformer_features(var_names, num_tokens=20):
    """
    Keep all ordered lineage genes except the minimum number from the end that
    must be removed so the feature number is divisible by 20 tokens.
    """

    feature_genes = [str(gene) for gene in var_names]

    if len(feature_genes) < num_tokens:
        raise ValueError("At least {} genes are required".format(num_tokens))

    n_keep = (
        len(feature_genes) // num_tokens
    ) * num_tokens

    n_removed = len(feature_genes) - n_keep

    if n_removed > 0:
        print(
            "Removing the final {} gene(s) so {} features are divisible by {} tokens.".format(
                n_removed,
                n_keep,
                num_tokens
            )
        )

    return feature_genes[:n_keep]


def prepare_feature_matrix(adata, feature_genes):
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


def make_donor_folds(donor_ids, n_folds=5, random_seed=42):
    """Shuffle unique donors once and split them into donor-disjoint folds."""

    unique_donors = np.unique(
        np.asarray(donor_ids).astype(str)
    )

    if len(unique_donors) < n_folds:
        raise ValueError("Not enough donors for {}-fold CV".format(n_folds))

    rng = np.random.RandomState(
        random_seed
    )

    rng.shuffle(
        unique_donors
    )

    return [
        np.asarray(fold, dtype=str)
        for fold in np.array_split(
            unique_donors,
            n_folds
        )
    ]


# ============================================================================
# 3. TRANSFORMER ARCHITECTURE
# ============================================================================


class GeneAttentionAgePredictor(nn.Module):
    """Transformer-based transcriptomic age regressor used in scBACs."""

    def __init__(
        self,
        input_size,
        embed_dim=128,
        num_heads=4,
        num_layers=1,
        dropout_prob=0.5,
        num_tokens=20
    ):
        super(GeneAttentionAgePredictor, self).__init__()

        if input_size % num_tokens != 0:
            raise ValueError(
                "input_size must be divisible by num_tokens"
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads"
            )

        self.input_size = int(input_size)
        self.embed_dim = int(embed_dim)
        self.num_tokens = int(num_tokens)
        self.token_dim = self.input_size // self.num_tokens

        self.projector = nn.Linear(
            self.token_dim,
            self.embed_dim
        )

        self.pos_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.num_tokens,
                self.embed_dim
            )
        )

        self.input_dropout = nn.Dropout(
            dropout_prob
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=self.embed_dim * 2,
            dropout=dropout_prob,
            activation="gelu",
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.regressor = nn.Sequential(
            nn.LayerNorm(
                self.embed_dim
            ),
            nn.Linear(
                self.embed_dim,
                self.embed_dim // 2
            ),
            nn.GELU(),
            nn.Dropout(
                dropout_prob
            ),
            nn.Linear(
                self.embed_dim // 2,
                1
            )
        )

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

    def forward(self, x):
        batch_size = x.shape[0]

        x = x.reshape(
            batch_size,
            self.num_tokens,
            self.token_dim
        )

        x = self.projector(
            x
        )

        x = x + self.pos_embedding
        x = self.input_dropout(
            x
        )

        x = self.transformer_encoder(
            x
        )

        x = x.mean(
            dim=1
        )

        return self.regressor(
            x
        )


# ============================================================================
# 4. TRAIN ONE FOLD
# ============================================================================


def train_one_fold(
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    fold_idx,
    num_tokens=20,
    embed_dim=128,
    num_heads=4,
    num_layers=1,
    dropout_prob=0.5,
    learning_rate=1e-3,
    weight_decay=1e-4,
    batch_size=32,
    max_epochs=100,
    lr_patience=5,
    lr_factor=0.5,
    early_stopping_patience=10,
    gradient_clip_norm=1.0,
    random_seed=42
):
    """Train one fold-specific Transformer and retain its best validation checkpoint."""

    set_random_seed(
        random_seed + fold_idx
    )

    train_dataset = TensorDataset(
        torch.from_numpy(
            X_train.astype(
                np.float32,
                copy=False
            )
        ),
        torch.from_numpy(
            y_train.astype(
                np.float32,
                copy=False
            )
        ).reshape(-1, 1)
    )

    val_dataset = TensorDataset(
        torch.from_numpy(
            X_val.astype(
                np.float32,
                copy=False
            )
        ),
        torch.from_numpy(
            y_val.astype(
                np.float32,
                copy=False
            )
        ).reshape(-1, 1)
    )

    generator = torch.Generator()
    generator.manual_seed(
        random_seed + fold_idx
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = GeneAttentionAgePredictor(
        input_size=X_train.shape[1],
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_prob=dropout_prob,
        num_tokens=num_tokens
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience
    )

    best_validation_loss = np.inf
    best_epoch = 0
    best_state = None
    without_improvement = 0
    history = []

    for epoch in range(
        1,
        max_epochs + 1
    ):
        model.train()
        train_loss_sum = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            prediction = model(
                batch_X
            )

            loss = criterion(
                prediction,
                batch_y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm
            )

            optimizer.step()

            train_loss_sum += (
                loss.item()
                * batch_X.shape[0]
            )

        train_loss = (
            train_loss_sum
            / len(train_loader.dataset)
        )

        model.eval()
        validation_loss_sum = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                prediction = model(
                    batch_X
                )

                loss = criterion(
                    prediction,
                    batch_y
                )

                validation_loss_sum += (
                    loss.item()
                    * batch_X.shape[0]
                )

        validation_loss = (
            validation_loss_sum
            / len(val_loader.dataset)
        )

        scheduler.step(
            validation_loss
        )

        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "validation_mse": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"]
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

        if epoch == 1 or epoch % 10 == 0:
            print(
                "Epoch {:3d}: train MSE={:.4f}, val MSE={:.4f}".format(
                    epoch,
                    train_loss,
                    validation_loss
                )
            )

        if without_improvement >= early_stopping_patience:
            print(
                "Early stopping at epoch {}; best epoch={}".format(
                    epoch,
                    best_epoch
                )
            )
            break

    if best_state is None:
        raise RuntimeError("No valid Transformer checkpoint was produced")

    model.load_state_dict(
        best_state
    )

    model.eval()

    return (
        model,
        best_epoch,
        best_validation_loss,
        pd.DataFrame(history)
    )


def predict_one_model(model, X, device, batch_size=32):
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

            predictions.append(
                model(
                    batch_X
                ).cpu().numpy().reshape(-1)
            )

    if len(predictions) == 0:
        return np.empty(0)

    return np.concatenate(
        predictions
    )


# ============================================================================
# 5. TRAIN FIVE-FOLD TRANSFORMER CLOCK
# ============================================================================


def train_transformer_clock(
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
    num_tokens=20,
    embed_dim=128,
    num_heads=4,
    num_layers=1,
    dropout_prob=0.5,
    learning_rate=1e-3,
    weight_decay=1e-4,
    batch_size=32,
    max_epochs=100,
    lr_patience=5,
    lr_factor=0.5,
    early_stopping_patience=10,
    gradient_clip_norm=1.0,
    random_seed=42,
    device=None,
    feature_genes=None
):
    """Train one cell-type- and stage-specific five-fold Transformer ensemble."""

    set_random_seed(random_seed)
    device_use = get_device(device)

    print("\n" + "=" * 70)
    print("Transformer | {} | {}".format(celltype, stage))
    print("=" * 70)
    print("Device:", device_use)

    adata_ct = adata[
        (adata.obs[status_col].astype(str) == control_label)
        & (adata.obs[celltype_col].astype(str) == celltype),
        :
    ].copy()

    if adata_ct.n_obs == 0:
        raise ValueError("No control cells found for {}".format(celltype))

    # Normalize once for the requested lineage
    adata_ct = normalize_adata(
        adata_ct,
        copy=False
    )

    # Define the ordered feature space before applying the stage split, matching
    # the original working Transformer script.
    if feature_genes is None:
        feature_genes = prepare_transformer_features(
            adata_ct.var_names,
            num_tokens=num_tokens
        )
    else:
        feature_genes = [str(gene) for gene in feature_genes]

        n_keep = (
            len(feature_genes) // num_tokens
        ) * num_tokens

        feature_genes = feature_genes[:n_keep]

    ages_all = pd.to_numeric(
        adata_ct.obs[age_col],
        errors="coerce"
    ).values

    stage_mask = get_stage_mask(
        ages_all,
        stage,
        age_cut=age_cut
    ) & np.isfinite(ages_all)

    adata_stage = adata_ct[
        stage_mask,
        :
    ].copy()

    X, n_missing = prepare_feature_matrix(
        adata_stage,
        feature_genes
    )

    if n_missing > 0:
        print("Training feature list contains {} absent genes.".format(n_missing))

    y = pd.to_numeric(
        adata_stage.obs[age_col],
        errors="raise"
    ).values.astype(float)

    donors = make_unique_donor_id(
        adata_stage.obs,
        donor_col=donor_col,
        dataset_col=dataset_col
    )

    folds = make_donor_folds(
        donors,
        n_folds=n_folds,
        random_seed=random_seed
    )

    print("Cells:", adata_stage.n_obs)
    print("Donors:", len(np.unique(donors)))
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
        "Transf_{}".format(stage_label)
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

    oof_prediction = np.full(
        adata_stage.n_obs,
        np.nan
    )

    oof_fold = np.full(
        adata_stage.n_obs,
        -1,
        dtype=int
    )

    fold_results = []

    for fold_idx, validation_donors in enumerate(
        folds,
        start=1
    ):
        print("\nFold {}/{}".format(fold_idx, n_folds))

        val_mask = np.isin(
            donors,
            validation_donors
        )

        train_mask = ~val_mask

        train_idx = np.where(
            train_mask
        )[0]

        val_idx = np.where(
            val_mask
        )[0]

        model, best_epoch, best_val_loss, history = train_one_fold(
            X[train_idx],
            y[train_idx],
            X[val_idx],
            y[val_idx],
            device=device_use,
            fold_idx=fold_idx,
            num_tokens=num_tokens,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
            early_stopping_patience=early_stopping_patience,
            gradient_clip_norm=gradient_clip_norm,
            random_seed=random_seed
        )

        val_pred = predict_one_model(
            model,
            X[val_idx],
            device_use,
            batch_size=batch_size
        )

        oof_prediction[val_idx] = val_pred
        oof_fold[val_idx] = fold_idx

        cell_metrics = regression_metrics(
            y[val_idx],
            val_pred
        )

        donor_df = donor_median_predictions(
            adata_stage.obs.iloc[val_idx],
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
                "best_epoch": best_epoch,
                "best_validation_mse": best_val_loss,
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
                "input_size": int(X.shape[1]),
                "num_tokens": int(num_tokens),
                "embed_dim": int(embed_dim),
                "num_heads": int(num_heads),
                "num_layers": int(num_layers),
                "dropout_prob": float(dropout_prob)
            },
            "input_size": int(X.shape[1]),
            "num_tokens": int(num_tokens),
            "embed_dim": int(embed_dim),
            "num_heads": int(num_heads),
            "num_layers": int(num_layers),
            "dropout_prob": float(dropout_prob),
            "fold": int(fold_idx),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss)
        }
        auxiliary_checkpoint_file = os.path.join(
            model_dir,
            "fold_{}.pt".format(fold_idx)
        )
        public_checkpoint_file = os.path.join(
            public_stage_dir,
            "{}_geneatt_model_fold{}_age_cut{}.pt".format(
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
                "fold_{}_history.csv".format(fold_idx)
            ),
            index=False
        )

        print(
            "Cell MAE={:.2f}, donor MAE={:.2f}".format(
                cell_metrics["mae"],
                donor_metrics["mae"]
            )
        )

        del model

        if device_use.type == "cuda":
            torch.cuda.empty_cache()

    public_feature_file = os.path.join(
        feature_dir,
        "{}_geneatt_features_age_cut{}.csv".format(
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
            "{}_geneatt_model_fold{}_age_cut{}.pt".format(
                celltype,
                fold_idx,
                age_cut_filename
            )
        )
        for fold_idx in range(1, n_folds + 1)
    ]
    public_ensemble_info = {
        "n_folds": int(n_folds),
        "model_paths": public_model_paths,
        "fold_info": fold_results,
        "feature_genes": list(feature_genes),
        "model_config": {
            "input_size": int(X.shape[1]),
            "num_tokens": int(num_tokens),
            "embed_dim": int(embed_dim),
            "num_heads": int(num_heads),
            "num_layers": int(num_layers),
            "dropout_prob": float(dropout_prob)
        }
    }
    public_ensemble_info_file = os.path.join(
        public_stage_dir,
        "{}_geneatt_ensemble_info_age_cut{}.pkl".format(
            celltype,
            age_cut_filename
        )
    )
    joblib.dump(
        public_ensemble_info,
        public_ensemble_info_file
    )
    pd.DataFrame(
        fold_results
    ).to_csv(
        os.path.join(
            validation_dir,
            "{}_geneatt_cv_fold_info_age_cut{}.csv".format(
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

    oof_cell = adata_stage.obs.copy()
    oof_cell["true_age"] = y
    oof_cell["transformer_oof_prediction"] = oof_prediction
    oof_cell["cv_fold"] = oof_fold
    oof_cell.index.name = "cell_id"

    oof_cell.to_csv(
        os.path.join(
            model_dir,
            "oof_cell_predictions.csv"
        )
    )

    oof_donor = donor_median_predictions(
        adata_stage.obs,
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
        "model_type": "transformer",
        "celltype": celltype,
        "stage": stage,
        "age_cut": float(age_cut),
        "num_tokens": int(num_tokens),
        "embed_dim": int(embed_dim),
        "num_heads": int(num_heads),
        "num_layers": int(num_layers),
        "dropout_prob": float(dropout_prob),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "max_epochs": int(max_epochs),
        "lr_patience": int(lr_patience),
        "lr_factor": float(lr_factor),
        "early_stopping_patience": int(early_stopping_patience),
        "gradient_clip_norm": float(gradient_clip_norm),
        "n_folds": int(n_folds),
        "random_seed": int(random_seed),
        "n_cells": int(adata_stage.n_obs),
        "n_donors": int(len(np.unique(donors))),
        "n_features": int(len(feature_genes)),
        "public_age_cut_filename": int(age_cut_filename),
        "public_feature_file": public_feature_file,
        "public_ensemble_info_file": public_ensemble_info_file,
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
    print("Public Transformer stage directory:", public_stage_dir)
    print("Auxiliary training files saved to:", model_dir)

    return metadata


# ============================================================================
# 6. LOAD AND PREDICT
# ============================================================================


def load_transformer_ensemble(model_dir, device=None):
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

        model = GeneAttentionAgePredictor(
            input_size=checkpoint["input_size"],
            embed_dim=checkpoint["embed_dim"],
            num_heads=checkpoint["num_heads"],
            num_layers=checkpoint["num_layers"],
            dropout_prob=checkpoint["dropout_prob"],
            num_tokens=checkpoint["num_tokens"]
        ).to(device_use)

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.eval()
        models.append(model)

    return (
        models,
        feature_genes,
        metadata,
        device_use
    )


def predict_transformer_ensemble(
    adata,
    model_dir,
    normalize=True,
    chunk_size=5000,
    device=None,
    return_fold_predictions=False
):
    """Predict ages using the mean of five fold-specific Transformer models."""

    models, feature_genes, metadata, device_use = load_transformer_ensemble(
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

        fold_predictions = np.column_stack(
            [
                predict_one_model(
                    model,
                    X,
                    device_use,
                    batch_size=int(metadata["batch_size"])
                )
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

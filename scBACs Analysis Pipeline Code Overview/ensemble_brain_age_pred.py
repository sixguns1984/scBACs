"""
Fixed pretrained scBAC prediction utilities.

This module intentionally follows the model directory and filename conventions
used by the original scBAC prediction code. It is for APPLYING the supplied
pretrained models to new data; it does not retrain, refit, recalibrate, or tune
any model.

Expected public model layout
----------------------------
models/
├── benchmarking_model/
│   ├── sc_Exc.csv
│   ├── sc_Inh.csv
│   └── ...
└── full_genes_predictions_20260415/
    ├── Transf_Development/
    ├── Transf_Adult/
    ├── Transf_Full/
    ├── ElasticNet_Development/
    ├── ElasticNet_Adult/
    ├── ElasticNet_Full/
    ├── CLM_Development/
    ├── CLM_Adult/
    └── CLM_Full/

Within those directories, the exact filenames are the same as in the original
working prediction code:

Transformer
~~~~~~~~~~~
{ct}_geneatt_ensemble_info_age_cut{age_cut}.pkl
{ct}_geneatt_model_fold{fold}_age_cut{age_cut}.pt
model_features/{ct}_geneatt_features_age_cut{age_cut}.csv

Elastic Net
~~~~~~~~~~~
{ct}_elasticnet_5fold_ensemble_age_cut{age_cut}_n_features12779.pkl
model_features/{ct}_elasticnet_features_age_cut{age_cut}_n_features12779.csv

CLM
~~~
{ct}_clm_5fold_model_fold{fold}_age_cut{age_cut}_n_features12779.pt
model_features/{ct}_clm_5fold_features_age_cut{age_cut}_n_features12779.csv

Benchmarking model
~~~~~~~~~~~~~~~~~~
benchmarking_model/sc_{ct}.csv

Stage filename convention inherited from the original files:
- Development: age_cut18
- Adult:       age_cut18
- Full:        age_cut0

Python 3.9.20 compatible.
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

import os
import gc
import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse

from transformer_clock import GeneAttentionAgePredictor
from clm_clock import NeuralCumulativeLink


# ============================================================================
# 2. ORIGINAL MODEL NAMING CONVENTIONS
# ============================================================================

N_FEATURES = 12779

MODEL_AGE_CUT = {
    "transf_Development": 18,
    "transf_Adult": 18,
    "transf_Full": 0,
    "elasticnet_Development": 18,
    "elasticnet_Adult": 18,
    "elasticnet_Full": 0,
    "clm_Development": 18,
    "clm_Adult": 18,
    "clm_Full": 0,
}

MODEL_SUBDIRS = {
    "transf_Development": "Transf_Development",
    "transf_Adult": "Transf_Adult",
    "transf_Full": "Transf_Full",
    "elasticnet_Development": "ElasticNet_Development",
    "elasticnet_Adult": "ElasticNet_Adult",
    "elasticnet_Full": "ElasticNet_Full",
    "clm_Development": "CLM_Development",
    "clm_Adult": "CLM_Adult",
    "clm_Full": "CLM_Full",
}


# ============================================================================
# 3. SMALL HELPERS
# ============================================================================

def _torch_load_checkpoint(path, device):
    """Load original torch checkpoint dictionaries across PyTorch versions.

    PyTorch >=2.6 defaults to weights_only=True. The original scBAC checkpoint
    dictionaries contain model configuration and bin-center metadata in addition
    to tensors, so full checkpoint loading is required. Older PyTorch versions do
    not accept the weights_only keyword; the TypeError fallback preserves
    compatibility with those environments.
    """

    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device
        )


def _read_feature_genes(path):
    """Read the original feature CSV, requiring the original 'genename' field."""

    features_df = pd.read_csv(path)

    if "genename" not in features_df.columns:
        raise KeyError(
            "Expected original feature column 'genename' in {}".format(path)
        )

    return features_df["genename"].astype(str).tolist()


def _normalize_log1p(adata):
    """Apply the same preprocessing used by the original prediction code."""

    adata = adata.copy()
    adata.X = adata.X.astype("float32")
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    return adata


def _build_feature_matrix(adata, feature_genes):
    """
    Reindex expression to the exact training feature order.

    Missing genes are assigned expression 0, matching the original prediction
    behavior for unavailable model features.
    """

    feature_genes = [str(gene) for gene in feature_genes]
    var_names = pd.Index(adata.var_names.astype(str))
    positions = var_names.get_indexer(feature_genes)

    X = np.zeros(
        (adata.n_obs, len(feature_genes)),
        dtype=np.float32
    )

    present_model_positions = np.where(positions >= 0)[0]

    if len(present_model_positions) > 0:
        present_data_positions = positions[present_model_positions]
        block = adata[:, present_data_positions].X

        if sparse.issparse(block):
            block = block.toarray()

        X[:, present_model_positions] = np.asarray(
            block,
            dtype=np.float32
        )

    return X


def add_ensemble_predictions(result_df, age_col="Age_at_death", age_cut=18):
    """Create the original ensemble and development/adult routed columns."""

    result_df = result_df.copy()

    for stage in [
        "Development",
        "Adult",
        "Full"
    ]:
        cols = [
            "elasticnet_{}".format(stage),
            "clm_{}".format(stage),
            "transf_{}".format(stage),
        ]

        if all(col in result_df.columns for col in cols):
            result_df[
                "Ensemble_{}".format(stage)
            ] = result_df[cols].mean(axis=1)

    if age_col in result_df.columns:
        ages = pd.to_numeric(
            result_df[age_col],
            errors="coerce"
        )

        for model_type in [
            "elasticnet",
            "clm",
            "transf"
        ]:
            adult_col = "{}_Adult".format(model_type)
            development_col = "{}_Development".format(model_type)
            output_col = "{}_adult_deve".format(model_type)

            if (
                adult_col in result_df.columns
                and development_col in result_df.columns
            ):
                result_df[output_col] = result_df[adult_col].values
                development_mask = ages <= age_cut
                result_df.loc[
                    development_mask,
                    output_col
                ] = result_df.loc[
                    development_mask,
                    development_col
                ].values

        if (
            "Ensemble_Adult" in result_df.columns
            and "Ensemble_Development" in result_df.columns
        ):
            result_df["Ensemble_adult_deve"] = result_df[
                "Ensemble_Adult"
            ].values

            development_mask = ages <= age_cut
            result_df.loc[
                development_mask,
                "Ensemble_adult_deve"
            ] = result_df.loc[
                development_mask,
                "Ensemble_Development"
            ].values

    return result_df


# ============================================================================
# 4. BENCHMARKING CLOCK
# ============================================================================

def predict_cell_age_vectorized(
    adata,
    clock_df,
    fold_column="Fold 1 clock"
):
    """Apply one published benchmarking-clock fold exactly as in the original code."""

    feature_col = "clock feature (intercept or gene)"

    intercept_rows = clock_df[
        clock_df[feature_col] == "intercept"
    ]

    if intercept_rows.empty:
        raise ValueError(
            "Benchmarking clock does not contain an intercept row."
        )

    intercept = float(
        intercept_rows[
            fold_column
        ].iloc[0]
    )

    coefficients = clock_df[
        clock_df[feature_col] != "intercept"
    ].copy()

    coefficients = coefficients[
        pd.to_numeric(
            coefficients[fold_column],
            errors="coerce"
        ).fillna(0) != 0
    ].copy()

    model_genes = coefficients[
        feature_col
    ].astype(str).tolist()

    present_genes = [
        gene
        for gene in model_genes
        if gene in adata.var_names
    ]

    if len(present_genes) == 0:
        return np.full(
            adata.n_obs,
            intercept,
            dtype=float
        )

    X = adata[:, present_genes].X

    if sparse.issparse(X):
        X = X.toarray()

    weights = (
        coefficients
        .set_index(feature_col)
        .loc[present_genes, fold_column]
        .astype(float)
        .values
    )

    return np.dot(
        np.asarray(X),
        weights
    ) + intercept


def benchmarking_model(adata, clock_path):
    """
    Apply the supplied published benchmarking model and average its fold clocks.

    The expected filename is benchmarking_model/sc_{celltype}.csv.
    """

    clock_df = pd.read_csv(
        clock_path
    )

    fold_cols = [
        col
        for col in clock_df.columns
        if "Fold" in col
    ]

    if len(fold_cols) == 0:
        raise ValueError(
            "No benchmarking fold columns were found in {}".format(clock_path)
        )

    results = pd.DataFrame(
        index=adata.obs_names
    )

    for fold_col in fold_cols:
        results[fold_col] = predict_cell_age_vectorized(
            adata,
            clock_df,
            fold_column=fold_col
        )

    results["Benchmarking"] = results[
        fold_cols
    ].mean(axis=1)

    return results


# ============================================================================
# 5. FIXED PRETRAINED scBAC LOADER
# ============================================================================

class UnifiedAgePredictor:
    """
    Load and apply the supplied fixed scBAC model files.

    Parameters
    ----------
    device : str or torch.device
        CPU/GPU used for Transformer and CLM inference.

    model_root : str, optional
        Parent directory containing:
        - full_genes_predictions_20260415/
        - benchmarking_model/

        If omitted, defaults to <this_script_directory>/models.
    """

    def __init__(
        self,
        device,
        max_workers=4,
        stages=None,
        model_types=None,
        model_root=None,
        strict_five_folds=True
    ):
        self.device = torch.device(device)
        self.max_workers = max_workers
        self.stages = stages if stages is not None else [
            "Development",
            "Adult",
            "Full"
        ]
        self.model_types = model_types if model_types is not None else [
            "transf",
            "elasticnet",
            "clm"
        ]
        self.strict_five_folds = strict_five_folds

        if model_root is None:
            model_root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "models"
            )

        self.model_root = os.path.abspath(
            model_root
        )

        self.scbac_model_dir = os.path.join(
            self.model_root
        )

        self.benchmarking_dir = os.path.join(
            self.model_root,
            "benchmarking_model"
        )

        self._models_cache = {}

        print("UnifiedAgePredictor")
        print("  Fixed model root:", self.model_root)
        print("  Stages:", self.stages)
        print("  Model types:", self.model_types)

    def _model_dir(self, model_name):
        return os.path.join(
            self.scbac_model_dir,
            MODEL_SUBDIRS[model_name]
        )

    def _feature_dir(self, model_name):
        return os.path.join(
            self._model_dir(model_name),
            "model_features"
        )

    def _require_fold_count(self, paths, model_name, celltype):
        if self.strict_five_folds and len(paths) != 5:
            raise FileNotFoundError(
                "Expected 5 pretrained folds for {} / {}, found {}.".format(
                    model_name,
                    celltype,
                    len(paths)
                )
            )

    def load_transf_model(self, ct, stage):
        model_name = "transf_{}".format(stage)
        model_dir = self._model_dir(model_name)
        age_cut = MODEL_AGE_CUT[model_name]

        ensemble_info_path = os.path.join(
            model_dir,
            "{}_geneatt_ensemble_info_age_cut{}.pkl".format(
                ct,
                age_cut
            )
        )

        if not os.path.exists(ensemble_info_path):
            raise FileNotFoundError(ensemble_info_path)

        ensemble_info = joblib.load(
            ensemble_info_path
        )

        features = ensemble_info.get(
            "feature_genes"
        )

        if features is None:
            features_path = os.path.join(
                self._feature_dir(model_name),
                "{}_geneatt_features_age_cut{}.csv".format(
                    ct,
                    age_cut
                )
            )
            features = _read_feature_genes(
                features_path
            )
        else:
            features = [str(gene) for gene in features]

        model_paths = []

        for fold in range(1, 6):
            path = os.path.join(
                model_dir,
                "{}_geneatt_model_fold{}_age_cut{}.pt".format(
                    ct,
                    fold,
                    age_cut
                )
            )
            if os.path.exists(path):
                model_paths.append(path)

        self._require_fold_count(
            model_paths,
            model_name,
            ct
        )

        models = []
        default_config = ensemble_info.get(
            "model_config",
            {}
        )

        for path in model_paths:
            checkpoint = _torch_load_checkpoint(
                path,
                self.device
            )

            config = checkpoint.get(
                "model_config",
                default_config
            )

            model = GeneAttentionAgePredictor(
                input_size=config.get(
                    "input_size",
                    len(features)
                ),
                embed_dim=config.get(
                    "embed_dim",
                    128
                ),
                num_heads=config.get(
                    "num_heads",
                    4
                ),
                num_layers=config.get(
                    "num_layers",
                    1
                ),
                dropout_prob=config.get(
                    "dropout_prob",
                    0.5
                ),
                num_tokens=config.get(
                    "num_tokens",
                    20
                )
            ).to(
                self.device
            )

            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )
            model.eval()
            models.append(model)

        return {
            "models": models,
            "features": features,
            "type": "transf",
            "stage": stage,
            "age_cut": age_cut,
        }

    def load_elasticnet_model(self, ct, stage):
        model_name = "elasticnet_{}".format(stage)
        model_dir = self._model_dir(model_name)
        age_cut = MODEL_AGE_CUT[model_name]

        model_path = os.path.join(
            model_dir,
            "{}_elasticnet_5fold_ensemble_age_cut{}_n_features{}.pkl".format(
                ct,
                age_cut,
                N_FEATURES
            )
        )

        features_path = os.path.join(
            self._feature_dir(model_name),
            "{}_elasticnet_features_age_cut{}_n_features{}.csv".format(
                ct,
                age_cut,
                N_FEATURES
            )
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        if not os.path.exists(features_path):
            raise FileNotFoundError(features_path)

        models = joblib.load(
            model_path
        )

        if not isinstance(models, (list, tuple)):
            models = [models]

        self._require_fold_count(
            list(models),
            model_name,
            ct
        )

        return {
            "models": list(models),
            "features": _read_feature_genes(features_path),
            "type": "elasticnet",
            "stage": stage,
            "age_cut": age_cut,
        }

    def load_clm_model(self, ct, stage):
        model_name = "clm_{}".format(stage)
        model_dir = self._model_dir(model_name)
        age_cut = MODEL_AGE_CUT[model_name]

        features_path = os.path.join(
            self._feature_dir(model_name),
            "{}_clm_5fold_features_age_cut{}_n_features{}.csv".format(
                ct,
                age_cut,
                N_FEATURES
            )
        )

        if not os.path.exists(features_path):
            raise FileNotFoundError(features_path)

        features = _read_feature_genes(
            features_path
        )

        model_paths = []

        for fold in range(1, 6):
            path = os.path.join(
                model_dir,
                "{}_clm_5fold_model_fold{}_age_cut{}_n_features{}.pt".format(
                    ct,
                    fold,
                    age_cut,
                    N_FEATURES
                )
            )

            if os.path.exists(path):
                model_paths.append(path)

        self._require_fold_count(
            model_paths,
            model_name,
            ct
        )

        models = []
        bin_centers_list = []

        for path in model_paths:
            checkpoint = _torch_load_checkpoint(
                path,
                self.device
            )

            config = checkpoint.get(
                "model_config",
                {}
            )

            model = NeuralCumulativeLink(
                input_dim=config.get(
                    "input_dim",
                    len(features)
                ),
                n_classes=config.get(
                    "n_classes",
                    len(
                        checkpoint[
                            "bin_centers"
                        ]
                    )
                ),
                hidden_dims=config.get(
                    "hidden_dims",
                    [128, 64]
                ),
                dropout=config.get(
                    "dropout",
                    0.5
                )
            ).to(
                self.device
            )

            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )
            model.eval()
            models.append(model)
            bin_centers_list.append(
                np.asarray(
                    checkpoint[
                        "bin_centers"
                    ],
                    dtype=np.float32
                )
            )

        return {
            "models": models,
            "bin_centers_list": bin_centers_list,
            "features": features,
            "type": "clm",
            "stage": stage,
            "age_cut": age_cut,
        }

    def load_all_models_for_celltype(self, ct):
        cache_key = (
            ct,
            tuple(self.stages),
            tuple(self.model_types)
        )

        if cache_key in self._models_cache:
            return self._models_cache[cache_key]

        models_info = {}

        for model_type in self.model_types:
            for stage in self.stages:
                if model_type == "transf":
                    info = self.load_transf_model(
                        ct,
                        stage
                    )
                elif model_type == "elasticnet":
                    info = self.load_elasticnet_model(
                        ct,
                        stage
                    )
                elif model_type == "clm":
                    info = self.load_clm_model(
                        ct,
                        stage
                    )
                else:
                    raise ValueError(
                        "Unsupported model type: {}".format(model_type)
                    )

                models_info[
                    "{}_{}".format(
                        model_type,
                        stage
                    )
                ] = info

        self._models_cache[
            cache_key
        ] = models_info

        return models_info

    def _predict_one_model_in_chunks(
        self,
        adata,
        model_info,
        chunk_size=50000
    ):
        feature_genes = model_info[
            "features"
        ]

        predictions = []

        for start in range(
            0,
            adata.n_obs,
            chunk_size
        ):
            stop = min(
                start + chunk_size,
                adata.n_obs
            )

            X_chunk = _build_feature_matrix(
                adata[start:stop],
                feature_genes
            )

            if model_info["type"] == "elasticnet":
                fold_predictions = [
                    model.predict(
                        X_chunk
                    )
                    for model in model_info[
                        "models"
                    ]
                ]

            elif model_info["type"] == "transf":
                X_tensor = torch.from_numpy(
                    X_chunk
                ).float().to(
                    self.device
                )

                fold_predictions = []

                for model in model_info[
                    "models"
                ]:
                    with torch.no_grad():
                        fold_predictions.append(
                            model(
                                X_tensor
                            )
                            .detach()
                            .cpu()
                            .numpy()
                            .reshape(-1)
                        )

                del X_tensor

            elif model_info["type"] == "clm":
                X_tensor = torch.from_numpy(
                    X_chunk
                ).float().to(
                    self.device
                )

                fold_predictions = []

                for model, bin_centers in zip(
                    model_info[
                        "models"
                    ],
                    model_info[
                        "bin_centers_list"
                    ]
                ):
                    bin_tensor = torch.from_numpy(
                        bin_centers
                    ).float().to(
                        self.device
                    )

                    with torch.no_grad():
                        fold_predictions.append(
                            model.predict_age(
                                X_tensor,
                                bin_tensor
                            )
                            .detach()
                            .cpu()
                            .numpy()
                            .reshape(-1)
                        )

                del X_tensor

            else:
                raise ValueError(
                    "Unknown model type"
                )

            predictions.append(
                np.mean(
                    np.vstack(
                        fold_predictions
                    ),
                    axis=0
                )
            )

            del X_chunk

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if len(predictions) == 0:
            return np.array([], dtype=float)

        return np.concatenate(
            predictions
        )

    def predict_celltype_fast(
        self,
        adata_raw,
        ct,
        norm=True,
        chunk_size=50000,
        parallel=True
    ):
        """
        Apply all requested fixed pretrained scBAC component models to one cell type.

        ``parallel`` is retained in the call signature for compatibility with the
        original working code. Prediction is deliberately performed sequentially
        here to keep GPU/CPU memory use deterministic.
        """

        del parallel

        np.random.seed(42)
        torch.manual_seed(42)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        if norm:
            adata = _normalize_log1p(
                adata_raw
            )
        else:
            adata = adata_raw.copy()
            adata.X = adata.X.astype(
                "float32"
            )

        models_info = self.load_all_models_for_celltype(
            ct
        )

        result_df = adata.obs.copy()

        for model_name, model_info in models_info.items():
            print(
                "Predicting {} with fixed pretrained files...".format(
                    model_name
                )
            )

            result_df[
                model_name
            ] = self._predict_one_model_in_chunks(
                adata,
                model_info,
                chunk_size=chunk_size
            )

        result_df[
            "celltype"
        ] = ct

        result_df = add_ensemble_predictions(
            result_df
        )

        del adata
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result_df

    def predict_benchmarking_celltype(
        self,
        adata_raw,
        ct,
        norm=True
    ):
        """Apply benchmarking_model/sc_{ct}.csv to one cell type."""

        if norm:
            adata = _normalize_log1p(
                adata_raw
            )
        else:
            adata = adata_raw.copy()

        clock_path = os.path.join(
            self.benchmarking_dir,
            "sc_{}.csv".format(ct)
        )

        if not os.path.exists(clock_path):
            raise FileNotFoundError(clock_path)

        result = benchmarking_model(
            adata,
            clock_path
        )

        return result[
            "Benchmarking"
        ]

    def predict_celltype_with_benchmarking(
        self,
        adata_raw,
        ct,
        norm=True,
        chunk_size=50000,
        parallel=True
    ):
        """Apply fixed scBAC and the fixed benchmarking clock to the same cells."""

        result = self.predict_celltype_fast(
            adata_raw,
            ct,
            norm=norm,
            chunk_size=chunk_size,
            parallel=parallel
        )

        benchmark = self.predict_benchmarking_celltype(
            adata_raw,
            ct,
            norm=norm
        )

        result[
            "Benchmarking"
        ] = benchmark.loc[
            result.index
        ].values

        return result


def predict_all_celltypes(
    adata_obj,
    celltypes,
    predictor,
    norm=True,
    chunk_size=50000,
    include_benchmarking=True
):
    """Apply the fixed pretrained clocks to all requested cell types."""

    result_parts = []

    for ct in celltypes:
        print("\nCell type:", ct)

        subset = adata_obj[
            adata_obj.obs[
                "celltype"
            ].astype(str) == ct,
            :
        ].copy()

        if subset.n_obs == 0:
            print("  No cells. Skipping.")
            continue

        if include_benchmarking:
            result = predictor.predict_celltype_with_benchmarking(
                subset,
                ct,
                norm=norm,
                chunk_size=chunk_size,
                parallel=False
            )
        else:
            result = predictor.predict_celltype_fast(
                subset,
                ct,
                norm=norm,
                chunk_size=chunk_size,
                parallel=False
            )

        result_parts.append(
            result
        )

    if len(result_parts) == 0:
        return pd.DataFrame()

    return pd.concat(
        result_parts,
        axis=0
    )

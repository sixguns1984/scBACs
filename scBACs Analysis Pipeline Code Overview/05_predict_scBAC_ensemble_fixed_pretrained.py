"""
Apply the SUPPLIED FIXED PRETRAINED scBAC models to new single-cell data.

This is the public prediction script. It does not train, refit, recalibrate, or
select any model. It uses the exact pretrained model filenames used in the
original scBAC working prediction code, including the published benchmarking
clock files ``benchmarking_model/sc_{celltype}.csv``.

The output columns retain the original names:
- transf_Development / elasticnet_Development / clm_Development
- transf_Adult       / elasticnet_Adult       / clm_Adult
- transf_Full        / elasticnet_Full        / clm_Full
- Ensemble_Development / Ensemble_Adult / Ensemble_Full
- transf_adult_deve / elasticnet_adult_deve / clm_adult_deve
- Ensemble_adult_deve
- Benchmarking

Important
---------
Scripts 06 and the NDD/APOE4 downstream analyses use the supplied precomputed
metadata CSVs directly. They do not require this script to be rerun.

Python 3.9.20 compatible.
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

import os
import sys
import gc
import argparse

import pandas as pd
import scanpy as sc
import torch

import ensemble_brain_age_pred as eba_pred


# ============================================================================
# 2. PATHS AND PREDICTION SETTINGS
# ============================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset"
)

# The supplied fixed files are expected under:
# models/benchmarking_model/
# models/full_genes_predictions_20260415/
MODEL_ROOT = os.path.join(
    PROJECT_ROOT,
    "models"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "model_evaluation",
    "prediction_data"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Primary scBAC prediction cell types, matching the original main prediction code
# and the final manuscript clock analyses.
CELL_TYPES = [
    "Exc",
    "Inh",
    "Ast",
    "Oli",
    "OPC",
    "Mic"
]

CHUNK_SIZE = 50000

# For a user's own new dataset, edit these three variables.
# Leaving NEW_DATA_H5AD = None means only the known study files listed below
# are considered.
NEW_DATA_H5AD = None
NEW_DATA_NAME = "New_dataset"
NEW_DATA_NEEDS_NORMALIZATION = True


# ============================================================================
# OPTIONAL COMMAND-LINE PATH OVERRIDE
# ============================================================================

CLI_INPUT_H5AD = None
CLI_DATASET_NAME = None
CLI_NEEDS_NORMALIZATION = True

if __name__ == "__main__" and len(sys.argv) > 1:

    parser = argparse.ArgumentParser(
        description="Predict cellular age using the supplied fixed pretrained scBAC files."
    )

    parser.add_argument(
        "--input-h5ad",
        required=True
    )

    parser.add_argument(
        "--dataset-name",
        required=True
    )

    parser.add_argument(
        "--model-root",
        required=True,
        help=(
            "Directory containing benchmarking_model/ and "
            "full_genes_predictions_20260415/."
        )
    )

    parser.add_argument(
        "--output-dir",
        required=True
    )

    parser.add_argument(
        "--already-normalized",
        action="store_true",
        help="Use when input X is already on the log-normalized scale used by the clocks."
    )

    args = parser.parse_args()

    CLI_INPUT_H5AD = os.path.abspath(
        args.input_h5ad
    )

    CLI_DATASET_NAME = args.dataset_name
    MODEL_ROOT = os.path.abspath(
        args.model_root
    )
    RESULT_DIR = os.path.abspath(
        args.output_dir
    )
    CLI_NEEDS_NORMALIZATION = not args.already_normalized

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )


# ============================================================================
# 3. LOAD/PREPARE THE ORIGINAL VALIDATION AND DOWNSTREAM DATA RESOURCES
# ============================================================================

def prepare_frohlich(path):
    """Prepare GSE254569 exactly as in the original prediction working code."""

    adata = sc.read_h5ad(
        path
    )

    if "counts" in adata.layers:
        adata.X = adata.layers[
            "counts"
        ].copy()

    adata.obs[
        "celltype"
    ] = adata.obs[
        "major_celltypes"
    ].replace(
        [
            "Oligodendrocyte",
            "In_Neurons",
            "Exc_Neurons",
            "Astrocytes",
            "OPC",
            "Microglia",
            "Endothelial"
        ],
        [
            "Oli",
            "Inh",
            "Exc",
            "Ast",
            "OPC",
            "Mic",
            "End"
        ]
    )

    adata.obs[
        "dataset"
    ] = "GSE254569"

    adata.obs[
        "donor_id"
    ] = (
        "GSE254569_"
        + adata.obs[
            "Donor"
        ].astype(str)
    )

    adata.obs[
        "status"
    ] = adata.obs[
        "Disease_Status"
    ].map(
        {
            0: "CT"
        }
    ).fillna(
        "Disease"
    )

    adata.obs[
        "Age_at_death"
    ] = adata.obs[
        "Age"
    ]

    adata.obs[
        "analysis_group"
    ] = "Frohlich_et_al_adult"

    adata.obs[
        "sub_tissue"
    ] = "Orbitofrontal cortex"

    return adata


def prepare_jeffries_legacy_metadata(path):
    """
    Load the lifespan external cohort.

    The original supplied metadata uses the historical internal group label
    ``Alisa_et_al_full_life``; this label is retained for exact reproduction.
    """

    adata = sc.read_h5ad(
        path
    )

    adata.obs[
        "analysis_group"
    ] = "Alisa_et_al_full_life"

    if "dataset" not in adata.obs.columns:
        adata.obs[
            "dataset"
        ] = "Nature"

    return adata


def prepare_apoe4_replication(path):
    adata = sc.read_h5ad(
        path
    )

    adata.obs[
        "analysis_group"
    ] = "APOE4_aging_replication"

    return adata


def prepare_integrated_validation(path):
    adata = sc.read_h5ad(
        path
    )

    adata.obs[
        "analysis_group"
    ] = "Integrated_Original"

    return adata


def prepare_training_resource(old_path, new_path):
    old = sc.read_h5ad(
        old_path
    )

    new = sc.read_h5ad(
        new_path
    )

    old.obs[
        "analysis_group"
    ] = "Training"

    new.obs[
        "analysis_group"
    ] = "Training"

    return sc.concat(
        [
            old,
            new
        ],
        axis=0,
        join="inner",
        merge="same",
        index_unique=None
    )


def prepare_seaad(path):
    """Prepare SEAAD metadata fields used in the original working prediction code."""

    adata = sc.read_h5ad(
        path
    )

    columns = [
        "sample_id",
        "Donor ID",
        "Brain Region",
        "Sex",
        "Age at Death",
        "Years of education",
        "PMI",
        "Thal",
        "Braak",
        "CERAD score",
        "Overall AD neuropathological Change",
        "Overall CAA Score",
        "Cognitive Status",
        "Last MMSE Score",
        "Interval from last MMSE in months",
        "Last MOCA Score",
        "Interval from last MOCA in months",
        "APOE Genotype",
        "Class",
        "Subclass",
        "Supertype"
    ]

    available_columns = [
        col
        for col in columns
        if col in adata.obs.columns
    ]

    adata.obs = adata.obs.loc[
        :,
        available_columns
    ].copy()

    rename_map = {
        "Donor ID": "donor_id",
        "Brain Region": "sub_tissue",
        "Age at Death": "Age_at_Death",
        "Braak": "Braak_stage",
        "Last MMSE Score": "MMSE",
        "Last MOCA Score": "MOCA"
    }

    adata.obs.rename(
        columns=rename_map,
        inplace=True
    )

    adata.obs[
        "dataset"
    ] = "SEAAD"

    adata.obs[
        "Age_at_death"
    ] = adata.obs[
        "Age_at_Death"
    ]

    adata.obs[
        "sub_tissue"
    ] = "Prefrontal cortex"

    adata.obs[
        "celltype"
    ] = adata.obs[
        "Subclass"
    ].astype(str)

    if "Overall AD neuropathological Change" in adata.obs.columns:
        adata.obs[
            "status"
        ] = adata.obs[
            "Overall AD neuropathological Change"
        ].replace(
            [
                "Not AD",
                "Low",
                "Intermediate",
                "High"
            ],
            [
                "CT",
                "AD",
                "AD",
                "AD"
            ]
        )

    if "Class" in adata.obs.columns:
        adata.obs.loc[
            adata.obs[
                "Class"
            ] == "Neuronal: Glutamatergic",
            "celltype"
        ] = "Exc"

        adata.obs.loc[
            adata.obs[
                "Class"
            ] == "Neuronal: GABAergic",
            "celltype"
        ] = "Inh"

    adata.obs[
        "celltype"
    ] = adata.obs[
        "celltype"
    ].replace(
        [
            "Oligodendrocyte",
            "OPC",
            "Astrocyte",
            "Microglia-PVM",
            "Endothelial"
        ],
        [
            "Oli",
            "OPC",
            "Ast",
            "Mic",
            "End"
        ]
    )

    adata.obs[
        "analysis_group"
    ] = "SEAAD"

    return adata


# ============================================================================
# 4. BUILD THE DATASET LIST
# ============================================================================

if CLI_INPUT_H5AD is not None:

    DATASETS_TO_PREDICT = [
        {
            "name": CLI_DATASET_NAME,
            "kind": "generic",
            "path": CLI_INPUT_H5AD,
            "normalize": CLI_NEEDS_NORMALIZATION
        }
    ]

else:

    DATASETS_TO_PREDICT = [
        {
            "name": "Frohlich_et_al_adult",
            "kind": "frohlich",
            "path": os.path.join(
                DATA_DIR,
                "GSE254569_adata_RNA.h5ad"
            ),
            "normalize": True
        },
        {
            "name": "Alisa_et_al_full_life",
            "kind": "jeffries",
            "path": os.path.join(
                DATA_DIR,
                "sce_scBACs_external_validation.h5ad"
            ),
            "normalize": True
        },
        {
            "name": "APOE4_aging_replication",
            "kind": "apoe4",
            "path": os.path.join(
                DATA_DIR,
                "sce_APOE4_aging_replication.h5ad"
            ),
            "normalize": True
        },
        {
            "name": "Integrated_Original",
            "kind": "integrated",
            "path": os.path.join(
                DATA_DIR,
                "sce_test.h5ad"
            ),
            "normalize": True
        },
        {
            "name": "Training",
            "kind": "training",
            "path": os.path.join(
                DATA_DIR,
                "sce_train.h5ad"
            ),
            "new_training_path": os.path.join(
                DATA_DIR,
                "GSE291605_PFC.h5ad"
            ),
            "normalize": True
        },
        {
            "name": "SEAAD",
            "kind": "seaad",
            "path": os.path.join(
                DATA_DIR,
                "SEAAD_A9_RNAseq_final-nuclei.2024-02-13.h5ad"
            ),
            # Original working prediction code used norm=False for SEAAD.
            "normalize": False
        }
    ]

    if NEW_DATA_H5AD is not None:
        DATASETS_TO_PREDICT.append(
            {
                "name": NEW_DATA_NAME,
                "kind": "generic",
                "path": NEW_DATA_H5AD,
                "normalize": NEW_DATA_NEEDS_NORMALIZATION
            }
        )


# ============================================================================
# 5. INITIALIZE THE FIXED PRETRAINED PREDICTOR
# ============================================================================

predictor = eba_pred.UnifiedAgePredictor(
    DEVICE,
    max_workers=4,
    model_root=MODEL_ROOT,
    strict_five_folds=True
)


# ============================================================================
# 6. PREDICT EACH DATASET
# ============================================================================

prediction_summary = []

for dataset_info in DATASETS_TO_PREDICT:

    dataset_name = dataset_info[
        "name"
    ]

    dataset_path = dataset_info[
        "path"
    ]

    dataset_kind = dataset_info[
        "kind"
    ]

    normalize_input = dataset_info[
        "normalize"
    ]

    if not os.path.exists(
        dataset_path
    ):
        print(
            "Skipping missing dataset:",
            dataset_path
        )
        continue

    print("\n" + "=" * 70)
    print("Predicting dataset:", dataset_name)
    print("=" * 70)

    if dataset_kind == "frohlich":
        adata = prepare_frohlich(
            dataset_path
        )

    elif dataset_kind == "jeffries":
        adata = prepare_jeffries_legacy_metadata(
            dataset_path
        )

    elif dataset_kind == "apoe4":
        adata = prepare_apoe4_replication(
            dataset_path
        )

    elif dataset_kind == "integrated":
        adata = prepare_integrated_validation(
            dataset_path
        )

    elif dataset_kind == "training":
        new_training_path = dataset_info[
            "new_training_path"
        ]

        if not os.path.exists(
            new_training_path
        ):
            print(
                "Skipping Training prediction because the 2025 training file is missing:",
                new_training_path
            )
            continue

        adata = prepare_training_resource(
            dataset_path,
            new_training_path
        )

    elif dataset_kind == "seaad":
        adata = prepare_seaad(
            dataset_path
        )

    else:
        adata = sc.read_h5ad(
            dataset_path
        )

    if "celltype" not in adata.obs.columns:
        raise KeyError(
            "{} does not contain adata.obs['celltype'].".format(
                dataset_name
            )
        )

    prediction = eba_pred.predict_all_celltypes(
        adata_obj=adata,
        celltypes=CELL_TYPES,
        predictor=predictor,
        norm=normalize_input,
        chunk_size=CHUNK_SIZE,
        include_benchmarking=True
    )

    if prediction.empty:
        print(
            "No predictions generated for:",
            dataset_name
        )
        continue

    # Original combined prediction filename.
    combined_output = os.path.join(
        RESULT_DIR,
        "{}_All_celltypes_full_life_genes_Models_Predictions.csv".format(
            dataset_name
        )
    )

    prediction.to_csv(
        combined_output,
        index=True
    )

    # Original benchmarking-only filename is also retained.
    benchmark_columns = [
        col
        for col in prediction.columns
        if col not in [
            "transf_Development",
            "transf_Adult",
            "transf_Full",
            "elasticnet_Development",
            "elasticnet_Adult",
            "elasticnet_Full",
            "clm_Development",
            "clm_Adult",
            "clm_Full",
            "Ensemble_Development",
            "Ensemble_Adult",
            "Ensemble_Full",
            "transf_adult_deve",
            "elasticnet_adult_deve",
            "clm_adult_deve",
            "Ensemble_adult_deve"
        ]
    ]

    benchmark_output = os.path.join(
        RESULT_DIR,
        "{}_Benchmarking_Predictions.csv".format(
            dataset_name
        )
    )

    prediction.loc[
        :,
        benchmark_columns
    ].to_csv(
        benchmark_output,
        index=True
    )

    prediction_summary.append(
        {
            "dataset": dataset_name,
            "n_cells": prediction.shape[0],
            "combined_prediction_file": combined_output,
            "benchmarking_prediction_file": benchmark_output,
            "normalized_before_prediction": normalize_input
        }
    )

    print(
        "Saved:",
        combined_output
    )

    del adata
    del prediction
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# 7. SAVE PREDICTION SUMMARY
# ============================================================================

prediction_summary = pd.DataFrame(
    prediction_summary
)

prediction_summary.to_csv(
    os.path.join(
        RESULT_DIR,
        "prediction_file_summary.csv"
    ),
    index=False
)

print("\nAll requested fixed-model predictions completed.")
print(prediction_summary)


# ============================================================================
# 8. COMMAND-LINE USAGE
# ============================================================================

"""
For a new h5ad file:

python 05_predict_scBAC_ensemble.py \
    --input-h5ad /path/to/new_data.h5ad \
    --dataset-name New_dataset \
    --model-root /path/to/models \
    --output-dir /path/to/results/prediction_data

If the input expression matrix is already log-normalized on the scale expected
by the clocks, add:

    --already-normalized
"""

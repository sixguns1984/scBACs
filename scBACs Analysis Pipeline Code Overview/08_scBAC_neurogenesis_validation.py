"""
Biological-sensitivity validation using adult human hippocampal neurogenesis.

Dataset:
- Disouky et al. adult human hippocampal multi-omic atlas (GSE268609)

Analysis:
1. retain the original neuronal annotations from the published dataset;
2. apply the fixed pretrained development/adult-routed Exc scBAC and the fixed
   Muralidharan Exc benchmarking clock to the same neuronal population;
3. apply the fixed unified full-lifespan scBAC comparison model;
4. calculate cell-level relative age acceleration (RAA) from a control-derived
   model: predicted age ~ chronological age + sex + donor cell count;
5. calculate clock-specific q5/q25/q75/q95 RAA thresholds from control neurons;
6. calculate the fraction of rejuvenated-state and accelerated-aging-state
   neurons for each donor;
7. define newborn neurons using the original immature-granule-neuron label and
   calculate their donor-level proportion;
8. correlate newborn-neuron proportion with the clock-defined neuronal-state
   fractions using Spearman correlation.

The old exploratory working script also used a broader 'Neurogenic' metadata
label. For transparency, this script calculates that legacy ratio when the
column exists, but the primary analysis uses the final manuscript definition:
immature granule neurons / total analyzed neuronal nuclei.
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import torch

from scipy.stats import spearmanr

import ensemble_brain_age_pred as eba_pred


# ============================================================================
# 2. PATHS AND ANALYSIS SETTINGS
# ============================================================================
__file__ = ''
PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset"
)

MODEL_ROOT = os.path.join(
    PROJECT_ROOT,
    "models"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "neurogenesis_validation"
)

FIGURE_DIR = os.path.join(
    OUTPUT_DIR,
    "figures"
)

RAA_MODEL_DIR = os.path.join(
    OUTPUT_DIR,
    "RAA_models"
)

for directory in [
    OUTPUT_DIR,
    FIGURE_DIR,
    RAA_MODEL_DIR
]:
    os.makedirs(
        directory,
        exist_ok=True
    )

HIPPOCAMPUS_H5AD = os.path.join(
    DATA_DIR,
    "GSE268609_hipocampus.h5ad"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AGE_CUT = 18

# Original Disouky metadata columns used in the working analysis
DONOR_COL = "orig.ident"
GROUP_COL = "Group"
SEX_COL = "Sex"
AGE_SOURCE_COL = "Age at Death"
CLUSTER_COL = "Cluster"

# Healthy agers and young adults define the control RAA reference distribution.
CONTROL_GROUPS = [
    "HA",
    "YA"
]

# The same neuronal lineage population receives the Exc clock.
# These are the original labels present in the working GSE268609 file.
ANALYZED_NEURON_LABELS = [
    "CA_neurons",
    "GABA_neurons",
    "Immature",
    "CA2-4_neurons",
    "Neuroblast",
    "NSC"
]

# Final manuscript definition: immature granule neurons are treated as newborn.
# The released file used the shorter label "Immature" in the original working
# analysis. Add/remove names here if the public annotation spelling differs.
NEWBORN_NEURON_LABELS = [
    "Immature",
    "Immature granule neurons",
    "Immature granule neuron"
]

CLOCK_COLUMNS = [
    "Muralidharan_age",
    "scBAC_age",
    "scBAC_full"
]

QUANTILES = [
    0.05,
    0.25,
    0.75,
    0.95
]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Helvetica"
]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
sns.set_style("ticks")


# ============================================================================
# OPTIONAL COMMAND-LINE PATH OVERRIDE
# ============================================================================

if __name__ == "__main__" and len(sys.argv) > 1:
    parser = argparse.ArgumentParser(
        description="Validate scBAC biological sensitivity using hippocampal neurogenesis."
    )

    parser.add_argument(
        "--hippocampus-h5ad",
        required=True
    )

    parser.add_argument(
        "--model-root",
        required=True,
        help="Root containing full_genes_predictions_20260415/ and benchmarking_model/."
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

    HIPPOCAMPUS_H5AD = args.hippocampus_h5ad
    MODEL_ROOT = os.path.abspath(
        args.model_root
    )
    OUTPUT_DIR = os.path.abspath(
        args.output_dir
    )

    FIGURE_DIR = os.path.join(
        OUTPUT_DIR,
        "figures"
    )

    RAA_MODEL_DIR = os.path.join(
        OUTPUT_DIR,
        "RAA_models"
    )

    DEVICE = args.device

    for directory in [
        OUTPUT_DIR,
        FIGURE_DIR,
        RAA_MODEL_DIR
    ]:
        os.makedirs(
            directory,
            exist_ok=True
        )


# ============================================================================
# 3. RAA FUNCTIONS
# ============================================================================


def encode_sex(series):
    """Fixed sex coding used by the RAA model: Female=0, Male=1."""
    mapping = {
        "Female": 0,
        "female": 0,
        "F": 0,
        "f": 0,
        "Male": 1,
        "male": 1,
        "M": 1,
        "m": 1
    }
    return series.map(
        mapping
    )


def fit_control_raa_model(
    cell_df,
    predicted_age_col,
    control_groups,
    group_col,
    donor_col,
    age_col,
    sex_col,
    model_name
):
    """
    Fit the final manuscript cell-level RAA reference model using controls only:
        predicted cell age ~ chronological age + sex + donor cell count
    """
    data = cell_df.copy()
    # Cell count is calculated within this analyzed neuronal population.
    donor_cell_count = (
        data.groupby(
            donor_col
        )
        .size()
        .rename(
            "donor_cell_count"
        )
    )
    data["donor_cell_count"] = data[
        donor_col
    ].map(
        donor_cell_count
    )
    data["sex_encoded"] = encode_sex(
        data[sex_col]
    )
    control_mask = data[
        group_col
    ].isin(
        control_groups
    )
    control_data = data.loc[
        control_mask,
        :
    ].copy()
    required = [
        predicted_age_col,
        age_col,
        "sex_encoded",
        "donor_cell_count"
    ]
    control_clean = control_data.dropna(
        subset=required
    ).copy()
    if control_clean.shape[0] < 10:
        raise ValueError(
            "Too few control cells for RAA model: {}".format(
                control_clean.shape[0]
            )
        )
    X_control = control_clean[
        [
            age_col,
            "sex_encoded",
            "donor_cell_count"
        ]
    ].astype(float)
    X_control = sm.add_constant(
        X_control,
        has_constant="add"
    )
    y_control = pd.to_numeric(
        control_clean[predicted_age_col],
        errors="coerce"
    ).values
    model = sm.OLS(
        y_control,
        X_control
    ).fit()
    print("\nRAA model:", model_name)
    print(
        "  Control cells:",
        control_clean.shape[0]
    )
    print(
        "  Control donors:",
        control_clean[donor_col].nunique()
    )
    print(
        "  R2:",
        model.rsquared
    )
    print(
        "  Coefficients:"
    )
    print(
        model.params
    )
    # Apply the fixed control coefficients to all analyzed cells
    valid = data[
        required
    ].notna().all(
        axis=1
    )
    X_all = data.loc[
        valid,
        [
            age_col,
            "sex_encoded",
            "donor_cell_count"
        ]
    ].astype(float)
    X_all = sm.add_constant(
        X_all,
        has_constant="add"
    )
    expected_age = model.predict(
        X_all
    )
    raa = np.full(
        data.shape[0],
        np.nan
    )
    valid_position = np.where(
        valid.values
    )[0]
    observed_age = pd.to_numeric(
        data.loc[
            valid,
            predicted_age_col
        ],
        errors="coerce"
    ).values
    raa[
        valid_position
    ] = (
        observed_age
        - expected_age
    )
    # Save coefficients rather than a version-dependent statsmodels object
    model_info = {
        "model_name": model_name,
        "predicted_age_col": predicted_age_col,
        "control_groups": list(control_groups),
        "formula": "predicted_age ~ chronological_age + sex + donor_cell_count",
        "sex_encoding": "Female=0, Male=1",
        "coefficients": {
            key: float(value)
            for key, value in model.params.to_dict().items()
        },
        "r_squared": float(model.rsquared),
        "n_control_cells": int(control_clean.shape[0]),
        "n_control_donors": int(control_clean[donor_col].nunique())
    }
    with open(
        os.path.join(
            RAA_MODEL_DIR,
            "{}_RAA_model.json".format(
                model_name
            )
        ),
        "w"
    ) as handle:
        json.dump(
            model_info,
            handle,
            indent=2
        )
    return data, raa, model_info


def calculate_control_thresholds(
    cell_df,
    raa_col,
    control_groups,
    group_col
):
    """Calculate q5, q25, q75 and q95 from the corresponding control RAA."""
    control_values = cell_df.loc[
        cell_df[group_col].isin(
            control_groups
        ),
        raa_col
    ].dropna()
    thresholds = {
        "q5": float(
            control_values.quantile(
                0.05
            )
        ),
        "q25": float(
            control_values.quantile(
                0.25
            )
        ),
        "q75": float(
            control_values.quantile(
                0.75
            )
        ),
        "q95": float(
            control_values.quantile(
                0.95
            )
        ),
        "n_control_cells": int(
            control_values.shape[0]
        )
    }
    return thresholds


# ============================================================================
# 4. LOAD THE DISOUKY HIPPOCAMPAL DATASET
# ============================================================================

print("\n" + "=" * 70)
print("Loading Disouky hippocampal dataset")
print("=" * 70)

sce = sc.read_h5ad(
    HIPPOCAMPUS_H5AD
)

print(
    "Full data shape:",
    sce.shape
)

print(
    "Original cluster labels:"
)

print(
    sce.obs[CLUSTER_COL].value_counts()
)


# ============================================================================
# 5. RETAIN THE ORIGINAL NEURONAL POPULATION
# ============================================================================

neuron_mask = sce.obs[
    CLUSTER_COL
].isin(
    ANALYZED_NEURON_LABELS
)

sce_neuron = sce[
    neuron_mask,
    :
].copy()

# Original annotations are retained; we only add the model lineage label needed
# to apply the excitatory-neuron brain-age clock.
sce_neuron.obs["celltype"] = "Exc"

sce_neuron.obs["donor_id"] = sce_neuron.obs[
    DONOR_COL
].astype(str)

sce_neuron.obs["Age_at_death"] = pd.to_numeric(
    sce_neuron.obs[
        AGE_SOURCE_COL
    ],
    errors="coerce"
)

print(
    "Analyzed neuronal nuclei:",
    sce_neuron.n_obs
)

print(
    "Donors:",
    sce_neuron.obs[DONOR_COL].nunique()
)

print(
    "Groups:"
)

print(
    sce_neuron.obs[GROUP_COL].value_counts()
)


# ============================================================================
# 6. PREDICT NEURONAL AGE WITH THE FIXED PRETRAINED SCBAC EXC CLOCK
# ============================================================================

print("\n" + "=" * 70)
print("Predicting neuronal age with fixed pretrained scBAC")
print("=" * 70)

predictor = eba_pred.UnifiedAgePredictor(
    DEVICE,
    max_workers=4,
    model_root=MODEL_ROOT,
    strict_five_folds=True
)

fixed_predictions = eba_pred.predict_all_celltypes(
    adata_obj=sce_neuron,
    celltypes=["Exc"],
    predictor=predictor,
    norm=True,
    chunk_size=5000,
    include_benchmarking=True
)

if fixed_predictions.empty:
    raise RuntimeError(
        "No fixed pretrained predictions were generated for the hippocampal neurons."
    )

neuron_results = fixed_predictions.copy()

# Keep explicit original metadata columns used below.
for col in [
    DONOR_COL,
    GROUP_COL,
    SEX_COL,
    AGE_SOURCE_COL,
    CLUSTER_COL,
    "Neurogenic"
]:
    if col in sce_neuron.obs.columns:
        neuron_results.loc[
            sce_neuron.obs_names,
            col
        ] = sce_neuron.obs[
            col
        ].values

neuron_results.loc[
    sce_neuron.obs_names,
    "Age_at_death"
] = sce_neuron.obs[
    "Age_at_death"
].values

neuron_results["celltype"] = "Exc"

# Preserve the downstream column names used by the original neurogenesis script.
# Ensemble_adult_deve is the stage-routed scBAC prediction generated by the
# current fixed prediction code; Ensemble_Full is retained as the comparison.
required_prediction_columns = [
    "Ensemble_adult_deve",
    "Ensemble_Full",
    "Benchmarking"
]

missing_prediction_columns = [
    col
    for col in required_prediction_columns
    if col not in neuron_results.columns
]

if len(missing_prediction_columns) > 0:
    raise KeyError(
        "Missing required fixed-prediction columns: {}".format(
            missing_prediction_columns
        )
    )

neuron_results["scBAC_age"] = pd.to_numeric(
    neuron_results["Ensemble_adult_deve"],
    errors="coerce"
)

neuron_results["scBAC_full"] = pd.to_numeric(
    neuron_results["Ensemble_Full"],
    errors="coerce"
)

neuron_results["Muralidharan_age"] = pd.to_numeric(
    neuron_results["Benchmarking"],
    errors="coerce"
)


# ============================================================================
# 7. FIXED MURALIDHARAN BENCHMARKING CLOCK
# ============================================================================

print("\n" + "=" * 70)
print("Muralidharan Exc age loaded from fixed benchmarking_model/sc_Exc.csv")
print("=" * 70)


# ============================================================================
# 8. CALCULATE CELL-LEVEL RAA FOR EACH CLOCK
# ============================================================================

print("\n" + "=" * 70)
print("Calculating control-referenced RAA")
print("=" * 70)

clock_column_map = {
    "Muralidharan": "Muralidharan_age",
    "scBAC_Adult": "scBAC_age"
}

if neuron_results["scBAC_full"].notna().sum() > 0:
    clock_column_map[
        "scBAC_Full"
    ] = "scBAC_full"

raa_model_summary = []
threshold_rows = []

for clock_name, prediction_col in clock_column_map.items():
    neuron_results, raa_values, model_info = fit_control_raa_model(
        cell_df=neuron_results,
        predicted_age_col=prediction_col,
        control_groups=CONTROL_GROUPS,
        group_col=GROUP_COL,
        donor_col=DONOR_COL,
        age_col="Age_at_death",
        sex_col=SEX_COL,
        model_name=clock_name
    )
    raa_col = "{}_RAA".format(
        clock_name
    )
    neuron_results[
        raa_col
    ] = raa_values
    thresholds = calculate_control_thresholds(
        neuron_results,
        raa_col=raa_col,
        control_groups=CONTROL_GROUPS,
        group_col=GROUP_COL
    )
    for threshold_name, threshold_value in thresholds.items():
        if threshold_name == "n_control_cells":
            continue
        threshold_rows.append(
            {
                "clock": clock_name,
                "threshold": threshold_name,
                "value": threshold_value,
                "n_control_cells": thresholds["n_control_cells"]
            }
        )
    # Cell-state assignments
    neuron_results[
        "{}_rejuvenated_q5".format(clock_name)
    ] = (
        neuron_results[raa_col]
        < thresholds["q5"]
    ).astype(float)
    neuron_results[
        "{}_rejuvenated_q25".format(clock_name)
    ] = (
        neuron_results[raa_col]
        < thresholds["q25"]
    ).astype(float)
    neuron_results[
        "{}_accelerated_q75".format(clock_name)
    ] = (
        neuron_results[raa_col]
        > thresholds["q75"]
    ).astype(float)
    neuron_results[
        "{}_accelerated_q95".format(clock_name)
    ] = (
        neuron_results[raa_col]
        > thresholds["q95"]
    ).astype(float)
    # Preserve missing-state status when RAA could not be calculated
    missing_raa = neuron_results[
        raa_col
    ].isna()
    for suffix in [
        "rejuvenated_q5",
        "rejuvenated_q25",
        "accelerated_q75",
        "accelerated_q95"
    ]:
        neuron_results.loc[
            missing_raa,
            "{}_{}".format(
                clock_name,
                suffix
            )
        ] = np.nan
    raa_model_summary.append(
        model_info
    )


threshold_table = pd.DataFrame(
    threshold_rows
)

threshold_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "control_RAA_thresholds.csv"
    ),
    index=False
)


# ============================================================================
# 9. CALCULATE DONOR-LEVEL NEWBORN-NEURON PROPORTION
# ============================================================================
neuron_results["is_neurogenic"] = (
    neuron_results[NEUROGENIC_COL].astype(str) == NEUROGENIC_VALUE
).astype(int)

newborn_summary = (
    neuron_results.groupby(
        DONOR_COL,
        observed=True
    )
    .agg(
        Neurogenic_count=("is_neurogenic", "sum"),
        total_neuronal_nuclei=("is_neurogenic", "size"),
        Neurogenic_Ratio=("is_neurogenic", "mean"),
        Group=(GROUP_COL, "first"),
        Sex=(SEX_COL, "first"),
        Age_at_death=("Age_at_death", "first")
    )
    .reset_index()
)

# Keep a descriptive alias for downstream/public outputs, but it is defined
# exclusively by the original Neurogenic annotation above.
newborn_summary["newborn_neuron_proportion"] = newborn_summary["Neurogenic_Ratio"]
newborn_summary["newborn_neuron_count"] = newborn_summary["Neurogenic_count"]

print(
    newborn_summary[
        [
            DONOR_COL,
            "Group",
            "Neurogenic_count",
            "total_neuronal_nuclei",
            "Neurogenic_Ratio"
        ]
    ]
)



# ============================================================================
# 10. CALCULATE DONOR-LEVEL AGING-STATE FRACTIONS
# ============================================================================

state_columns = []

for clock_name in clock_column_map.keys():
    state_columns.extend(
        [
            "{}_rejuvenated_q5".format(clock_name),
            "{}_rejuvenated_q25".format(clock_name),
            "{}_accelerated_q75".format(clock_name),
            "{}_accelerated_q95".format(clock_name)
        ]
    )

state_fraction = (
    neuron_results.groupby(
        DONOR_COL,
        observed=True
    )[state_columns]
    .mean()
    .reset_index()
)

# Mean of 0/1 cell-state labels is the donor-level state fraction.
state_fraction = state_fraction.rename(
    columns={
        col: "{}_fraction".format(col)
        for col in state_columns
    }
)

neurogenesis_donor_table = newborn_summary.merge(
    state_fraction,
    on=DONOR_COL,
    how="inner"
)

neurogenesis_donor_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "donor_newborn_neuron_and_aging_state_fractions.csv"
    ),
    index=False
)


# ============================================================================
# 11. SPEARMAN CORRELATIONS WITH NEWBORN-NEURON PROPORTION
# ============================================================================

print("\n" + "=" * 70)
print("Donor-level correlations with newborn-neuron proportion")
print("=" * 70)

correlation_rows = []

for clock_name in clock_column_map.keys():
    for state_name in [
        "rejuvenated_q5",
        "rejuvenated_q25",
        "accelerated_q75",
        "accelerated_q95"
    ]:
        fraction_col = "{}_{}_fraction".format(
            clock_name,
            state_name
        )
        valid = neurogenesis_donor_table[
            [
                "Neurogenic_Ratio",
                fraction_col
            ]
        ].dropna()
        if valid.shape[0] >= 3:
            rho, p_value = spearmanr(
                valid["Neurogenic_Ratio"],
                valid[fraction_col]
            )
        else:
            rho = np.nan
            p_value = np.nan
        correlation_rows.append(
            {
                "clock": clock_name,
                "state_definition": state_name,
                "n_donors": valid.shape[0],
                "spearman_rho": rho,
                "p_value": p_value
            }
        )

neurogenesis_correlations = pd.DataFrame(
    correlation_rows
)

neurogenesis_correlations.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "newborn_neuron_state_fraction_correlations.csv"
    ),
    index=False
)

print(
    neurogenesis_correlations
)


# ============================================================================
# 12. SAVE CELL-LEVEL PREDICTIONS AND RAA
# ============================================================================

neuron_results.index.name = "cell_id"

neuron_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "GSE268609_neuronal_cell_age_and_RAA.csv.gz"
    ),
    compression="gzip"
)


# ============================================================================
# 13. PLOT THE TWO THRESHOLD SCHEMES
# ============================================================================


def plot_threshold_scheme(
    donor_df,
    clock_names,
    lower_state,
    upper_state,
    output_file
):
    """Plot newborn-neuron proportion against younger/older neuronal fractions."""

    n_rows = len(clock_names)

    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(8, 3.2 * n_rows),
        squeeze=False
    )

    for row_index, clock_name in enumerate(
        clock_names
    ):

        for col_index, state_name in enumerate(
            [
                lower_state,
                upper_state
            ]
        ):

            ax = axes[
                row_index,
                col_index
            ]

            fraction_col = "{}_{}_fraction".format(
                clock_name,
                state_name
            )

            sub = donor_df[
                [
                    fraction_col,
                    "Neurogenic_Ratio",
                    "Group"
                ]
            ].dropna()

            if sub.shape[0] >= 3:
                rho, p_value = spearmanr(
                    sub[fraction_col],
                    sub["Neurogenic_Ratio"]
                )
            else:
                rho = np.nan
                p_value = np.nan

            sns.regplot(
                data=sub,
                x=fraction_col,
                y="Neurogenic_Ratio",
                scatter_kws={
                    "s": 28,
                    "alpha": 0.75
                },
                line_kws={
                    "linewidth": 1.2
                },
                ax=ax
            )

            ax.set_title(
                "{} | {}\nr={:.3f}, P={:.3g}".format(
                    clock_name,
                    state_name,
                    rho,
                    p_value
                ),
                fontsize=9
            )

            ax.set_xlabel(
                "Cell-state fraction"
            )

            ax.set_ylabel(
                "Newborn-neuron proportion"
                if col_index == 0
                else ""
            )

            sns.despine(
                ax=ax
            )

    plt.tight_layout()

    plt.savefig(
        output_file,
        bbox_inches="tight",
        dpi=600
    )

    plt.close()


# Stringent 5th/95th-percentile state definitions
plot_threshold_scheme(
    neurogenesis_donor_table,
    list(clock_column_map.keys()),
    lower_state="rejuvenated_q5",
    upper_state="accelerated_q95",
    output_file=os.path.join(
        FIGURE_DIR,
        "neurogenesis_correlations_q5_q95.pdf"
    )
)

# Broader 25th/75th-percentile state definitions
plot_threshold_scheme(
    neurogenesis_donor_table,
    list(clock_column_map.keys()),
    lower_state="rejuvenated_q25",
    upper_state="accelerated_q75",
    output_file=os.path.join(
        FIGURE_DIR,
        "neurogenesis_correlations_q25_q75.pdf"
    )
)


# ============================================================================
# 14. COMMAND-LINE USAGE
# ============================================================================

"""
python 08_scBAC_neurogenesis_validation.py \
    --hippocampus-h5ad /path/to/GSE268609_hipocampus.h5ad \
    --model-root /path/to/models \
    --output-dir /path/to/results/neurogenesis_validation \
    --device cuda
"""
